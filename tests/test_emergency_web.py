"""Web-level tests: concurrent starts, emergency preemption, pause-while-yield,
emergency failure auto-resume and restart recovery.

A monkeypatched ``engine.read_chunk`` gate makes the "job is inside its current
read segment" state deterministic, so assertions about when the yield lands do
not depend on timing or large files.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app, worker_running  # noqa: E402
from trainer import engine  # noqa: E402

CSV = "measured,reference\n" + "".join(
    f"{i * 0.5 - 20:.4f},{1.5 * (i * 0.5 - 20) + 0.75:.4f}\n" for i in range(48)
)


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.02):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s (last={last!r})")


class ChunkGate:
    """Block the first read_chunk call of selected worker threads."""

    def __init__(self, thread_names):
        self.names = set(thread_names)
        self.entered = {n: threading.Event() for n in thread_names}
        self.release = {n: threading.Event() for n in thread_names}
        self._orig = engine.read_chunk

    def __enter__(self):
        orig = self._orig
        gate = self

        def wrapped(fh, offset, want, w, b, part, chunk_limit):
            name = threading.current_thread().name
            if name in gate.names:
                gate.names.discard(name)  # one-shot gate
                gate.entered[name].set()
                gate.release[name].wait(15)
            return orig(fh, offset, want, w, b, part, chunk_limit)

        engine.read_chunk = wrapped
        return self

    def __exit__(self, *_exc):
        engine.read_chunk = self._orig
        for event in self.release.values():
            event.set()
        return False

    def wait_entered(self, name: str):
        if not self.entered[name].wait(15):
            raise AssertionError(f"{name} did not enter read_chunk")

    def free(self, name: str):
        self.release[name].set()


class EmergencyWebBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="emergency-web-")
        self.app = create_app({
            "DB_PATH": os.path.join(self.tmp, "db.sqlite3"),
            "UPLOAD_DIR": os.path.join(self.tmp, "uploads"),
            "SECRET_KEY": "test",
            "TESTING": True,
        })
        self.client = self.app.test_client()
        self._run_ids: list[int] = []

    def tearDown(self) -> None:
        coord = self.app.extensions.get("emergency_coordinator")
        if coord is not None:
            deadline = time.time() + 10
            while coord.active and time.time() < deadline:
                time.sleep(0.02)
        for rid in self._run_ids:
            deadline = time.time() + 10
            while worker_running(rid) and time.time() < deadline:
                time.sleep(0.02)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload_named(self, name: str = "cal.csv", csv: str = CSV, **form) -> int:
        data = {
            "csv": (io.BytesIO(csv.encode()), name),
            "group_size": "6",
            "chunk_limit": "6",
            "lr": "0.02",
            "momentum": "0.9",
            "init_w": "0",
            "init_b": "0",
        }
        data.update({k: str(v) for k, v in form.items()})
        resp = self.client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 302)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            rid = engine.list_runs(conn)[0]["id"]
        self._run_ids.append(rid)
        return rid

    def row(self, rid: int):
        with engine.session(self.app.config["DB_PATH"]) as conn:
            return dict(engine.get_run(conn, rid))

    def start(self, rid: int, **data):
        payload = {"pause_after": "", "chunk_limit": "6"}
        payload.update(data)
        return self.client.post(f"/runs/{rid}/start", data=payload)

    def wait_status(self, rid: int, status: str):
        return wait_for(lambda: self.row(rid)["status"] == status and self.row(rid))

    def wait_idle(self, rid: int):
        def idle():
            row = self.row(rid)
            return (not worker_running(rid) and row["status"] != "running" and row) or None
        return wait_for(idle)

    def wait_emergency_inactive(self):
        coord = self.app.extensions["emergency_coordinator"]
        wait_for(lambda: (not coord.active) or None, timeout=20)

    def assert_exact_and_audit(self, rid: int, **ref_kw):
        row = self.row(rid)
        self.assertEqual(row["status"], "finished")
        audit = self.client.get(f"/api/runs/{rid}/audit").get_json()
        self.assertTrue(audit["ok"], audit["problems"])
        params = dict(group_size=row["group_size"], chunk_limit=6,
                      lr=row["lr"], momentum=row["momentum"],
                      init_w=0.0, init_b=0.0)
        params.update(ref_kw)
        path = engine.resolve_data_path(self.app.config["UPLOAD_DIR"], row["stored_name"])
        ref = engine.reference_fit(
            path, data_start=row["data_start"], total_rows=row["total_rows"], **params
        )
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)


class TestConcurrentStartAndEmergency(EmergencyWebBase):
    def test_simultaneous_start_requests_both_run(self):
        a = self.upload_named("a.csv")
        b = self.upload_named("b.csv")
        with ChunkGate({f"train-{a}", f"train-{b}"}) as gate:
            def do_start(rid):
                client = self.app.test_client()
                return client.post(f"/runs/{rid}/start",
                                   data={"pause_after": "", "chunk_limit": "6"})

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(do_start, [a, b]))
            self.assertTrue(all(r.status_code == 302 for r in results))
            gate.wait_entered(f"train-{a}")
            gate.wait_entered(f"train-{b}")
            self.assertTrue(worker_running(a))
            self.assertTrue(worker_running(b))
            gate.free(f"train-{a}")
            gate.free(f"train-{b}")
        self.wait_status(a, "finished")
        self.wait_status(b, "finished")

    def test_emergency_yields_rejects_starts_and_auto_resumes(self):
        a = self.upload_named("a.csv")
        b = self.upload_named("b.csv")
        e = self.upload_named("e.csv")
        with ChunkGate({f"train-{a}", f"train-{b}"}) as gate:
            self.assertEqual(self.start(a).status_code, 302)
            self.assertEqual(self.start(b).status_code, 302)
            gate.wait_entered(f"train-{a}")
            gate.wait_entered(f"train-{b}")

            resp = self.client.post(f"/runs/{e}/emergency")
            self.assertEqual(resp.status_code, 302)

            def active_waiting():
                snap = self.client.get("/api/emergency").get_json()
                return snap if snap["active"] and snap["phase"] == "waiting" else None
            snap = wait_for(active_waiting)
            self.assertEqual(snap["emergency_run_id"], e)

            # Form start during the epoch is rejected with a flash and no worker.
            page = self.client.post(
                f"/runs/{e}/start", data={"pause_after": ""},
                follow_redirects=True,
            ).get_data(as_text=True)
            self.assertIn("紧急", page)
            self.assertFalse(worker_running(e))

            # JSON clients get an explicit 409.
            rj = self.client.post(f"/runs/{b}/start?format=json",
                                  data={"pause_after": ""})
            self.assertEqual(rj.status_code, 409)
            self.assertFalse(rj.get_json()["ok"])

            # A second emergency is also rejected.
            re2 = self.client.post(f"/runs/{a}/emergency?format=json")
            self.assertEqual(re2.status_code, 409)

            # Arming is durable in the DB.
            self.assertEqual(self.row(a)["yield_requested"], 1)
            self.assertEqual(self.row(b)["yield_requested"], 1)

            gate.free(f"train-{a}")
            gate.free(f"train-{b}")

        # Both ordinary jobs parked at the boundary, and the chunk's group
        # update (chunk_limit == group_size == 6) committed before yielding.
        for rid in (a, b):
            row = self.wait_idle(rid)
            self.assertEqual(row["status"], "paused")
            self.assertEqual(row["yielded"], 1)
            self.assertEqual(row["updates_done"], 1)
            self.assertEqual(row["stop_kind"], "yield")

        # Emergency gets exclusivity, runs to completion, epoch ends.
        self.wait_status(e, "finished")
        self.wait_emergency_inactive()

        # Exactly the yielded ordinary jobs are automatically resumed.
        for rid in (a, b):
            self.wait_status(rid, "finished")
            self.assert_exact_and_audit(rid)
        e_row = self.row(e)
        self.assertEqual(e_row["emergency"], 1)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            self.assertTrue(engine.audit(conn, e)["ok"])

        # After the epoch ordinary starts are accepted again (no emergency
        # rejection): a finished run reports its normal "已完成" error instead.
        again = self.client.post(f"/runs/{e}/start?format=json",
                                 data={"pause_after": ""})
        self.assertNotEqual(again.status_code, 409)


class TestPauseDuringYield(EmergencyWebBase):
    def test_pause_while_waiting_is_not_auto_resumed(self):
        a = self.upload_named("a.csv")
        e = self.upload_named("e.csv")
        with ChunkGate({f"train-{a}"}) as gate:
            self.start(a)
            gate.wait_entered(f"train-{a}")
            self.assertEqual(self.client.post(f"/runs/{e}/emergency").status_code, 302)
            wait_for(lambda: self.row(a)["yield_requested"] == 1 or None)

            self.assertEqual(self.client.post(f"/runs/{a}/pause").status_code, 302)
            gate.free(f"train-{a}")

        row = self.wait_idle(a)
        self.assertEqual(row["status"], "paused")
        self.assertEqual(row["yielded"], 0)
        self.assertEqual(row["stop_kind"], "user")
        self.assertEqual(row["updates_done"], 1)

        self.wait_status(e, "finished")
        self.wait_emergency_inactive()
        time.sleep(0.4)
        row = self.row(a)
        self.assertEqual((row["status"], row["updates_done"]), ("paused", 1))
        self.assertFalse(worker_running(a))

        # explicit manual continuation still completes exactly
        self.start(a)
        self.wait_status(a, "finished")
        self.assert_exact_and_audit(a)

    def test_pause_yielded_during_exclusive_phase_vetoes_resume(self):
        a = self.upload_named("a.csv")
        e = self.upload_named("e.csv")
        with ChunkGate({f"train-{a}", f"train-{e}"}) as gate:
            self.start(a)
            gate.wait_entered(f"train-{a}")
            self.client.post(f"/runs/{e}/emergency")
            gate.free(f"train-{a}")

            row = self.wait_idle(a)
            self.assertEqual(row["yielded"], 1)

            # coordinator moves to exclusive and blocks the emergency worker in
            # its first chunk so the exclusive window is observable/stable.
            gate.wait_entered(f"train-{e}")
            deadline = time.time() + 10
            phase = None
            while time.time() < deadline:
                phase = self.client.get("/api/emergency").get_json()["phase"]
                if phase == "exclusive":
                    break
                time.sleep(0.02)
            self.assertEqual(phase, "exclusive")

            # actively pause the parked job during exclusivity
            page = self.client.post(f"/runs/{a}/pause",
                                    follow_redirects=True).get_data(as_text=True)
            self.assertIn("不自动续跑", page)
            self.assertEqual(self.row(a)["yielded"], 0)
            gate.free(f"train-{e}")

        self.wait_status(e, "finished")
        self.wait_emergency_inactive()
        time.sleep(0.4)
        row = self.row(a)
        self.assertEqual(row["status"], "paused")
        self.assertFalse(worker_running(a))

        self.start(a)
        self.wait_status(a, "finished")
        self.assert_exact_and_audit(a)


class TestEmergencyFailure(EmergencyWebBase):
    def test_failed_emergency_still_resumes_yielded_normals(self):
        a = self.upload_named("a.csv")
        e = self.upload_named("bad-e.csv")
        with engine.session(self.app.config["DB_PATH"]) as conn:
            stored = engine.get_run(conn, e)["stored_name"]
        path = engine.resolve_data_path(self.app.config["UPLOAD_DIR"], stored)
        os.unlink(path)

        with ChunkGate({f"train-{a}"}) as gate:
            self.start(a)
            gate.wait_entered(f"train-{a}")
            self.client.post(f"/runs/{e}/emergency")
            gate.free(f"train-{a}")

        row = self.wait_idle(a)
        self.assertEqual(row["yielded"], 1)

        e_row = self.wait_status(e, "failed")
        self.assertIn("数据文件", e_row["message"])
        self.wait_emergency_inactive()
        coord = self.app.extensions["emergency_coordinator"]
        self.assertEqual(coord.last_outcome, "failed")

        # failure of the emergency job must not strand the normal job
        self.wait_status(a, "finished")
        self.assert_exact_and_audit(a)


class TestPageStates(EmergencyWebBase):
    def test_pages_show_waiting_yielded_and_exclusive(self):
        a = self.upload_named("a.csv")
        e = self.upload_named("e.csv")
        with ChunkGate({f"train-{a}", f"train-{e}"}) as gate:
            self.start(a)
            gate.wait_entered(f"train-{a}")
            self.client.post(f"/runs/{e}/emergency")
            wait_for(lambda: self.row(a)["yield_requested"] == 1 or None)

            page_a = self.client.get(f"/runs/{a}").get_data(as_text=True)
            self.assertIn("等待让位", page_a)
            self.assertIn("紧急运行", page_a)
            # ordinary start control is disabled while the emergency epoch is on
            self.assertIn("紧急通道处理中", page_a)
            # pause stays available for the running yielder
            self.assertNotIn("当前没有运行中", page_a)

            page_e = self.client.get(f"/runs/{e}").get_data(as_text=True)
            self.assertIn("等待让位", page_e)
            self.assertIn("其他启动请求一律退回", page_e)

            index = self.client.get("/").get_data(as_text=True)
            self.assertEqual(self.client.get("/").status_code, 200)
            self.assertIn("等待让位", index)

            gate.free(f"train-{a}")
            self.wait_idle(a)
            gate.wait_entered(f"train-{e}")
            # let the phase flip to exclusive
            wait_for(lambda: (
                self.client.get("/api/emergency").get_json()["phase"] == "exclusive"
            ) or None)

            page_e2 = self.client.get(f"/runs/{e}").get_data(as_text=True)
            self.assertIn("独占中", page_e2)
            page_a2 = self.client.get(f"/runs/{a}").get_data(as_text=True)
            self.assertIn("已让位", page_a2)
            self.assertIn("转为主动暂停", page_a2)

            gate.free(f"train-{e}")

        self.wait_emergency_inactive()


class TestEmergencyPaused(EmergencyWebBase):
    def test_pausing_emergency_job_still_resumes_normals(self):
        a = self.upload_named("a.csv")
        e = self.upload_named("e.csv")
        with ChunkGate({f"train-{a}", f"train-{e}"}) as gate:
            self.start(a)
            gate.wait_entered(f"train-{a}")
            self.client.post(f"/runs/{e}/emergency")
            gate.free(f"train-{a}")
            self.wait_idle(a)  # A yielded

            gate.wait_entered(f"train-{e}")
            # pause the emergency job while it holds exclusivity
            self.client.post(f"/runs/{e}/pause")
            gate.free(f"train-{e}")

        e_row = self.wait_status(e, "paused")
        self.assertIn("暂停", e_row["message"])
        self.wait_emergency_inactive()
        # emergency paused is one of the terminal triggers for auto-resume
        self.wait_status(a, "finished")
        self.assert_exact_and_audit(a)
        coord = self.app.extensions["emergency_coordinator"]
        self.assertEqual(coord.last_outcome, "paused")


class TestRestartManualResume(EmergencyWebBase):
    def test_restart_clears_auto_state_but_keeps_checkpoint(self):
        rid = self.upload_named("r.csv", group_size="10", chunk_limit="3")
        self.start(rid, pause_after="1", chunk_limit="3")
        row = self.wait_status(rid, "paused")
        self.assertEqual(row["part_rows"], 3)
        mark_conn = engine.connect(self.app.config["DB_PATH"])
        try:
            with engine.transaction(mark_conn):
                mark_conn.execute("UPDATE runs SET yielded=1 WHERE id=?", (rid,))
        finally:
            mark_conn.close()

        # simulate service restart: a brand new app on the same data dir
        app2 = create_app({
            "DB_PATH": self.app.config["DB_PATH"],
            "UPLOAD_DIR": self.app.config["UPLOAD_DIR"],
            "SECRET_KEY": "test",
            "TESTING": True,
        })
        with engine.session(self.app.config["DB_PATH"]) as conn:
            row = dict(engine.get_run(conn, rid))
        self.assertEqual(row["yielded"], 0)
        self.assertIn("重启", row["message"])

        # nothing auto-runs after restart
        time.sleep(0.3)
        self.assertFalse(worker_running(rid))

        client2 = app2.test_client()
        resp = client2.post(f"/runs/{rid}/start",
                            data={"pause_after": "", "chunk_limit": "3"})
        self.assertEqual(resp.status_code, 302)
        wait_for(lambda: (
            not worker_running(rid) and self.row(rid)["status"] == "finished"
        ) or None)
        self.assert_exact_and_audit(rid, group_size=10, chunk_limit=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
