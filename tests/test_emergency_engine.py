"""Engine-level tests for emergency yielding and budget continuation."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer import engine  # noqa: E402


def write_csv(path: str, n: int) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("x,y\n")
        for i in range(n):
            fh.write(f"{i * 0.37 - 4.0!r},{2.5 * (i * 0.37 - 4.0) - 1.25!r}\n")


class EmergencyEngineBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="emergency-engine-")
        self.db = os.path.join(self.tmp, "t.sqlite3")
        self.csv = os.path.join(self.tmp, "data.csv")
        self.conn = engine.connect(self.db)

    def tearDown(self) -> None:
        self.conn.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def create(self, n: int = 40, **kw) -> int:
        write_csv(self.csv, n)
        defaults = dict(group_size=4, chunk_limit=4, lr=0.01, momentum=0.9,
                        init_w=0.0, init_b=0.0)
        defaults.update(kw)
        return engine.create_run(self.conn, self.csv, "data.csv", self.csv, **defaults)

    def reference(self, n: int = 40, **kw) -> dict:
        defaults = dict(group_size=4, chunk_limit=4, lr=0.01, momentum=0.9,
                        init_w=0.0, init_b=0.0)
        defaults.update(kw)
        meta = engine.scan_file(self.csv)
        return engine.reference_fit(
            self.csv, data_start=meta.data_start, total_rows=meta.rows, **defaults
        )

    def row(self, rid: int):
        return engine.get_run(self.conn, rid)

    def finish_after_auto_resume(self, rid: int, **ref_kw) -> dict:
        ids = engine.take_yielded(self.conn, 0)
        self.assertEqual(ids, [rid])
        result = engine.train(self.conn, rid, self.csv)
        self.assertEqual(result.status, "finished")
        ref = self.reference(**ref_kw)
        r = self.row(rid)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(r[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])
        return ref


class TestYieldSemantics(EmergencyEngineBase):
    def test_yield_after_chunk_commits_triggered_group_update(self):
        # chunk_limit == group_size: every chunk triggers a group update.
        rid = self.create(n=32, group_size=4, chunk_limit=4)
        armed = {"done": False}

        def on_chunk(event):
            if event["event"] == "chunk" and not armed["done"]:
                armed["done"] = True
                self.assertTrue(engine.request_yield(self.conn, rid))

        result = engine.train(self.conn, rid, self.csv, progress=on_chunk)
        self.assertEqual((result.status, result.kind), ("paused", "yield"))
        r = self.row(rid)
        self.assertEqual(r["yielded"], 1)
        self.assertEqual(r["yield_requested"], 0)
        # Yield was armed from the "chunk" progress event, which fires before
        # that chunk's group commit.  The triggered update must still land:
        # exactly one group (the armed chunk itself) is durable, not zero.
        self.assertEqual(r["updates_done"], 1)
        self.assertEqual(r["samples_done"], 4)
        self.finish_after_auto_resume(rid, n=32, group_size=4, chunk_limit=4)

    def test_yield_midgroup_checkpoints_partial_accumulator(self):
        rid = self.create(n=40, group_size=10, chunk_limit=3)
        armed = {"done": False}

        def on_chunk(event):
            if event["event"] == "chunk" and not armed["done"]:
                armed["done"] = True
                engine.request_yield(self.conn, rid)

        result = engine.train(self.conn, rid, self.csv, progress=on_chunk)
        self.assertEqual(result.kind, "yield")
        r = self.row(rid)
        self.assertEqual((r["yielded"], r["part_rows"]), (1, 3))
        self.assertGreater(r["next_offset"], r["data_start"])
        # no row double counting after resume
        self.finish_after_auto_resume(rid, n=40, group_size=10, chunk_limit=3)

    def test_user_pause_takes_precedence_over_yield(self):
        rid = self.create(n=40)

        def on_chunk(event):
            if event["event"] == "chunk":
                engine.request_yield(self.conn, rid)
                engine.request_pause(self.conn, rid)

        result = engine.train(self.conn, rid, self.csv, progress=on_chunk)
        self.assertEqual(result.kind, "user")
        r = self.row(rid)
        self.assertEqual((r["yielded"], r["pause_requested"]), (0, 0))
        # coordinator must not auto-resume it
        self.assertEqual(engine.take_yielded(self.conn, 0), [])

    def test_hold_yielded_vetoes_auto_resume(self):
        rid = self.create(n=40, group_size=10, chunk_limit=3)

        def on_chunk(event):
            if event["event"] == "chunk":
                engine.request_yield(self.conn, rid)

        result = engine.train(self.conn, rid, self.csv, progress=on_chunk)
        self.assertEqual(result.kind, "yield")
        self.assertTrue(engine.hold_yielded(self.conn, rid))
        r = self.row(rid)
        self.assertEqual((r["yielded"], r["pause_requested"]), (0, 1))
        self.assertEqual(engine.take_yielded(self.conn, 999), [])
        # manual resume clears the veto and finishes correctly
        with engine.transaction(self.conn):
            self.conn.execute("UPDATE runs SET pause_requested=0 WHERE id=?", (rid,))
        engine.train(self.conn, rid, self.csv)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])


class TestBudgetContinuation(EmergencyEngineBase):
    def test_chunk_budget_count_continues_across_yield(self):
        rid = self.create(n=95, group_size=10, chunk_limit=3)
        engine.set_pause_after_chunks(self.conn, rid, 4)
        chunks = {"n": 0}

        def on_chunk(event):
            if event["event"] == "chunk":
                chunks["n"] += 1
                if chunks["n"] == 2:
                    engine.request_yield(self.conn, rid)

        r1 = engine.train(self.conn, rid, self.csv, progress=on_chunk)
        self.assertEqual(r1.kind, "yield")
        # 4 - 2 chunks remain in the budget
        self.assertEqual(self.row(rid)["pause_after_chunks"], 2)

        engine.take_yielded(self.conn, 0)
        r2 = engine.train(self.conn, rid, self.csv)
        self.assertEqual((r2.status, r2.kind, r2.chunks_this_session),
                         ("paused", "chunk", 2))
        # total chunks read towards the same target == 4
        self.assertEqual(self.row(rid)["updates_done"], 1)
        engine.set_pause_after_chunks(self.conn, rid, None)
        r3 = engine.train(self.conn, rid, self.csv)
        self.assertEqual(r3.status, "finished")
        ref = self.reference(n=95, group_size=10, chunk_limit=3)
        r = self.row(rid)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(r[key], ref[key], key)

    def test_group_budget_count_continues_across_yield(self):
        rid = self.create(n=95, group_size=10, chunk_limit=4)
        engine.set_pause_after_groups(self.conn, rid, 3)
        armed = {"n": 0}

        def on_commit(event):
            if event["event"] == "committed":
                armed["n"] += 1
                if armed["n"] == 1:
                    engine.request_yield(self.conn, rid)

        r1 = engine.train(self.conn, rid, self.csv, progress=on_commit)
        self.assertEqual(r1.kind, "yield")
        self.assertEqual(self.row(rid)["pause_after_groups"], 2)
        self.assertEqual(self.row(rid)["updates_done"], 1)

        engine.take_yielded(self.conn, 0)
        r2 = engine.train(self.conn, rid, self.csv)
        self.assertEqual((r2.status, r2.kind), ("paused", "group"))
        self.assertEqual(self.row(rid)["updates_done"], 3)

        engine.set_pause_after_groups(self.conn, rid, None)
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(n=95, group_size=10, chunk_limit=4)
        r = self.row(rid)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(r[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])


class TestRecoverEmergency(EmergencyEngineBase):
    def test_recover_clears_yield_markers_keeps_checkpoint(self):
        rid = self.create(n=40, group_size=10, chunk_limit=3)
        engine.set_pause_after_chunks(self.conn, rid, 1)
        first = engine.train(self.conn, rid, self.csv)
        self.assertEqual(first.status, "paused")
        checkpoint_offset = self.row(rid)["next_offset"]
        with engine.transaction(self.conn):
            self.conn.execute(
                "UPDATE runs SET yielded=1, yield_requested=1 WHERE id=?", (rid,)
            )
        self.assertEqual(engine.recover(self.conn), 0)
        r = self.row(rid)
        self.assertEqual((r["yielded"], r["yield_requested"]), (0, 0))
        self.assertEqual(r["next_offset"], checkpoint_offset)
        self.assertEqual(r["part_rows"], 3)
        self.assertIn("重启", r["message"])
        # the real checkpoint remains manually resumable to an exact finish
        engine.set_pause_after_chunks(self.conn, rid, None)
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(n=40, group_size=10, chunk_limit=3)
        r = self.row(rid)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(r[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
