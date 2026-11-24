"""Process-local emergency scheduler.

Rules implemented here (see the feature request):

* From acceptance of an emergency run until the emergency worker exits, every
  other start request is rejected (:meth:`EmergencyCoordinator.start_normal`
  and :meth:`EmergencyCoordinator.request`).
* Live ordinary workers are asked to yield via the per-run
  ``yield_requested`` flag.  The engine stops them only after the current read
  chunk and the group update that chunk triggers.
* The emergency worker starts only once every asked worker thread has fully
  stopped ("全部停稳").
* After the emergency job finishes, pauses or fails, exactly the runs parked
  with ``yielded=1`` (and not actively paused) are automatically restarted.
* All state here is in-memory on purpose: after a service restart there is no
  auto-resume; :func:`trainer.engine.recover` leaves checkpoints for manual
  continuation.
"""

from __future__ import annotations

import logging
import threading
import time

from trainer import engine

from .workers import running_worker_ids, start_worker, worker_running

log = logging.getLogger(__name__)

POLL_INTERVAL = 0.02


class EmergencyCoordinator:
    def __init__(self, app) -> None:
        self.app = app
        self.lock = threading.RLock()
        self.active = False
        self.phase: str | None = None  # 'waiting' | 'exclusive'
        self.emergency_id: int | None = None
        self.promoted = False
        self.yield_ids: list[int] = []
        self.resumed_ids: list[int] = []
        self.thread: threading.Thread | None = None
        # Summary of the most recent finished emergency epoch.
        self.last_emergency_id: int | None = None
        self.last_resumed_ids: list[int] = []
        self.last_outcome: str = ""

    # ------------------------------------------------------------------ #
    # Request entry points (called from Flask route threads)
    # ------------------------------------------------------------------ #
    def ensure_inactive(self) -> None:
        with self.lock:
            if self.active:
                raise engine.StateError(
                    f"紧急作业 #{self.emergency_id} 通道处理中（{self._phase_text()}）："
                    "从受理到紧急作业退出，其他启动请求一律退回，请等待紧急作业完成、暂停或失败"
                )

    def start_normal(self, run_id: int) -> bool:
        """Gate every ordinary worker start behind the emergency epoch."""
        with self.lock:
            self.ensure_inactive()
            if not start_worker(self.app, run_id):
                raise engine.StateError("后台训练线程仍在收尾，请稍后重试")
            return True

    def request(self, run_id: int) -> dict:
        with self.lock:
            if self.active:
                raise engine.StateError(
                    f"已有紧急作业 #{self.emergency_id} 正在处理（{self._phase_text()}），"
                    "不能再受理新的紧急运行"
                )
            db_path = self.app.config["DB_PATH"]
            promoted = worker_running(run_id)
            live_ids = running_worker_ids()
            with engine.session(db_path) as conn:
                # Raises InputError if the run does not exist.
                ids = engine.arm_emergency(conn, run_id, live_ids)
                if not promoted:
                    engine.prepare_emergency_run(conn, run_id)
            self.active = True
            self.phase = "waiting"
            self.emergency_id = run_id
            self.promoted = promoted
            self.yield_ids = ids
            self.resumed_ids = []
            self.last_outcome = ""
            thread = threading.Thread(
                target=self._orchestrate_safe,
                name=f"emergency-{run_id}",
                daemon=True,
            )
            self.thread = thread
        thread.start()
        return self.snapshot()

    # ------------------------------------------------------------------ #
    # Orchestration (single coordinator thread per emergency epoch)
    # ------------------------------------------------------------------ #
    def _orchestrate_safe(self) -> None:
        run_id = self.emergency_id
        try:
            self._wait_yielders_settled()
            with self.lock:
                if self.active:
                    self.phase = "exclusive"
            if not self.promoted:
                start_worker(self.app, run_id)
            self._wait_worker_exit(run_id)
            try:
                with engine.session(self.app.config["DB_PATH"]) as conn:
                    outcome_row = engine.get_run(conn, run_id)
                    self.last_outcome = str(outcome_row["status"])
                    ids = engine.take_yielded(conn, run_id)
            except Exception:  # noqa: BLE001
                log.exception("emergency %s: auto-resume query failed", run_id)
                ids = []
            # Start resume workers while the epoch is still marked active so a
            # concurrent user request cannot double-start any of them; the
            # worker registry itself also deduplicates live threads.
            for rid in ids:
                try:
                    start_worker(self.app, rid)
                except Exception:  # noqa: BLE001
                    log.exception("auto-resume of run %s failed", rid)
            with self.lock:
                self.resumed_ids = ids
                self.last_resumed_ids = list(ids)
        except Exception:  # noqa: BLE001
            log.exception("emergency coordinator for run %s crashed", run_id)
        finally:
            with self.lock:
                self.active = False
                self.phase = None
                self.last_emergency_id = self.emergency_id
                self.emergency_id = None
                self.yield_ids = []
                self.thread = None

    def _wait_yielders_settled(self) -> None:
        while True:
            with self.lock:
                ids = list(self.yield_ids)
                if not self.active:
                    return
            if not any(worker_running(rid) for rid in ids):
                return
            time.sleep(POLL_INTERVAL)

    def _wait_worker_exit(self, run_id: int) -> None:
        # start_worker registers the thread before this point; wait until the
        # thread is fully gone (its final checkpoint transaction has committed
        # and the DB connection closed).
        while worker_running(run_id):
            time.sleep(POLL_INTERVAL)
        # One extra poll window lets the worker's finally-block pop the handle
        # and the status become observable to other connections.
        deadline_loops = 50
        for _ in range(deadline_loops):
            with engine.session(self.app.config["DB_PATH"]) as conn:
                row = engine.get_run(conn, run_id)
            if row["status"] != "running":
                return
            time.sleep(POLL_INTERVAL)

    # ------------------------------------------------------------------ #
    # Introspection for JSON API / templates
    # ------------------------------------------------------------------ #
    def _phase_text(self) -> str:
        if self.phase == "waiting":
            return "等待普通作业让位"
        if self.phase == "exclusive":
            return "紧急作业独占训练中"
        return "受理中"

    def snapshot(self) -> dict:
        with self.lock:
            active = self.active
            base = {
                "active": active,
                "phase": self.phase,
                "phase_text": self._phase_text() if active else "",
                "emergency_run_id": self.emergency_id,
                "promoted": self.promoted,
                "yield_ids": list(self.yield_ids),
                "resumed_ids": list(self.resumed_ids),
                "last_emergency_id": self.last_emergency_id,
                "last_resumed_ids": list(self.last_resumed_ids),
                "last_outcome": self.last_outcome,
            }
            eid = self.emergency_id
        if not active:
            base.update({"runs": [], "yielded_ids": [], "waiting_ids": []})
            return base
        runs: list[dict] = []
        yielded_ids: list[int] = []
        waiting_ids: list[int] = []
        try:
            with engine.session(self.app.config["DB_PATH"]) as conn:
                rows = conn.execute(
                    "SELECT id, status, message, yield_requested, yielded, "
                    "pause_requested, emergency, updates_done, samples_done, total_rows "
                    "FROM runs WHERE yield_requested=1 OR yielded=1 OR id=? ORDER BY id",
                    (eid,),
                ).fetchall()
            for row in rows:
                item = {
                    "id": row["id"],
                    "status": row["status"],
                    "message": row["message"],
                    "yield_requested": int(row["yield_requested"]),
                    "yielded": int(row["yielded"]),
                    "pause_requested": int(row["pause_requested"]),
                    "emergency": int(row["emergency"]),
                    "worker_running": worker_running(int(row["id"])),
                }
                runs.append(item)
                if item["yielded"]:
                    yielded_ids.append(item["id"])
                if item["yield_requested"] and item["status"] == "running":
                    waiting_ids.append(item["id"])
        except Exception:  # noqa: BLE001 - banner must never break a page
            log.exception("emergency snapshot failed")
        base["runs"] = runs
        base["yielded_ids"] = yielded_ids
        base["waiting_ids"] = waiting_ids
        return base

    def banner(self) -> dict | None:
        """Small payload for the site-wide banner; never raises."""
        try:
            snap = self.snapshot()
        except Exception:  # noqa: BLE001
            return None
        return snap if snap.get("active") else None


__all__ = ["EmergencyCoordinator"]
