"""Process-local registry of background training threads.

The deployment is deliberately a single gunicorn worker process (see
Dockerfile): training checkpoints live in SQLite, but the question "is a
training thread for run N alive?" and the emergency coordinator rely on
in-process thread handles.
"""

from __future__ import annotations

import threading

from trainer import engine

_workers: dict[int, threading.Thread] = {}
_workers_lock = threading.Lock()


def worker_running(run_id: int) -> bool:
    with _workers_lock:
        thread = _workers.get(run_id)
        return thread is not None and thread.is_alive()


def running_worker_ids() -> list[int]:
    """Ids whose worker thread is currently alive."""
    with _workers_lock:
        return [rid for rid, thread in _workers.items() if thread.is_alive()]


def _worker_main(db_path: str, upload_dir: str, run_id: int) -> None:
    conn = engine.connect(db_path)
    try:
        row = engine.get_run(conn, run_id)
        path = engine.resolve_data_path(upload_dir, row["stored_name"])
        engine.train(conn, run_id, path)
    except (engine.StateError, engine.InputError) as exc:
        engine._set_status(conn, run_id, "failed", str(exc))
    except BaseException as exc:  # noqa: BLE001 - keep the process and the DB consistent
        engine._set_status(
            conn, run_id, "crashed", f"训练线程异常: {type(exc).__name__}: {exc}"
        )
    finally:
        conn.close()
        with _workers_lock:
            if _workers.get(run_id) is threading.current_thread():
                _workers.pop(run_id, None)


def start_worker(app, run_id: int) -> bool:
    """Start a daemon training thread. Returns False if one is already alive."""
    with _workers_lock:
        existing = _workers.get(run_id)
        if existing is not None and existing.is_alive():
            return False
        thread = threading.Thread(
            target=_worker_main,
            args=(app.config["DB_PATH"], app.config["UPLOAD_DIR"], run_id),
            name=f"train-{run_id}",
            daemon=True,
        )
        _workers[run_id] = thread
    thread.start()
    return True


__all__ = [
    "worker_running",
    "running_worker_ids",
    "start_worker",
]
