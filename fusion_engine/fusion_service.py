"""Top‑level entrypoint for the multimodal fusion engine.

This script periodically polls the face and gait databases, buffers incoming
recognition events, and applies hybrid score fusion according to the rules
specified by the architecture document.  It prints fused decisions to stdout
for now but can be extended to log results or call downstream components.
"""
import time
from datetime import datetime
from typing import Any

# face_service and gait_service are large components with their own
# dependencies; import them lazily so that the fusion engine can be imported
# without immediately triggering missing‑package errors.
DatabaseConfig = None
DatabaseManager = None
GaitDatabase = None


def _import_databases():
    global DatabaseConfig, DatabaseManager, GaitDatabase
    try:
        from face_service.fastapi_service.db import DatabaseConfig, DatabaseManager
    except ImportError as e:  # pragma: no cover - environment may not be set up
        raise ImportError(
            "Failed to import face_service database module. "
            "Make sure face_service requirements are installed: pgvector, psycopg2, etc.\n"
            f"Original error: {e}"
        )
    try:
        from gait_service.multi_gait_system.database.database import GaitDatabase
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Failed to import gait_service database module. "
            "Ensure gait_service dependencies are available.\n"
            f"Original error: {e}"
        )

from fusion_engine.config_loader import load_config
from fusion_engine.event_buffer import Event, EventBuffer
from fusion_engine.fusion_logic import fuse_identity


def _row_to_face_event(row: dict) -> Event:
    # row after get_recognition_logs() already includes timestamp as datetime
    identity = row.get("name") or row.get("person_name")
    score = float(row.get("similarity", 0.0) or 0.0)
    ts = row.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return Event(identity=identity, score=score, timestamp=ts, modality="face", details=row)


def _result_to_gait_event(result: Any) -> Event:
    # result tuple as returned by GaitDatabase.get_recognition_results
    # (track_id, identity, modality, avg_similarity, max_similarity, prediction_count, start_time, end_time)
    track_id, identity, modality, avg_sim, max_sim, _, _, end_time = result
    score = float(avg_sim or 0.0)
    ts = end_time
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return Event(identity=identity, score=score, timestamp=ts, modality="gait", details={"track_id": track_id})


def main(poll_interval: float = 1.0):
    cfg = load_config()

    # load heavy dependencies lazily
    _import_databases()

    face_db_cfg = DatabaseConfig()
    face_db = DatabaseManager(face_db_cfg)
    face_db.init_sync_pool()

    gait_db = GaitDatabase(cfg.gait_db_path)

    buffer = EventBuffer(cfg.time_window)

    seen_face_ids = set()
    seen_gait_keys = set()

    print("[fusion_service] started; press Ctrl+C to quit")

    try:
        while True:
            # poll face logs
            face_logs = face_db.get_recognition_logs(limit=200)
            for row in reversed(face_logs):
                log_id = row.get("id")
                if log_id in seen_face_ids:
                    continue
                seen_face_ids.add(log_id)
                event = _row_to_face_event(row)
                buffer.add_face(event)

            # poll gait results
            gait_results = gait_db.get_recognition_results(limit=200)
            for result in reversed(gait_results):
                track_id = result[0]
                end_time = result[7]
                key = (track_id, end_time)
                if key in seen_gait_keys:
                    continue
                seen_gait_keys.add(key)
                event = _result_to_gait_event(result)
                buffer.add_gait(event)

            # fuse any matching pairs and print outcome
            pairs = buffer.pop_pairs()
            for f_event, g_event in pairs:
                identity, final_score = fuse_identity(
                    f_event.identity,
                    f_event.score,
                    g_event.identity,
                    g_event.score,
                    cfg.alpha,
                )
                print(
                    "[fusion]",
                    f"face={f_event.identity}@{f_event.timestamp.isoformat()}",
                    f"gait={g_event.identity}@{g_event.timestamp.isoformat()}",
                    "->",
                    f"{identity} ({final_score:.3f})",
                )
                
                # Log fusion result to PostgreSQL database
                face_db.log_fusion_result(
                    identity=identity,
                    face_identity=f_event.identity,
                    gait_identity=g_event.identity,
                    face_score=f_event.score,
                    gait_score=g_event.score,
                    fused_score=final_score,
                )

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[fusion_service] shutting down")
    finally:
        face_db.close_sync_pool()


if __name__ == "__main__":
    main()
