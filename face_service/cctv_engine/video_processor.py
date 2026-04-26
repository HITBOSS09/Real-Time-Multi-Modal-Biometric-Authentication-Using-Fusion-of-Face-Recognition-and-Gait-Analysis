"""
Real-time multi-person CCTV face recognition service.

Reads video source (RTSP stream, webcam, or file) and processes frames
synchronously in a single process.  Detection and embedding extraction are
done locally with InsightFace (RetinaFace) and recognition is performed
entirely in memory using vectors pre-loaded from the PostgreSQL database at
startup.  Database writes are offloaded to a background thread, avoiding any
IO on the critical path.  In-memory cooldown and smoothing keep overlay
stable between detection intervals.

FastAPI remains deployed for administrative endpoints (enrollment, threshold
management, log inspection) but is no longer involved in the realtime loop.
The engine is deliberately simple and modular to allow future extensions (e.g.
gait analysis or multimodal fusion).  YOLO/ByteTrack have been removed,
returning to a face-only pipeline with tighter bounding boxes and higher FPS.
"""

import argparse
import logging
import time
from typing import Any, Optional

import cv2
import numpy as np
import threading
import queue
import asyncio

# Enable OpenCV SIMD optimizations for faster image processing
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


class FrameGrabber(threading.Thread):
    """Threaded frame grabber for non‑blocking capture.

    Reads from a video source in a background thread, keeping the most recent
    frame available to the main loop via ``read()``.  This allows the detection
    and recognition workload to proceed at its own pace without being held up
    by ``VideoCapture.read()`` latency.
    """

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()

    def run(self):

        # regular VideoCapture behaviour
        if isinstance(self.source, str) and self.source.isdigit():
            src = int(self.source)
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        # Reduce RTSP latency by limiting buffer size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        while self.running:
            ret, frm = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frm
        self.cap.release()

    def read(self) -> Optional[np.ndarray]:
        """Return the last captured frame (copy) or ``None`` if none yet."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self, timeout: float = 1.0):
        """Stop the grabber and join the thread.

        The background loop may be blocked waiting on ``cap.read()``; we set
        ``running`` to ``False`` and then join with a timeout to avoid hanging
        indefinitely.  If the thread does not exit within ``timeout`` seconds a
        warning is logged and we return anyway, allowing callers to proceed.
        """
        self.running = False
        self.join(timeout)
        if self.is_alive():
            logging.getLogger(__name__).warning(
                "FrameGrabber thread did not exit after %.1fs", timeout
            )

from .config import (
    DETECTION_INTERVAL_FRAMES,
    COOLDOWN_SECONDS,
)
from .cooldown_manager import CooldownManager

# import face engine & db manager directly
from fastapi_service.face_engine import init_face_engine, FaceEngineConfig, FaceEngine
from fastapi_service.db import init_db
from .db_logger import DBLogger

# placeholder classes for future gait/fusion
class GaitEngine:
    def process(self, frame: Any):
        pass

class FusionManager:
    def fuse(self, face_id: Any, gait_id: Any):
        pass


class RecognitionLoggerThread(threading.Thread):
    """Background thread that writes recognition events to the database.

    The thread owns its own ``asyncio`` event loop and an ``asyncpg`` pool
    managed by ``cctv_engine.db_logger.DBLogger``.  The main processing loop
    simply enqueues events, so there is no blocking I/O on the hot path.
    """

    def __init__(self, max_queue_size: int = 1000):
        super().__init__(daemon=True)
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._db_logger = DBLogger()
        # event loop for async operations
        self._loop = asyncio.new_event_loop()
        # metrics for monitoring blocking/drops
        self._dropped_events = 0
        self._max_queue_size = max_queue_size

    def run(self):
        asyncio.set_event_loop(self._loop)
        # initialise pool
        self._loop.run_until_complete(self._db_logger.init())
        while True:
            item = self._queue.get()
            if item is None:  # sentinel to stop thread
                break
            # each item is a dict of arguments accepted by DBLogger.log
            self._loop.run_until_complete(self._db_logger.log(**item))
        # close pool before exiting
        self._loop.run_until_complete(self._db_logger.close())

    def log(
        self,
        person_id: Optional[int],
        name: Optional[str],
        confidence: Optional[str],
        similarity: float,
        decision: str,
        track_id: Optional[int] = None,
        camera_id: Optional[str] = None,
    ):
        """Queue an event for asynchronous logging (non-blocking).
        
        If the queue is full, logs a warning and drops the event to prevent
        blocking the main recognition loop.
        """
        event = {
            'person_id': person_id,
            'name': name,
            'confidence': confidence,
            'similarity': similarity,
            'decision': decision,
            'track_id': track_id,
            'camera_id': camera_id,
        }
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events % 10 == 0:
                logger.warning(
                    f"Logger queue full: dropped {self._dropped_events} events. "
                    f"DB logging falling behind (max_queue={self._max_queue_size})"
                )

    def stop(self):
        """Signal the thread to shutdown and wait for completion."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.put(None, timeout=2.0)
            except queue.Full:
                logger.error("Could not send stop signal to logger thread (queue full)")
                return
        self.join()

logger = logging.getLogger(__name__)


def bbox_iou(a: list, b: list) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        a: [x1, y1, x2, y2]
        b: [x1, y1, x2, y2]
    
    Returns:
        float: IoU score in [0, 1]
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    
    inter = max(0, xB - xA) * max(0, yB - yA)
    
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class VideoProcessor:
    """Frame capture + in-process face recognition engine.

    Performs detection via InsightFace directly, matches embeddings against
    local PostgreSQL/pgvector database, and applies in-memory cooldown.
    ``FastAPI`` is no longer involved in the realtime path.  The service is
    synchronous and optimized for high FPS.
    """

    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        display: bool = True,
        detection_interval: int = DETECTION_INTERVAL_FRAMES,
        source: str = "rtsp",
    ):
        self.rtsp_url = rtsp_url
        self.source = source
        self.camera_id = camera_id
        self.display = display
        self.detection_interval = detection_interval
        # initialize engines and DB later in initialize()

        # memory for smoothing/associating bounding boxes between frames
        # keyed by track identifier (person name or generated track_x)
        # values: { 'bbox':[x1,y1,x2,y2], 'last_seen': frame_index }
        self.track_memory: dict[str, dict] = {}
        # smoothing coefficient: 0.3 retains 30% of previous bbox
        self._smooth_alpha = 0.3
        # store last set of results so that overlay can be drawn every frame
        self.current_face_results: list[dict] = []
        # distance threshold for matching new bbox to existing track
        self._match_dist_thresh = 80  # pixels (~depends on resolution)
        # counter used when creating new track IDs for unknown faces
        self._track_id_counter = 0

        # optional hooks for external consumers (GUI, fusion modules, etc.)
        self.result_callbacks: list = []
        self._callback_lock = threading.Lock()  # protect callback list

        # helpers for GUI integration
        self.grabber: Optional[FrameGrabber] = None  # set when run() starts
        self.last_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        # Cooldown for UNKNOWN detections to reduce false logging
        self._unknown_cooldown: dict[str, int] = {}  # track_id -> frame counter
        self._unknown_min_frames = 3  # require UNKNOWN to persist this many frames before logging
        
        # IoU tracking threshold for simple track association
        self._iou_threshold = 0.3  # minimum overlap to match track
        
        # smoothing coefficient: retain 30% of previous bbox (stable)
        self._smooth_alpha = 0.3

        # embedding cache: track_id -> {'embedding': np.array, 'last_time': float}
        self._embedding_cache: dict[str, dict] = {}
        self._embedding_interval = 0.5  # seconds between recomputes

    def initialize(self):
        """Initialize engines and database connections."""
        # initialize face engine
        cfg = FaceEngineConfig()
        self.face_engine: FaceEngine = init_face_engine(cfg)
        # initialize database manager
        self.db = init_db()
        # initialize threshold manager (required for decisions)
        from fastapi_service.threshold import init_threshold_manager
        init_threshold_manager(self.db)
        # load all embeddings into memory for fast vector search
        self._embeddings, self._person_ids, self._person_names = self._load_embeddings()
        logger.info(f"Loaded {len(self._person_ids)} embeddings into memory")
        # cooldown manager
        self.cooldown = CooldownManager(COOLDOWN_SECONDS)
        # background logger thread
        self.logger_thread = RecognitionLoggerThread()
        self.logger_thread.start()
        logger.info("✓ VideoProcessor initialized (in-process face engine)")

    def _load_embeddings(self):
        """Pull every embedding from the database and normalize them.

        Returns:
            tuple: (embeddings_array, person_ids, person_names)
        """
        rows = self.db.get_all_embeddings()
        if not rows:
            # empty matrix
            return np.empty((0, 512), dtype=np.float32), [], []

        vectors = []
        ids = []
        names = []
        for r in rows:
            emb = np.array(r['embedding'], dtype=np.float32)
            # normalize each vector immediately
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            vectors.append(emb)
            ids.append(r['person_id'])
            names.append(r['person_name'])
        matrix = np.vstack(vectors)
        return matrix, ids, names
    def close(self):
        """Clean up resources (if any)."""
        # clear cooldown state
        self._unknown_cooldown.clear()
        # clear track memory
        self.track_memory.clear()
        # clear embedding cache
        self._embedding_cache.clear()
        # clear callbacks
        with self._callback_lock:
            self.result_callbacks.clear()
        # stop logger thread and flush queue
        if hasattr(self, 'logger_thread'):
            self.logger_thread.stop()
        logger.info("✓ VideoProcessor shutdown complete")

    # ------------------------------------------------------------------
    # Extensions / hooks for external consumers (GUI, fusion modules, etc.)
    # ------------------------------------------------------------------

    def add_result_callback(self, callback):
        """Register a function to be called whenever new face results arrive.

        The callback is invoked with a single argument: the list of face result
        dictionaries produced during the most recent detection interval.  This
        mechanism allows GUI code or other fusion components to be notified
        immediately without polling ``current_face_results``.
        """
        with self._callback_lock:
            self.result_callbacks.append(callback)
    
    def remove_result_callback(self, callback):
        """Remove a previously registered callback."""
        with self._callback_lock:
            try:
                self.result_callbacks.remove(callback)
            except ValueError:
                pass  # callback not in list

    def _update_results(self, updated_results: list):
        """Helper used internally to update state and fire callbacks."""
        self.current_face_results = updated_results
        with self._callback_lock:
            callbacks = self.result_callbacks.copy()  # snapshot to avoid lock hold
        for cb in callbacks:
            try:
                cb(updated_results)
            except Exception:
                logger.exception("result callback raised an exception")

    def _draw_face_result(self, frame: Any, face_result: dict):
        """Draw face bounding box and recognition result on frame.
        
        Uses bounding box from InsightFace detection (from API response).
        """
        decision = face_result.get('decision', 'UNKNOWN')
        person_name = face_result.get('person_name', 'UNKNOWN')
        similarity = face_result.get('similarity', 0.0)
        bbox = face_result.get('bbox', [])
        track_id = face_result.get('track_id')
        
        if not bbox or len(bbox) < 4:
            logger.warning(f"Invalid bbox: {bbox}")
            return
        
        # Extract bbox coordinates from InsightFace detection
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Draw bounding box rectangle
        color = (0, 255, 0) if decision == 'RECOGNIZED' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        # include track ID when available
        if track_id:
            label = f"{person_name} ({decision}) [{similarity:.2f}] #{track_id}"
        else:
            label = f"{person_name} ({decision}) [{similarity:.2f}]"
        
        # Draw label background and text above box
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, 25)  # Place above box, but not off-screen
        
        # Draw semi-transparent background for text
        cv2.rectangle(
            frame,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (text_x, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )



    def run(self):
        """Main processing loop (synchronous).

        - Capture frames from video source
        - Every ``detection_interval`` frames run InsightFace on the full
          image, extract embeddings, match in the local database, and apply
          cooldown.
        - Always draw the most recent results on every frame for smooth
          overlay.
        """
        logger.info("Opening video source: %s", self.rtsp_url)
        
        # prepare the grabber thread
        if self.source == "webcam":
            # Try primary webcam (0), fallback to secondary (1) if available
            source = 0
        else:
            # RTSP mode
            source = self.rtsp_url
            if not source:
                raise ValueError("RTSP URL required when source mode is 'rtsp'")

        grabber = FrameGrabber(source)
        self.grabber = grabber  # expose for external consumers (e.g. GUI)
        try:
            grabber.start()
        except RuntimeError as e:
            # If webcam 0 fails, try webcam 1
            if self.source == "webcam" and source == 0:
                logger.warning(f"Webcam 0 failed: {e}. Attempting webcam 1...")
                source = 1
                grabber = FrameGrabber(source)
                self.grabber = grabber
                grabber.start()
            else:
                raise RuntimeError(f"Cannot open webcam or RTSP source: {e}")
        
        # Wait for first frame
        max_wait = 5.0  # seconds
        start_wait = time.perf_counter()
        while grabber.read() is None and (time.perf_counter() - start_wait) < max_wait:
            time.sleep(0.05)
        
        if grabber.read() is None:
            raise RuntimeError("No frames captured from video source after 5 seconds")

        # allow a moment for stable first frame
        time.sleep(0.1)

        frame_count = 0
        last_time = time.perf_counter()
        fps_frames = 0
        fps_val = 0.0

        try:
            while True:
                frame = grabber.read()
                if frame is None:
                    # no frame yet / stream ended (prevent busy loop on freeze)
                    time.sleep(0.01)
                    continue

                frame_count += 1
                fps_frames += 1
                now = time.perf_counter()
                
                if now - last_time >= 1.0:
                    fps_val = fps_frames / (now - last_time)
                    fps_frames = 0
                    last_time = now

                # detection on every frame, no motion/ROI logic
                # optimize resolution: resize to ~416 width for faster inference
                h, w = frame.shape[:2]
                scale_factor = 416 / w if w > 416 else 1.0
                small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                faces = self.face_engine.detect_faces(small_frame)
                # scale bboxes back to original frame size
                for face in faces:
                    face.bbox = [b / scale_factor for b in face.bbox]

                # cleanup tracks older than age limit
                track_age_limit = 30
                to_delete = [k for k, v in self.track_memory.items()
                             if frame_count - v.get('last_seen', 0) > track_age_limit]
                for k in to_delete:
                    del self.track_memory[k]
                    if k in self._unknown_cooldown:
                        del self._unknown_cooldown[k]
                    if k in self._embedding_cache:
                        del self._embedding_cache[k]
                if to_delete:
                    logger.debug(f"Pruned {len(to_delete)} stale tracks, {len(self.track_memory)} remain")

                updated_results = []
                processed_track_ids = set()

                for idx, face in enumerate(faces):
                    if not hasattr(face, 'bbox') or face.bbox is None:
                        logger.debug(f"Face {idx}: missing bbox, skipping")
                        continue
                    raw_bbox = face.bbox
                    if len(raw_bbox) < 4:
                        logger.debug(f"Face {idx}: invalid bbox length {len(raw_bbox)}, skipping")
                        continue
                    try:
                        raw_bbox_floats = [float(b) for b in raw_bbox[:4]]
                        if not all(np.isfinite(b) for b in raw_bbox_floats):
                            logger.debug(f"Face {idx}: non-finite bbox values, skipping")
                            continue
                    except (ValueError, TypeError):
                        logger.debug(f"Face {idx}: non-numeric bbox, skipping")
                        continue

                    det_score = getattr(face, 'det_score', 1.0)
                    if det_score < 0.65:
                        continue

                    new_bbox = [int(b) for b in raw_bbox]
                    width = new_bbox[2] - new_bbox[0]
                    height = new_bbox[3] - new_bbox[1]
                    if width < 60 or height < 60:
                        continue
                    
                    # expand bbox by 10% for better coverage
                    x1, y1, x2, y2 = new_bbox
                    w = x2 - x1
                    h = y2 - y1
                    x1 = int(x1 - 0.1 * w)
                    y1 = int(y1 - 0.1 * h)
                    x2 = int(x2 + 0.1 * w)
                    y2 = int(y2 + 0.1 * h)
                    new_bbox = [x1, y1, x2, y2]
                    h, w = frame.shape[:2]
                    new_bbox[0] = max(0, min(w - 1, new_bbox[0]))
                    new_bbox[1] = max(0, min(h - 1, new_bbox[1]))
                    new_bbox[2] = max(0, min(w - 1, new_bbox[2]))
                    new_bbox[3] = max(0, min(h - 1, new_bbox[3]))
                    if new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
                        logger.warning(f"Invalid bbox after clamping: {new_bbox}, skipping")
                        continue

                    # track association using IoU
                    best_key = None
                    best_iou = 0.0
                    for track_id, info in self.track_memory.items():
                        old_bbox = info['bbox']
                        iou = bbox_iou(new_bbox, old_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_key = track_id
                    if best_key is not None and best_iou > self._iou_threshold:
                        assigned_id = best_key
                    else:
                        if self._track_id_counter > 100000:
                            self._track_id_counter = 0
                        self._track_id_counter += 1
                        assigned_id = f"track_{self._track_id_counter}"

                    # compute embedding only if needed (cache per track, recompute every 0.5s)
                    current_time = time.time()
                    if (assigned_id not in self._embedding_cache or 
                        current_time - self._embedding_cache[assigned_id]['last_time'] > self._embedding_interval):
                        emb = face.embedding
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        else:
                            logger.warning(f"Zero-norm embedding at face {idx}, skipping")
                            continue
                        self._embedding_cache[assigned_id] = {'embedding': emb, 'last_time': current_time}
                    else:
                        emb = self._embedding_cache[assigned_id]['embedding']

                    # recognition using cached embedding
                    if self._embeddings is not None and self._embeddings.size > 0:
                        sims = np.dot(self._embeddings, emb)
                        best_idx = int(np.argmax(sims))
                        similarity = float(sims[best_idx])
                        person_id = self._person_ids[best_idx]
                        person_name = self._person_names[best_idx]
                        from fastapi_service.threshold import get_threshold_manager
                        thr_mgr = get_threshold_manager()
                        decision = thr_mgr.get_decision(similarity, person_name)
                    else:
                        similarity = 0.0
                        person_name = None
                        person_id = None
                        from fastapi_service.threshold import get_threshold_manager
                        thr_mgr = get_threshold_manager()
                        decision = {
                            'decision': 'UNKNOWN',
                            'person_name': None,
                            'confidence': 'low',
                            'threshold': thr_mgr.get_threshold(),
                        }

                    if decision['decision'] == 'RECOGNIZED' and person_id is not None:
                        if self.cooldown.can_log(person_id):
                            self.logger_thread.log(
                                person_id=person_id,
                                name=decision.get('person_name'),
                                confidence=decision.get('confidence'),
                                similarity=similarity,
                                decision=decision['decision'],
                                camera_id=self.camera_id,
                            )
                            self.cooldown.update(person_id)

                    if assigned_id in self.track_memory:
                        old_bbox = self.track_memory[assigned_id]['bbox']
                        alpha = self._smooth_alpha
                        smoothed = [
                            int(alpha * old_bbox[i] + (1 - alpha) * new_bbox[i])
                            for i in range(4)
                        ]
                    else:
                        smoothed = new_bbox.copy()

                    self.track_memory[assigned_id] = {
                        'bbox': smoothed,
                        'last_seen': frame_count,
                        'person_id': person_id,
                        'person_name': person_name,
                        'similarity': similarity,
                        'decision': decision,
                    }

                    face_result = {
                        'decision': decision['decision'],
                        'person_name': decision.get('person_name'),
                        'similarity': similarity,
                        'bbox': new_bbox,
                        'track_id': assigned_id,
                    }
                    updated_results.append(face_result)
                    processed_track_ids.add(assigned_id)

                    if decision['decision'] != 'RECOGNIZED':
                        if assigned_id not in self._unknown_cooldown:
                            self._unknown_cooldown[assigned_id] = 0
                        self._unknown_cooldown[assigned_id] += 1
                        if self._unknown_cooldown[assigned_id] >= self._unknown_min_frames:
                            self.logger_thread.log(
                                person_id=None,
                                name=None,
                                confidence=None,
                                similarity=similarity,
                                decision=decision['decision'],
                                camera_id=self.camera_id,
                            )
                            self._unknown_cooldown[assigned_id] = -1

                self._unknown_cooldown = {
                    track_id: count
                    for track_id, count in self._unknown_cooldown.items()
                    if track_id in self.track_memory
                }

                # update results and callbacks
                self._update_results(updated_results)

                # always draw using last known results, but only if track updated within 1 frame
                for face_res in self.current_face_results:
                    track_id = face_res.get('track_id')
                    if track_id and track_id in self.track_memory:
                        if frame_count - self.track_memory[track_id]['last_seen'] > 1:
                            continue
                    self._draw_face_result(frame, face_res)

                # keep a copy of the annotated frame for GUI consumers
                with self._frame_lock:
                    self.last_frame = frame.copy()

                # Draw FPS counter
                cv2.putText(
                    frame,
                    f"FPS: {fps_val:.1f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

                # Display frame
                if self.display:
                    cv2.imshow('Face Recognition Engine', frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        logger.info("Exit key pressed")
                        break

        finally:
            # stop grabber if running
            try:
                grabber.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="CCTV face recognition engine (InsightFace + local database)"
    )
    parser.add_argument("--source", default="rtsp", help="Video source mode: 'webcam' or 'rtsp'")
    parser.add_argument("--rtsp", default=None, help="RTSP URL (required if source is 'rtsp')")
    parser.add_argument("--camera", default="default_camera", help="Unique camera identifier")
    parser.add_argument("--interval", type=int, default=DETECTION_INTERVAL_FRAMES, help="Recognition interval in frames")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI output")

    args = parser.parse_args()

    processor = VideoProcessor(
        rtsp_url=args.rtsp,
        camera_id=args.camera,
        display=not args.no_display,
        detection_interval=args.interval,
        source=args.source,
    )
    processor.initialize()
    try:
        processor.run()
    finally:
        processor.close()


if __name__ == "__main__":
    main()
