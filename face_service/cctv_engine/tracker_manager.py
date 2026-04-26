"""
Tracker manager maintaining per-track state for recognition decisions.

This module no longer depends on the external ``bytetrack`` package
– tracking is performed by the YOLOv8 model itself (see
``ultralytics.YOLO.track`` with ``tracker="bytetrack.yaml"``).  The
manager simply accepts lists of ``(track_id, bbox)`` pairs and updates
internal state, handling cleanup and decision locking.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# avoid circular import; import individual constants
from .config import STATE_LOCK_CONSECUTIVE

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """State information maintained for each active track."""
    track_id: int
    person_id: Optional[int] = None
    name: Optional[str] = None
    similarity_history: List[float] = field(default_factory=list)
    frames_in_row: int = 0
    last_seen: float = field(default_factory=time.time)
    decision_locked: bool = False
    logged: bool = False



class TrackerManager:
    """Manage per-track state for recognition decisions.

    The underlying object tracking is performed by the YOLOv8 model; this
    class merely records the last-seen time for each ``track_id`` and
    implements the stable decision logic that determines when a track has been
    observed consistently enough to lock and log an identity.

    ``update`` now accepts a list of ``(track_id, bbox)`` pairs produced by the
    detector (typically from ``YOLO.track(..., persist=True)``) and returns the
    same list for convenience.  It also purges stale tracks after a brief
    timeout to prevent memory leaks.
    """

    def __init__(self, cleanup_sec: float = 2.0):
        # solely responsible for state; no external tracker library needed
        self.states: Dict[int, TrackState] = {}
        self.cleanup_sec = cleanup_sec

    def update(self, track_list: List[Tuple[int, List[float]]]) -> List[Tuple[int, List[float]]]:
        """Ingest a list of ``(track_id, bbox)`` pairs and refresh their state.

        The list is returned unchanged, allowing callers to chain the call.
        """
        now = time.time()
        for tid, bbox in track_list:
            state = self.states.get(tid)
            if state is None:
                state = TrackState(track_id=tid)
                self.states[tid] = state
            state.last_seen = now
        self._cleanup_old(now)
        return track_list

    def _cleanup_old(self, now: float):
        stale = [tid for tid, s in self.states.items() if (now - s.last_seen) > self.cleanup_sec]
        for tid in stale:
            logger.debug(f"Removing stale track {tid}")
            del self.states[tid]

    def update_state(
        self,
        track_id: int,
        person_id: Optional[int],
        name: Optional[str],
        similarity: float,
        threshold: float,
    ) -> bool:
        """Update the decision state for ``track_id``.

        Returns ``True`` when the stable decision rule is satisfied and the
        decision is locked (i.e. ready to be logged).  The caller should then
        log the event and call ``mark_logged``.
        """
        state = self.states.get(track_id)
        if state is None:
            return False
        if state.decision_locked:
            return False
        if person_id is None:
            # unknown face resets the running count
            state.person_id = None
            state.name = None
            state.frames_in_row = 0
            state.similarity_history.clear()
            return False
        # same identity as before?
        if state.person_id == person_id:
            state.frames_in_row += 1
        else:
            state.person_id = person_id
            state.name = name
            state.frames_in_row = 1
            state.similarity_history.clear()
        state.similarity_history.append(similarity)
        if state.frames_in_row >= STATE_LOCK_CONSECUTIVE:
            # compute average of the most recent N similarities
            recent = state.similarity_history[-STATE_LOCK_CONSECUTIVE:]
            avg = sum(recent) / len(recent)
            if avg > threshold:
                state.decision_locked = True
                logger.info(f"Track {track_id} locked on person {person_id} ({name}), avg_sim={avg:.3f}")
                return True
        return False

    def mark_logged(self, track_id: int):
        """Mark a locked decision as having been logged to the database."""
        state = self.states.get(track_id)
        if state:
            state.logged = True

    def get_display_info(self, track_id: int) -> Dict[str, Optional[float]]:
        """Return current overlay information for ``track_id``."""
        state = self.states.get(track_id)
        if not state:
            return {}
        status = "UNKNOWN"
        if state.decision_locked and state.person_id is not None:
            status = "RECOGNIZED"
        return {
            "name": state.name or "UNKNOWN",
            "track_id": state.track_id,
            "confidence": state.similarity_history[-1] if state.similarity_history else 0.0,
            "status": status,
        }
