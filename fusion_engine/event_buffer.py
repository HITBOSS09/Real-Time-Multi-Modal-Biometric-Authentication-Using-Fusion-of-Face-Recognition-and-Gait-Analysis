from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple


@dataclass
class Event:
    identity: Optional[str]
    score: float
    timestamp: datetime
    modality: str  # 'face' or 'gait'
    details: dict = None


class EventBuffer:
    """Temporal buffer for face/gait events that enables fusion and
    multi-person tracking.

    The object keeps two deques (one per modality) of recent events; entries
    older than ``time_window`` seconds are automatically pruned.  When a new
    event is added the buffer can be queried for matching cross‑modality pairs
    based on the configured time window.  Gait predictions of ``Unknown`` are
    ignored at insertion time in order to comply with the requirement to drop
    these during temporal voting.
    """

    def __init__(self, time_window: float):
        self.time_window = time_window
        self.face_events: deque[Event] = deque()
        self.gait_events: deque[Event] = deque()

    def _prune(self, now: datetime):
        cutoff = now - timedelta(seconds=self.time_window)
        for dq in (self.face_events, self.gait_events):
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()

    def add_face(self, event: Event) -> None:
        now = datetime.utcnow()
        self._prune(now)
        self.face_events.append(event)

    def add_gait(self, event: Event) -> None:
        # drop unknown gait identities as per requirement
        if event.identity is None or str(event.identity).lower() == "unknown":
            return
        now = datetime.utcnow()
        self._prune(now)
        self.gait_events.append(event)

    def get_pairs(self) -> List[Tuple[Event, Event]]:
        """Return list of (face_event, gait_event) pairs whose timestamps are
        within ``time_window`` of each other.
        """
        now = datetime.utcnow()
        self._prune(now)
        pairs: List[Tuple[Event, Event]] = []
        for f in self.face_events:
            for g in self.gait_events:
                delta = abs((f.timestamp - g.timestamp).total_seconds())
                if delta <= self.time_window:
                    pairs.append((f, g))
        return pairs

    def pop_pairs(self) -> List[Tuple[Event, Event]]:
        """Return and remove all matching pairs from the buffer.

        The pairs are determined by ``get_pairs``; after constructing the list
        the underlying events are purged so that the same combination is not
        reported again on subsequent calls.
        """
        pairs = self.get_pairs()
        # remove individual events that appeared in any pair
        for f, g in pairs:
            try:
                self.face_events.remove(f)
            except ValueError:
                pass
            try:
                self.gait_events.remove(g)
            except ValueError:
                pass
        return pairs

    def clear(self) -> None:
        """Empty the buffer (useful for resets or tests)."""
        self.face_events.clear()
        self.gait_events.clear()
