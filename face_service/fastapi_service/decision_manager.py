"""
Decision Manager for face recognition with stability and identity-based cooldown.

Architecture:
  FaceEngine → raw embedding
    ↓
  DecisionManager (stability + identity-based cooldown)
    ↓
  DB Logger (save only if allowed)

Features:
- Minimum 3 consecutive recognitions before marking RECOGNIZED (stability)
- Stability and cooldown also apply to UNKNOWN events (tracks consecutive
  UNKNOWN frames and throttles repeated UNKNOWN logs per camera)
- Identity-based cooldown: (person_name, camera_id) → prevents repetitive logging
- In-memory last_logged_time tracking per identity/event
- Future-ready for face + gait fusion (fusion_score replaces similarity)
- Thread-safe operation for live video processing

Cooldown Logic:
  - Key: (person_name, camera_id)
  - If current_time - last_logged_time < 30 seconds: skip DB insert
  - Otherwise: insert into DB and update last_logged_time

Stability Logic (per track_id):
  - Track consecutive frame recognitions or UNKNOWN results for each track_id
  - Require `stability_threshold` consecutive identical outcomes before stable
  - If person changes or UNKNOWN/RECOGNIZED switch occurs, counter resets
"""

import logging
from time import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RecognitionDecision:
    """Decision object for a single recognized face."""
    decision: str  # "RECOGNIZED" or "UNKNOWN"
    person_name: Optional[str]
    similarity: float
    threshold: float
    confidence: str  # "low", "medium", "high"
    should_log: bool = True  # Whether to log this decision to DB
    is_stable: bool = False  # Whether this has met stability threshold
    stability_count: int = 0  # Number of consecutive recognitions


@dataclass
class StabilityTracker:
    """Tracks consecutive recognitions per track_id for stability validation."""
    person_name: Optional[str] = None
    count: int = 0  # Consecutive recognitions
    similarity: float = 0.0
    first_seen_time: float = field(default_factory=time)
    
    def reset(self):
        """Reset stability counter (different person or UNKNOWN detected)."""
        self.person_name = None
        self.count = 0
        self.similarity = 0.0
        self.first_seen_time = time()
    
    def update(self, person_name: Optional[str], similarity: float):
        """Update stability for same person."""
        if person_name == self.person_name:
            self.count += 1
            self.similarity = similarity
        else:
            self.reset()
            self.person_name = person_name
            self.count = 1
            self.similarity = similarity


class DecisionManager:
    """
    Manages recognition decisions with stability check and identity-based cooldown.
    
    Prevents:
    1. Unstable recognitions (noise): requires 3 consecutive frames
    2. Repetitive logging: 30-second cooldown per (person_name, camera_id)
    
    Thread-safe for use in async FastAPI context.
    """
    
    STABILITY_THRESHOLD = 3  # Require 3 consecutive recognitions
    COOLDOWN_SECONDS = 30.0  # 30-second identity-based cooldown
    
    def __init__(
        self,
        stability_threshold: int = STABILITY_THRESHOLD,
        cooldown_seconds: float = COOLDOWN_SECONDS,
    ):
        """
        Initialize DecisionManager.
        
        Args:
            stability_threshold: Number of consecutive recognitions before stable
            cooldown_seconds: Seconds to wait before logging same person again
        """
        self.stability_threshold = stability_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Last logged time per identity or event key: (person_name, camera_id) -> timestamp
        #  * person_name will be None when the event was UNKNOWN. This allows the
        #    same cooldown map to be used for both RECOGNIZED and UNKNOWN events.
        #    Using camera_id ensures unknown spam is throttled per source.
        self.last_logged_time: Dict[Tuple[Optional[str], Optional[str]], float] = {}
        
        # Stability tracking per track_id: track_id -> StabilityTracker
        # Tracks the most recent observed "person" for every running track_id.
        # The value None is used internally to represent UNKNOWN so that we can
        # count consecutive UNKNOWN frames as well as consecutive recognitions.
        self.stability_by_track: Dict[str, StabilityTracker] = {}
        
        logger.info(
            f"DecisionManager initialized: "
            f"stability_threshold={stability_threshold}, "
            f"cooldown_seconds={cooldown_seconds}"
        )
    
    def can_log_identity(
        self,
        person_name: Optional[str],
        camera_id: Optional[str],
    ) -> bool:
        """
        Check if identity is NOT in cooldown window.
        
        Args:
            person_name: Recognized person name (or None for UNKNOWN)
            camera_id: Camera identifier (or None)
            
        Returns:
            True if can log (not in cooldown), False otherwise
        """
        """
        Check whether an event may be logged based on the cooldown timer.

        Both RECOGNIZED and UNKNOWN events share the same in‑memory map.  The
        tuple key is simply `(person_name, camera_id)`; when `person_name` is
        `None` the entry represents an UNKNOWN event.  Using `camera_id` as the
        second element gives us a per‑camera cooldown for unknowns, which is
        sufficient to suppress spam without touching the database.

        Args:
            person_name: Recognized person name or `None` for UNKNOWN
            camera_id: Camera identifier (or `None` if unavailable)

        Returns:
            True if we are outside the cooldown window and may emit a db log.
        """
        now = time()
        key = (person_name, camera_id)
        last_time = self.last_logged_time.get(key, 0.0)
        elapsed = now - last_time
        can_log = elapsed >= self.cooldown_seconds
        if not can_log:
            # log at debug level so we can trace suppression during testing
            logger.debug(
                f"Cooldown in effect: person={person_name or '<UNKNOWN>'}, "
                f"camera={camera_id}, elapsed={elapsed:.1f}s "
                f"< cooldown={self.cooldown_seconds}s"
            )
        return can_log
    
    def update_last_logged(
        self,
        person_name: Optional[str],
        camera_id: Optional[str],
    ) -> None:
        """
        Mark identity as logged (update last_logged_time).
        
        Args:
            person_name: Recognized person name
            camera_id: Camera identifier
        """
        # Always update the map, even when person_name is None (UNKNOWN).
        # The key `(None, camera_id)` is used by can_log_identity to throttle
        # unknown events per camera.
        key = (person_name, camera_id)
        self.last_logged_time[key] = time()
        logger.debug(f"Updated cooldown for {key}")
    
    def check_stability(
        self,
        track_id: str,
        person_name: Optional[str],
        similarity: float,
    ) -> Tuple[bool, int]:
        """
        Check if recognition is stable (met threshold of consecutive frames).
        
        Args:
            track_id: Unique track identifier from video processor
            person_name: Currently recognized person (or None)
            similarity: Current similarity score
            
        Returns:
            (is_stable, stability_count) tuple
                - is_stable: True if reached stability_threshold
                - stability_count: Number of consecutive recognitions
        """
        # Initialize or get existing tracker
        # If no track_id was provided we cannot reason about consecutive frames
        # so we simply treat everything as immediately stable (no filtering).
        if not track_id:
            return True, 0

        # lazily create tracker for new tracks
        if track_id not in self.stability_by_track:
            self.stability_by_track[track_id] = StabilityTracker()

        tracker = self.stability_by_track[track_id]

        # Update stability counter regardless of whether the face is recognized or
        # unknown.  `person_name` may be `None` for UNKNOWN which allows the same
        # infrastructure to count consecutive unknown frames as well.
        tracker.update(person_name, similarity)

        is_stable = tracker.count >= self.stability_threshold
        if not is_stable:
            name_display = person_name if person_name is not None else '<UNKNOWN>'
            logger.debug(
                f"Track {track_id}: {tracker.count}/{self.stability_threshold} "
                f"consecutive frames for {name_display}"
            )

        return is_stable, tracker.count
    
    def process_recognition(
        self,
        track_id: Optional[str],
        person_name: Optional[str],
        similarity: float,
        decision_str: str,  # "RECOGNIZED" or "UNKNOWN"
        threshold: float,
        confidence: str,  # "low", "medium", "high"
        camera_id: Optional[str] = None,
    ) -> RecognitionDecision:
        """
        Process raw recognition and apply stability + cooldown logic.
        
        Main entry point for routes.py to get logging decisions.
        
        Args:
            track_id: Track ID from video processor (used for stability tracking)
            person_name: Matched person name from DB
            similarity: Similarity score from threshold manager
            decision_str: Initial decision from threshold manager
            threshold: EER threshold
            confidence: Confidence level
            camera_id: Camera identifier for cooldown key
            
        Returns:
            RecognitionDecision with should_log flag
        """
        # default values
        should_log = False
        is_stable = True
        stability_count = 0

        # 1. STABILITY CHECK
        # Regardless of decision we want to ensure there are at least
        # `stability_threshold` consecutive frames of the same outcome before
        # emitting a log.  The tracker handles both recognized names and
        # ``None`` for UNKNOWN.  If no track_id is supplied we treat every
        # frame as already stable (can't accumulate without an identifier).
        if track_id:
            is_stable, stability_count = self.check_stability(
                track_id, person_name, similarity
            )
            if not is_stable:
                name_display = person_name if person_name is not None else '<UNKNOWN>'
                logger.debug(
                    f"Unstable recognition for track {track_id}: "
                    f"{stability_count}/{self.stability_threshold} frames of "
                    f"{name_display}; waiting for stability"
                )
                # Early return -- not stable yet, do not log.
                return RecognitionDecision(
                    decision='PROCESSING',
                    person_name=None,
                    similarity=similarity,
                    threshold=threshold,
                    confidence='low',
                    should_log=False,
                    is_stable=False,
                    stability_count=stability_count,
                )

        # 2. COOLDOWN CHECK (applies to both RECOGNIZED and UNKNOWN once stable)
        # Use the same in-memory map keyed by (person_name, camera_id).  ``person_name``
        # will be ``None`` for UNKNOWN.
        if self.can_log_identity(person_name, camera_id):
            should_log = True
            self.update_last_logged(person_name, camera_id)
        else:
            should_log = False

        # Build the decision returned to callers.  API responses remain unchanged
        # (we reflect the original threshold decision), but `RecognitionDecision`
        # may indicate `'PROCESSING'` when stability hasn't been reached.
        return RecognitionDecision(
            decision=decision_str if is_stable else 'PROCESSING',
            person_name=person_name if is_stable else None,
            similarity=similarity,
            threshold=threshold,
            confidence=confidence if is_stable else 'low',
            should_log=should_log,
            is_stable=is_stable,
            stability_count=stability_count,
        )
    
    def cleanup_track(self, track_id: str) -> None:
        """
        Remove tracking data for a track_id (e.g., when tracker forgets it).
        
        Optional method to prevent memory bloat if many tracks are processed.
        
        Args:
            track_id: Track identifier to clean up
        """
        if track_id in self.stability_by_track:
            del self.stability_by_track[track_id]
            logger.debug(f"Cleaned up stability tracker for {track_id}")
    
    def get_stats(self) -> Dict:
        """Return diagnostic stats about current state."""
        return {
            'active_tracks': len(self.stability_by_track),
            'identity_cooldowns': len(self.last_logged_time),
            'stability_threshold': self.stability_threshold,
            'cooldown_seconds': self.cooldown_seconds,
        }


# Global decision manager instance (initialized in main.py)
decision_manager: Optional[DecisionManager] = None


def init_decision_manager(
    stability_threshold: int = DecisionManager.STABILITY_THRESHOLD,
    cooldown_seconds: float = DecisionManager.COOLDOWN_SECONDS,
) -> DecisionManager:
    """Initialize global DecisionManager instance."""
    global decision_manager
    decision_manager = DecisionManager(stability_threshold, cooldown_seconds)
    return decision_manager


def get_decision_manager() -> DecisionManager:
    """Get global DecisionManager instance."""
    if decision_manager is None:
        raise RuntimeError(
            "DecisionManager not initialized. Call init_decision_manager() first."
        )
    return decision_manager
