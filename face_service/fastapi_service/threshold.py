"""
Threshold management for face recognition.

Handles:
- Loading EER (Equal Error Rate) threshold from database
- Caching threshold for performance
- Similarity score decision logic
"""

import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ThresholdManager:
    """Manages recognition thresholds."""
    
    DEFAULT_EER_THRESHOLD = 0.45  # Default EER threshold (0.0-1.0)
    CONFIG_KEY = 'eer_threshold'
    
    def __init__(self, db_manager):
        """
        Initialize threshold manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self._eer_threshold: Optional[float] = None
        self._cached_at: Optional[datetime] = None
        self._load_threshold()
    
    def _load_threshold(self):
        """Load EER threshold from database."""
        try:
            threshold = self.db.get_config_float(
                self.CONFIG_KEY,
                default=self.DEFAULT_EER_THRESHOLD
            )
            self._eer_threshold = threshold
            self._cached_at = datetime.now()
            logger.info(f"✓ EER threshold loaded: {threshold}")
        except Exception as e:
            logger.warning(f"Failed to load threshold, using default: {e}")
            self._eer_threshold = self.DEFAULT_EER_THRESHOLD
    
    def set_threshold(self, threshold: float):
        """
        Set new EER threshold.
        
        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        try:
            self.db.set_config(self.CONFIG_KEY, str(threshold))
            self._eer_threshold = threshold
            self._cached_at = datetime.now()
            logger.info(f"✓ EER threshold updated to {threshold}")
        except Exception as e:
            logger.error(f"Failed to set threshold: {e}")
            raise
    
    def get_threshold(self) -> float:
        """Get current EER threshold."""
        if self._eer_threshold is None:
            self._load_threshold()
        return self._eer_threshold
    
    def is_recognized(self, similarity: float) -> bool:
        """
        Determine if similarity score indicates recognition.
        
        Args:
            similarity: Similarity score (0.0-1.0)
            
        Returns:
            True if similarity >= threshold
        """
        threshold = self.get_threshold()
        return similarity >= threshold
    
    def get_decision(self, similarity: float, person_name: Optional[str] = None) -> dict:
        """
        Get recognition decision based on similarity.
        
        Args:
            similarity: Similarity score (0.0-1.0)
            person_name: Name of matched person (if similarity >= threshold)
            
        Returns:
            Dictionary with:
            - decision: "RECOGNIZED" or "UNKNOWN"
            - person_name: Name if recognized, else None
            - similarity: The similarity score
            - threshold: The EER threshold used
            - confidence: Confidence level (low/medium/high)
        """
        threshold = self.get_threshold()
        is_recognized = similarity >= threshold
        
        # Confidence levels based on distance from threshold
        margin = similarity - threshold if is_recognized else threshold - similarity
        if margin > 0.1:
            confidence = "high"
        elif margin > 0.05:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            'decision': 'RECOGNIZED' if is_recognized else 'UNKNOWN',
            'person_name': person_name if is_recognized else None,
            'similarity': round(similarity, 4),
            'threshold': round(threshold, 4),
            'confidence': confidence
        }


# Global threshold manager instance
threshold_manager: Optional[ThresholdManager] = None


def init_threshold_manager(db_manager) -> ThresholdManager:
    """Initialize global threshold manager."""
    global threshold_manager
    threshold_manager = ThresholdManager(db_manager)
    return threshold_manager


def get_threshold_manager() -> ThresholdManager:
    """Get global threshold manager."""
    if threshold_manager is None:
        raise RuntimeError("ThresholdManager not initialized. Call init_threshold_manager() first.")
    return threshold_manager
