"""
In-memory cooldown tracker to prevent logging the same identity too often.

This is the first layer of protection against duplicate events.  The video
processor will check this before even contacting the database.  A second,
stricter check is implemented at the database level in ``db_logger.py``.
"""

import time
from typing import Dict


class CooldownManager:
    def __init__(self, cooldown_sec: float):
        self.cooldown_sec = cooldown_sec
        self._last_logged: Dict[int, float] = {}

    def can_log(self, person_id: int) -> bool:
        """Return ``True`` if ``person_id`` may be logged again.

        The first time a person is seen the method will always return ``True``.
        Subsequent calls return ``False`` until the configured interval has
        elapsed.
        """
        last = self._last_logged.get(person_id)
        if last is None:
            return True
        return (time.time() - last) >= self.cooldown_sec

    def update(self, person_id: int):
        """Record that ``person_id`` was just logged."""
        self._last_logged[person_id] = time.time()

    def clear(self):
        """Clear all stored cooldowns (useful for testing)."""
        self._last_logged.clear()
