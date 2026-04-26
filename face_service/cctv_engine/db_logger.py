"""
Asynchronous database logger for recognition events.

This module uses ``asyncpg`` directly instead of going through the FastAPI
service so that the video processing component can continue operating even if
HTTP connectivity to the API is impaired.  The logger implements the second
(layered) cooldown protection by querying the database before attempting an
insert.
"""

import logging
from typing import Optional

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None


logger = logging.getLogger(__name__)


class DBLogger:
    class DatabaseConfig:
        """Minimal subset of configuration used by the CCTV engine.

        This mirrors the implementation in ``fastapi_service/db.py`` but is
        declared locally to avoid a circular import when the package is
        imported from outside the FastAPI service directory.
        """
        def __init__(self):
            import os
            self.host = os.getenv('DB_HOST', 'localhost')
            self.port = int(os.getenv('DB_PORT', '5432'))
            self.database = os.getenv('DB_NAME', 'face_recognition')
            self.user = os.getenv('DB_USER', 'postgres')
            self.password = os.getenv('DB_PASSWORD', 'postgres')

        def get_connection_string(self) -> str:
            return (
                f"postgresql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
            )

    def __init__(self):
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for DBLogger but is not installed")
        cfg = self.DatabaseConfig()
        # asyncpg expects postgresql:// uri
        self._dsn = cfg.get_connection_string()
        self._pool: Optional[asyncpg.pool.Pool] = None

    async def init(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
            logger.info("Async DBLogger pool established")

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Async DBLogger pool closed")

    async def _recent_exists(self, person_id: int, cooldown_sec: int = 45) -> bool:
        """Return ``True`` if a log for ``person_id`` exists within cooldown window."""
        if self._pool is None:
            raise RuntimeError("DBLogger is not initialized")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM recognition_logs
                WHERE person_id = $1
                  AND timestamp > NOW() - INTERVAL '1 second' * $2
                LIMIT 1
                """,
                person_id,
                cooldown_sec,
            )
            return row is not None

    async def log(
        self,
        person_id: Optional[int],
        name: Optional[str],
        confidence: Optional[str],
        similarity: float,
        decision: str,
        track_id: Optional[int] = None,
        camera_id: Optional[str] = None,
    ) -> bool:
        """Insert a recognition event if it is not a recent duplicate.

        Returns ``True`` if the row was added, ``False`` if skipped due to cooldown.
        """
        if person_id is not None:
            if await self._recent_exists(person_id):
                logger.debug("DB duplicate suppressed for person %s", person_id)
                return False
        if self._pool is None:
            raise RuntimeError("DBLogger is not initialized")
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO recognition_logs
                    (person_id, name, confidence, similarity, decision, track_id, camera_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7);
                """,
                person_id,
                name,
                confidence,
                similarity,
                decision,
                track_id,
                camera_id,
            )
        logger.info("Logged recognition event person=%s decision=%s", person_id, decision)
        return True
