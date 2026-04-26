"""Database connectivity for NiceGUI dashboard.

Integrates with existing PostgreSQL (face recognition) and SQLite (gait) databases.
Provides common queries for displaying data in the dashboard.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Manages connections to both PostgreSQL (face) and SQLite (gait) databases."""
    
    def __init__(self):
        # PostgreSQL configuration (from face_service)
        self.pg_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'face_recognition'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
        }
        
        # SQLite configuration (from gait_service)
        self.sqlite_path = os.path.expanduser('~/Downloads/biometric_system/gait_service/gait.db')
        
        self._pg_conn = None
        self._sqlite_conn = None
    
    # ========================================================================
    # PostgreSQL FACE DATABASE
    # ========================================================================
    
    def get_pg_connection(self):
        """Get or create PostgreSQL connection."""
        try:
            if self._pg_conn is None or self._pg_conn.closed:
                self._pg_conn = psycopg2.connect(**self.pg_config)
            return self._pg_conn
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def close_pg_connection(self):
        """Close PostgreSQL connection."""
        if self._pg_conn:
            self._pg_conn.close()
            self._pg_conn = None
    
    def get_fusion_results(self, limit: int = 50) -> List[Dict]:
        """Fetch recent fusion results from PostgreSQL."""
        try:
            conn = self.get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT id, identity, face_identity, gait_identity, face_score, gait_score, 
                       fused_score, timestamp
                FROM fusion_results
                ORDER BY timestamp DESC
                LIMIT %s;
                """,
                (limit,)
            )
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch fusion results: {e}")
            return []
    
    def get_face_recognition_logs(self, limit: int = 50) -> List[Dict]:
        """Fetch recent face recognition logs."""
        try:
            conn = self.get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT id, name, decision, similarity, confidence, camera_id, timestamp
                FROM recognition_logs
                WHERE decision = 'RECOGNIZED'
                ORDER BY timestamp DESC
                LIMIT %s;
                """,
                (limit,)
            )
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch face recognition logs: {e}")
            return []
    
    def get_all_persons(self) -> List[Dict]:
        """Get list of all registered persons."""
        try:
            conn = self.get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, name, created_at FROM persons ORDER BY name;")
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch persons: {e}")
            return []
    
    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """Retrieve a registered person by name."""
        try:
            conn = self.get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, name, created_at FROM persons WHERE name = %s;", (name,))
            result = cursor.fetchone()
            cursor.close()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to fetch person by name '{name}': {e}")
            return None

    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Retrieve a registered person by ID."""
        try:
            conn = self.get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, name, created_at FROM persons WHERE id = %s;", (person_id,))
            result = cursor.fetchone()
            cursor.close()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to fetch person by id '{person_id}': {e}")
            return None

            return person_id
        except psycopg2.errors.UniqueViolation:
            logger.warning(f"Person '{name}' already exists")
            conn.rollback()
            return None
        except Exception as e:
            logger.error(f"Failed to create person: {e}")
            conn.rollback()
            return None
    
    def get_fusion_stats(self) -> Dict:
        """Get aggregate statistics from fusion results."""
        try:
            conn = self.get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_fusions,
                    COUNT(DISTINCT identity) as unique_identities,
                    AVG(fused_score) as avg_score,
                    MAX(fused_score) as max_score,
                    MIN(fused_score) as min_score
                FROM fusion_results
                WHERE timestamp > NOW() - INTERVAL '1 hour';
                """
            )
            result = cursor.fetchone()
            cursor.close()
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Failed to fetch fusion stats: {e}")
            return {}
    
    # ========================================================================
    # SQLite GAIT DATABASE
    # ========================================================================
    
    def get_sqlite_connection(self):
        """Get or create SQLite connection."""
        try:
            if not os.path.exists(self.sqlite_path):
                logger.warning(f"Gait database not found: {self.sqlite_path}")
                return None
            if self._sqlite_conn is None:
                self._sqlite_conn = sqlite3.connect(self.sqlite_path)
                self._sqlite_conn.row_factory = sqlite3.Row
            return self._sqlite_conn
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            return None
    
    def get_gait_recognition_logs(self, limit: int = 50) -> List[Dict]:
        """Fetch recent gait recognition logs."""
        try:
            conn = self.get_sqlite_connection()
            if not conn:
                return []
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT track_id, identity, modality, avg_similarity, 
                       max_similarity, prediction_count, start_time, end_time
                FROM recognition_results
                ORDER BY end_time DESC
                LIMIT ?;
                """,
                (limit,)
            )
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch gait logs: {e}")
            return []
    
    def close_sqlite_connection(self):
        """Close SQLite connection."""
        if self._sqlite_conn:
            self._sqlite_conn.close()
            self._sqlite_conn = None
    
    def close_all(self):
        """Close all database connections."""
        self.close_pg_connection()
        self.close_sqlite_connection()
