import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class GaitDatabase:
    """
    Gait identity database with multi-template gallery support.

    Each identity can store *multiple* embedding templates.  During matching
    the query is compared to all templates of every identity and the maximum
    similarity per identity is used.
    """

    def __init__(self, db_path: str = "multi_gait_system/database/gait.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        self._migrate_legacy_identities()

    # ─────────────────────────────────────────────
    # TABLES
    # ─────────────────────────────────────────────
    def _create_tables(self):
        cursor = self.conn.cursor()

        # Legacy single-embedding table (kept for backward compat / migration)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL
        )
        """)

        # Multi-template gallery: one row per template embedding
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS identity_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source_video TEXT,
            created_at TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_name TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            distance REAL,
            status TEXT NOT NULL
        )
        """)

        # Frame-level recognition attempts (stores every prediction)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS frame_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            person_name TEXT,
            similarity REAL,
            decision TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Final track-level recognition results
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            identity TEXT,
            modality TEXT,
            avg_similarity REAL,
            max_similarity REAL,
            prediction_count INTEGER,
            start_time DATETIME,
            end_time DATETIME
        )
        """)

        self.conn.commit()

    def _migrate_legacy_identities(self):
        """Migrate rows from legacy `identities` table into `identity_templates`."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, embedding FROM identities")
        legacy_rows = cursor.fetchall()
        if not legacy_rows:
            return

        for name, blob in legacy_rows:
            # Only migrate if no templates exist yet for this identity
            cursor.execute(
                "SELECT COUNT(*) FROM identity_templates WHERE identity_name = ?",
                (name,)
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO identity_templates (identity_name, embedding) VALUES (?, ?)",
                    (name, blob)
                )
        self.conn.commit()
        print(f"[DB] Migrated {len(legacy_rows)} legacy identity row(s) to multi-template gallery")

    # ─────────────────────────────────────────────
    # REGISTER IDENTITY (multi-template: appends)
    # ─────────────────────────────────────────────
    def register_identity(self, name: str, embedding: np.ndarray, source_video: Optional[str] = None):
        """Append a new template embedding for *name*.

        Args:
            name: identity name
            embedding: numpy array embedding
            source_video: optional path to the source video used to generate this template

        This adds a new template row to `identity_templates` and updates the
        legacy `identities` table with the latest embedding for compatibility.
        """
        embedding = self._normalize(embedding)
        blob = embedding.astype(np.float32).tobytes()

        cursor = self.conn.cursor()

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append to multi-template table (store source video and timestamp)
        cursor.execute("""
            INSERT INTO identity_templates (identity_name, embedding, source_video, created_at)
            VALUES (?, ?, ?, ?)
        """, (name, blob, source_video, created_at))

        # Keep legacy table in sync (latest embedding)
        cursor.execute("""
            INSERT OR REPLACE INTO identities (name, embedding)
            VALUES (?, ?)
        """, (name, blob))

        self.conn.commit()
        template_count = self._template_count(name)
        print(f"[DB] Registered template for '{name}' (total templates: {template_count}) source={source_video}")

    def _template_count(self, name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM identity_templates WHERE identity_name = ?",
            (name,)
        )
        return cursor.fetchone()[0]

    # ─────────────────────────────────────────────
    # FETCH IDENTITIES (multi-template)
    # ─────────────────────────────────────────────
    def get_all_identities(self) -> List[Tuple[str, List[np.ndarray]]]:
        """Return all identities with their template embeddings.

        Returns:
            List of (name, [emb1, emb2, ...]) tuples.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT identity_name, embedding FROM identity_templates ORDER BY identity_name")
        rows = cursor.fetchall()

        grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
        for name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            grouped[name].append(emb)

        return list(grouped.items())

    # ─────────────────────────────────────────────
    # LOG ENTRY WITH DATE + TIME
    # ─────────────────────────────────────────────
    def log_entry(self,
                  identity_name: Optional[str],
                  distance: float,
                  status: str):

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S.%f")[:-3]  # milliseconds

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO logs (identity_name, date, time, distance, status)
            VALUES (?, ?, ?, ?, ?)
        """, (
            identity_name,
            date_str,
            time_str,
            distance,
            status
        ))

        self.conn.commit()

    # ─────────────────────────────────────────────
    # MATCHING (multi-template: max sim per identity)
    # ─────────────────────────────────────────────
    def find_match(self,
                   query_embedding: np.ndarray,
                   threshold: float = 0.65,
                   margin: float = 0.02
                   ) -> Tuple[str, float]:
        """
        Find best matching identity using cosine similarity with margin check.

        For each identity the *maximum* similarity across all stored templates
        is used as the identity score.

        Returns:
            (identity_name, best_score): Best matching identity or ("Unknown", best_score)
        """
        identities = self.get_all_identities()

        if not identities:
            return "Unknown", 0.0

        query_embedding = self._normalize(query_embedding)

        sims = []
        for name, templates in identities:
            # Max similarity across all templates for this identity
            best_sim = max(
                float(np.dot(query_embedding, self._normalize(t)))
                for t in templates
            )
            sims.append((name, best_sim))

        # sort descending by similarity
        sims.sort(key=lambda x: x[1], reverse=True)
        print("Similarity scores:", sims)

        best_name, best_score = sims[0]

        # If only one identity, only apply threshold check
        if len(sims) == 1:
            if best_score >= threshold:
                return best_name, best_score
            return "Unknown", best_score

        second_best_score = sims[1][1]

        # Condition A: best meets threshold
        cond_a = best_score >= threshold

        # Condition B: margin between best and second best
        cond_b = (best_score - second_best_score) >= margin

        if cond_a and cond_b:
            return best_name, best_score

        return "Unknown", best_score

    # ─────────────────────────────────────────────
    @staticmethod
    def _normalize(v: np.ndarray):
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-12)

    # ─────────────────────────────────────────────
    # FRAME-LEVEL LOGGING
    # ─────────────────────────────────────────────
    def log_frame_prediction(self,
                            track_id: int,
                            person_name: Optional[str],
                            similarity: float,
                            decision: str):
        """
        Log a single frame-level recognition attempt.

        Args:
            track_id: Track ID from ByteTrack
            person_name: Predicted identity name or None
            similarity: Similarity score (0-1)
            decision: Decision type (e.g., "RECOGNIZED", "UNKNOWN", "MOTION_FILTER")
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO frame_predictions (track_id, person_name, similarity, decision)
            VALUES (?, ?, ?, ?)
        """, (track_id, person_name, similarity, decision))
        self.conn.commit()

    # ─────────────────────────────────────────────
    # TRACK-LEVEL AGGREGATION
    # ─────────────────────────────────────────────
    def get_track_predictions(self, track_id: int) -> List[Tuple[str, float, str]]:
        """
        Retrieve all frame predictions for a track.

        Returns:
            List of (person_name, similarity, decision) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_name, similarity, decision
            FROM frame_predictions
            WHERE track_id = ?
            ORDER BY timestamp ASC
        """, (track_id,))
        return cursor.fetchall()

    def compute_track_identity(self, track_id: int,
                              similarity_threshold: float = 0.5,
                              min_occurrences: int = 2) -> Tuple[Optional[str], float, float]:
        """
        Compute final identity for a track based on voting and similarity aggregation.

        Uses the candidate identity that appears multiple times with similarity
        above threshold.

        Args:
            track_id: Track ID to analyze
            similarity_threshold: Min similarity to count a prediction
            min_occurrences: Min number of occurrences to consider valid identity

        Returns:
            (identity, avg_similarity, max_similarity)
            identity is None if no valid candidate found
        """
        cursor = self.conn.cursor()

        # Get all predictions for this track
        cursor.execute("""
            SELECT person_name, similarity
            FROM frame_predictions
            WHERE track_id = ? AND person_name IS NOT NULL AND person_name != 'Unknown'
            ORDER BY timestamp ASC
        """, (track_id,))

        predictions = cursor.fetchall()

        if not predictions:
            return None, 0.0, 0.0

        # Filter predictions by similarity threshold
        valid_predictions = [(name, sim) for name, sim in predictions if sim >= similarity_threshold]

        if not valid_predictions:
            return None, 0.0, 0.0

        # Count occurrences of each identity
        identity_counts = {}
        identity_sims = defaultdict(list)

        for name, sim in valid_predictions:
            identity_counts[name] = identity_counts.get(name, 0) + 1
            identity_sims[name].append(sim)

        # Find identity with min_occurrences
        candidates = {name: count for name, count in identity_counts.items() if count >= min_occurrences}

        if not candidates:
            return None, 0.0, 0.0

        # Choose identity with highest average similarity
        best_identity = max(candidates.keys(), key=lambda n: float(np.mean(identity_sims[n])))

        avg_sim = float(np.mean(identity_sims[best_identity]))
        max_sim = float(np.max(identity_sims[best_identity]))

        return best_identity, avg_sim, max_sim

    def log_track_result(self,
                        track_id: int,
                        identity: Optional[str],
                        modality: str,
                        avg_similarity: float,
                        max_similarity: float,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None):
        """
        Log the final recognition result for a track.

        Args:
            track_id: Track ID
            identity: Final identity (or None/"Unknown")
            modality: Modality used ('gait', 'face', 'fusion', etc.)
            avg_similarity: Average similarity score for this identity
            max_similarity: Maximum similarity score for this identity
            start_time: Track start time (ISO format or None)
            end_time: Track end time (ISO format or None)
        """
        cursor = self.conn.cursor()

        # Get prediction count for this track
        cursor.execute("""
            SELECT COUNT(*) FROM frame_predictions
            WHERE track_id = ?
        """, (track_id,))

        prediction_count = cursor.fetchone()[0]

        # Get start_time if not provided
        if start_time is None:
            cursor.execute("""
                SELECT MIN(timestamp) FROM frame_predictions
                WHERE track_id = ?
            """, (track_id,))
            result = cursor.fetchone()
            start_time = result[0] if result and result[0] else datetime.now().isoformat()

        # Use current time if end_time not provided
        if end_time is None:
            end_time = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO recognition_results
            (track_id, identity, modality, avg_similarity, max_similarity, prediction_count, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (track_id, identity, modality, avg_similarity, max_similarity, prediction_count, start_time, end_time))

        self.conn.commit()

    def get_recognition_results(self, limit: int = 100) -> List[Tuple]:
        """
        Retrieve recent recognition results from database.

        Returns:
            List of (track_id, identity, modality, avg_similarity, max_similarity, prediction_count, start_time, end_time)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT track_id, identity, modality, avg_similarity, max_similarity, prediction_count, start_time, end_time
            FROM recognition_results
            ORDER BY end_time DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
