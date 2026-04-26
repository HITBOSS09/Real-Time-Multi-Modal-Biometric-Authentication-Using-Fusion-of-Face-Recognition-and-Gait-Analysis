"""
PostgreSQL database connection and models for face recognition service.

Handles:
- Connection pooling with psycopg2
- pgvector similarity search
- CRUD operations for persons, embeddings, and logs
- Configuration management
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import numpy as np

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '5432'))
        self.database = os.getenv('DB_NAME', 'face_recognition')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', 'postgres')
        
        # Connection pool settings
        self.min_pool_size = int(os.getenv('DB_MIN_POOL', '2'))
        self.max_pool_size = int(os.getenv('DB_MAX_POOL', '10'))
        
        logger.info(f"Database config: {self.host}:{self.port}/{self.database}")
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    def get_sync_connection_string(self) -> str:
        """Get connection string for psycopg2 (sync)."""
        return f"dbname={self.database} user={self.user} password={self.password} host={self.host} port={self.port}"


class DatabaseManager:
    """Manages PostgreSQL connections and database operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[SimpleConnectionPool] = None
    
    def _ensure_vector_registered(self, conn):
        """Register pgvector adapters/typecasters once per connection."""
        # psycopg2 connection objects allow custom attributes
        try:
            if getattr(conn, "_pgvector_registered", False):
                return
            register_vector(conn)
            setattr(conn, "_pgvector_registered", True)
        except Exception:
            # If attribute set fails for any reason, still register vector
            register_vector(conn)
        
    def init_sync_pool(self):
        """Initialize synchronous connection pool."""
        try:
            self.pool = SimpleConnectionPool(
                self.config.min_pool_size,
                self.config.max_pool_size,
                self.config.get_sync_connection_string()
            )
            logger.info("✓ Sync connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def close_sync_pool(self):
        """Close sync connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("✓ Sync connection pool closed")
    
    # ========================================================================
    # SCHEMA INITIALIZATION
    # ========================================================================
    
    def init_schema(self):
        """Initialize database schema with pgvector extension."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            
            # Enable pgvector extension
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("✓ pgvector extension enabled")
            except Exception as e:
                logger.warning(f"pgvector extension may already exist: {e}")
            
            # Create persons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✓ persons table created")
            
            # Create face_embeddings table with pgvector
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    embedding vector(512) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✓ face_embeddings table created")

            # If a legacy installation stored embeddings as TEXT/VARCHAR, migrate to vector(512)
            cursor.execute(
                """
                SELECT udt_name
                FROM information_schema.columns
                WHERE table_name = 'face_embeddings'
                  AND column_name = 'embedding';
                """
            )
            col = cursor.fetchone()
            if col and col[0] != 'vector':
                logger.warning(
                    "Legacy schema detected: face_embeddings.embedding is %s; migrating to vector(512)",
                    col[0],
                )
                # 1) Add temporary vector column
                cursor.execute(
                    "ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS embedding_vec vector(512);"
                )
                # 2) Cast legacy text like \"[0.1,0.2,...]\" directly to vector (no Python parsing)
                cursor.execute(
                    """
                    UPDATE face_embeddings
                    SET embedding_vec = embedding::vector
                    WHERE embedding_vec IS NULL;
                    """
                )
                # 3) Drop old column and rename new one
                cursor.execute("ALTER TABLE face_embeddings DROP COLUMN embedding;")
                cursor.execute("ALTER TABLE face_embeddings RENAME COLUMN embedding_vec TO embedding;")
                cursor.execute("ALTER TABLE face_embeddings ALTER COLUMN embedding SET NOT NULL;")
                logger.info("✓ Migrated face_embeddings.embedding to vector(512)")
            
            # Create index for pgvector similarity search
            # If we migrated columns, the old index (if any) is invalid; recreate safely
            cursor.execute("DROP INDEX IF EXISTS idx_embedding_vector;")
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_vector
                ON face_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )
            logger.info("✓ pgvector index created")
            
            # Create recognition_logs table (schema evolves via ALTERs below)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
                    name VARCHAR(255),
                    confidence VARCHAR(50),
                    similarity FLOAT NOT NULL,
                    decision VARCHAR(50) NOT NULL,
                    track_id VARCHAR(50),
                    camera_id VARCHAR(255),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✓ recognition_logs table created")
            # make sure old installations get new columns without dropping data
            cursor.execute("ALTER TABLE recognition_logs ADD COLUMN IF NOT EXISTS name VARCHAR(255);")
            cursor.execute("ALTER TABLE recognition_logs ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);")
            cursor.execute("ALTER TABLE recognition_logs ADD COLUMN IF NOT EXISTS track_id VARCHAR(50);")
            cursor.execute("ALTER TABLE recognition_logs ADD COLUMN IF NOT EXISTS camera_id VARCHAR(255);")
            logger.info("✓ recognition_logs columns ensured")
            
            # Create system_config table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key VARCHAR(255) PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✓ system_config table created")
            
            # Create fusion_results table (for multimodal fusion engine)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fusion_results (
                    id SERIAL PRIMARY KEY,
                    identity TEXT,
                    face_identity TEXT,
                    gait_identity TEXT,
                    face_score FLOAT,
                    gait_score FLOAT,
                    fused_score FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✓ fusion_results table created")
            
            conn.commit()
            logger.info("✓ Database schema initialized successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Schema initialization failed: {e}")
            raise
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    # ========================================================================
    # PERSON OPERATIONS
    # ========================================================================
    
    def create_person(self, name: str) -> int:
        """Create a new person. Returns person_id."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO persons (name) VALUES (%s) RETURNING id;",
                (name,)
            )
            person_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Created person: {name} (id={person_id})")
            return person_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create person {name}: {e}")
            raise
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Retrieve person by ID."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM persons WHERE id = %s;", (person_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """Retrieve person by name."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM persons WHERE name = %s;", (name,))
            result = cursor.fetchone()
            return dict(result) if result else None
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    # ========================================================================
    # FACE EMBEDDING OPERATIONS
    # ========================================================================
    
    def store_embedding(self, person_id: int, embedding: np.ndarray) -> int:
        """Store face embedding for a person. Returns embedding_id."""
        conn = self.pool.getconn()
        try:
            self._ensure_vector_registered(conn)
            cursor = conn.cursor()

            # Store as pgvector (vector(512)) via adapter; never stringify
            emb = np.asarray(embedding, dtype=np.float32)
            cursor.execute(
                "INSERT INTO face_embeddings (person_id, embedding) VALUES (%s, %s) RETURNING id;",
                (person_id, emb)
            )
            embedding_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Stored embedding for person {person_id} (embedding_id={embedding_id})")
            return embedding_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store embedding: {e}")
            raise
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    def get_face_sample_count(self, person_id: int) -> int:
        """Return number of face embeddings stored for the given person."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM face_embeddings WHERE person_id = %s;",
                (person_id,)
            )
            count = cursor.fetchone()[0]
            return count
        finally:
            cursor.close()
            self.pool.putconn(conn)

    # ------------------------------------------------------------------------
    def get_all_embeddings(self) -> list[Dict]:
        """Return all stored face embeddings along with person metadata.

        The return value is a list of dictionaries containing ``person_id``,
        ``person_name`` and ``embedding`` (list of floats).  This is used by the
        CCTV engine to load vectors into memory at startup for real‑time
        matching without touching the database on every frame.
        """
        conn = self.pool.getconn()
        try:
            self._ensure_vector_registered(conn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                """
                SELECT fe.person_id, p.name AS person_name, fe.embedding
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id;
                """
            )
            rows = cursor.fetchall()
            out: list[Dict] = []
            for r in rows:
                d = dict(r)
                emb = d.get("embedding")
                # With pgvector typecasters, this is already a numeric array-like
                if isinstance(emb, np.ndarray):
                    d["embedding"] = emb.astype(np.float32, copy=False)
                else:
                    # If typecasters weren't registered, we'd get a string here;
                    # we intentionally do not parse strings in production.
                    raise RuntimeError(
                        "Embedding fetched as non-numeric type. "
                        "pgvector typecasters not registered or schema is incorrect."
                    )
                out.append(d)
            return out
        finally:
            cursor.close()
            self.pool.putconn(conn)

    def find_similar_embeddings(
        self, 
        query_embedding: np.ndarray, 
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Find similar embeddings using pgvector cosine similarity.
        
        Returns list of dicts with:
        - id: embedding_id
        - person_id: person_id
        - person_name: name
        - similarity: cosine similarity score (0-1)
        """
        conn = self.pool.getconn()
        try:
            self._ensure_vector_registered(conn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            q = np.asarray(query_embedding, dtype=np.float32)
            # pgvector cosine similarity: 1 - (distance)
            cursor.execute(
                """
                SELECT 
                    fe.id,
                    fe.person_id,
                    p.name as person_name,
                    (1 - (fe.embedding <=> %s))::float as similarity
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
                WHERE (1 - (fe.embedding <=> %s)) >= %s
                ORDER BY fe.embedding <=> %s
                LIMIT %s;
                """,
                (q, q, threshold, q, limit),
            )
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    # ========================================================================
    # CONFIGURATION OPERATIONS
    # ========================================================================
    
    def set_config(self, key: str, value: str):
        """Set a configuration value."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_config (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP;",
                (key, value)
            )
            conn.commit()
            logger.info(f"Config updated: {key}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to set config {key}: {e}")
            raise
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """Get a configuration value."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = %s;", (key,))
            result = cursor.fetchone()
            return result[0] if result else default
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    def get_config_float(self, key: str, default: float = None) -> Optional[float]:
        """Get a configuration value as float."""
        value = self.get_config(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Config {key} is not a valid float: {value}")
            return default
    
    # ========================================================================
    # RECOGNITION LOGGING
    # ========================================================================
    
    def log_recognition(
        self,
        person_id: Optional[int],
        similarity: float,
        decision: str,
        name: Optional[str] = None,
        confidence: Optional[str] = None,
        track_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        cooldown_sec: int = 45,
    ):
        """Log a recognition attempt with optional metadata.

        This method implements database-level cooldown protection: if a row for
        ``person_id`` already exists within the last ``cooldown_sec`` seconds the
        insert is skipped.  This is the second layer of duplicate protection; a
        memory-based cooldown is expected to run in the video processor.
        """
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            # skip if within cooldown window
            if person_id is not None:
                cursor.execute(
                    """
                    SELECT 1 FROM recognition_logs
                    WHERE person_id = %s
                      AND timestamp > NOW() - INTERVAL '%s seconds'
                    LIMIT 1;
                    """,
                    (person_id, cooldown_sec)
                )
                if cursor.fetchone():
                    logger.debug(f"DB cooldown: skipping log for person {person_id}")
                    return

            cursor.execute(
                """
                INSERT INTO recognition_logs
                    (person_id, name, confidence, similarity, decision, track_id, camera_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (person_id, name, confidence, similarity, decision, track_id, camera_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log recognition: {e}")
        finally:
            cursor.close()
            self.pool.putconn(conn)
    
    def get_recognition_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent recognition logs."""
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT rl.*, p.name as person_name
                FROM recognition_logs rl
                LEFT JOIN persons p ON rl.person_id = p.id
                ORDER BY rl.timestamp DESC
                LIMIT %s;
            """, (limit,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
            self.pool.putconn(conn)

    def log_fusion_result(
        self,
        identity: Optional[str],
        face_identity: Optional[str],
        gait_identity: Optional[str],
        face_score: float,
        gait_score: float,
        fused_score: float,
    ):
        """Log a fusion result with face, gait, and fused scores.
        
        Args:
            identity: Final fused identity
            face_identity: Identity from face recognition
            gait_identity: Identity from gait recognition
            face_score: Similarity score from face
            gait_score: Similarity score from gait
            fused_score: Final fused score
        """
        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fusion_results
                    (identity, face_identity, gait_identity, face_score, gait_score, fused_score)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (identity, face_identity, gait_identity, face_score, gait_score, fused_score)
            )
            conn.commit()
            logger.info(
                f"Logged fusion result: fused={identity} (score={fused_score:.3f}), "
                f"face={face_identity} ({face_score:.3f}), gait={gait_identity} ({gait_score:.3f})"
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log fusion result: {e}")
        finally:
            cursor.close()
            self.pool.putconn(conn)


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def init_db():
    """Initialize global database manager."""
    global db_manager
    config = DatabaseConfig()
    db_manager = DatabaseManager(config)
    db_manager.init_sync_pool()
    db_manager.init_schema()
    return db_manager


def get_db() -> DatabaseManager:
    """Get global database manager."""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return db_manager
