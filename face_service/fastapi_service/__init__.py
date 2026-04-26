"""Face Recognition FastAPI Service."""

__version__ = "1.0.0"

from .db import init_db, get_db, DatabaseManager
from .face_engine import init_face_engine, get_face_engine, FaceEngine
from .threshold import init_threshold_manager, get_threshold_manager, ThresholdManager
from .main import app

__all__ = [
    'init_db',
    'get_db',
    'DatabaseManager',
    'init_face_engine',
    'get_face_engine',
    'FaceEngine',
    'init_threshold_manager',
    'get_threshold_manager',
    'ThresholdManager',
    'app',
]
