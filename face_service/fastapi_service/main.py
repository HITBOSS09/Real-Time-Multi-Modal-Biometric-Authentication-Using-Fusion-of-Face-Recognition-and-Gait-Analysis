"""
Main FastAPI application for face recognition service.

Initialization:
- Startup: Initialize database, face engine, threshold manager
- Shutdown: Clean up resources
- Diagnostics: Log startup information
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    import onnxruntime as rt
except ImportError:  # pragma: no cover
    rt = None

from .db import init_db, DatabaseConfig, get_db
from .face_engine import init_face_engine, FaceEngineConfig, get_face_engine
from .threshold import init_threshold_manager, get_threshold_manager
from .decision_manager import init_decision_manager
from . import routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFECYCLE MANAGEMENT
# ============================================================================

async def startup_event():
    """Initialize resources on startup."""
    logger.info("=" * 80)
    logger.info("FACE RECOGNITION SERVICE STARTUP")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # 1. Log environment configuration
    logger.info("\n[DATABASE CONFIGURATION]")
    db_config = DatabaseConfig()
    logger.info(f"  Host: {db_config.host}:{db_config.port}")
    logger.info(f"  Database: {db_config.database}")
    logger.info(f"  User: {db_config.user}")
    
    # 2. Initialize database
    logger.info("\n[INITIALIZING DATABASE]")
    try:
        db = init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise
    
    # 3. Log ONNX Runtime providers (if available)
    logger.info("\n[ONNX RUNTIME PROVIDERS]")
    if rt is not None:
        logger.info(rt.get_available_providers())
    else:
        logger.warning("onnxruntime not installed; GPU/CPU providers unavailable")
    available_providers = rt.get_available_providers()
    for provider in available_providers:
        marker = "•" if provider != "CPUExecutionProvider" else "•"
        logger.info(f"  {marker} {provider}")
    
    # If CUDA available, log CUDA info
    if 'CUDAExecutionProvider' in available_providers:
        try:
            import torch
            logger.info("\n[PyTorch CUDA INFO]")
            logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"  CUDA Version: {torch.version.cuda}")
        except ImportError:
            pass
    
    # 4. Initialize face engine
    logger.info("\n[INITIALIZING FACE ENGINE]")
    try:
        engine_config = FaceEngineConfig()
        engine = init_face_engine(engine_config)
        logger.info(f"✓ Face engine initialized")
        logger.info(f"  Provider Used: {engine.provider_used}")
        logger.info(f"  Embedding Dimension: {engine.config.embedding_dim}")
        logger.info("  Model Directory: default (insightface internal)")
    except Exception as e:
        logger.error(f"✗ Face engine initialization failed: {e}")
        raise
    
    # 5. Initialize threshold manager
    logger.info("\n[INITIALIZING THRESHOLD MANAGER]")
    try:
        threshold_mgr = init_threshold_manager(db)
        logger.info("✓ Threshold manager initialized")
        logger.info(f"  EER Threshold: {threshold_mgr.get_threshold()}")
    except Exception as e:
        logger.error(f"✗ Threshold manager initialization failed: {e}")
        raise
    
    # 6. Initialize decision manager (stability + identity-based cooldown)
    logger.info("\n[INITIALIZING DECISION MANAGER]")
    try:
        decision_mgr = init_decision_manager(
            stability_threshold=3,  # Require 3 consecutive recognitions
            cooldown_seconds=30.0,  # 30-second identity-based cooldown
        )
        logger.info("✓ Decision manager initialized")
        logger.info(f"  Stability Threshold: {decision_mgr.stability_threshold} frames")
        logger.info(f"  Cooldown Window: {decision_mgr.cooldown_seconds} seconds")
        logger.info("  Unknown events also use stability + per-camera cooldown")
    except Exception as e:
        logger.error(f"✗ Decision manager initialization failed: {e}")
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL SERVICES INITIALIZED SUCCESSFULLY")
    logger.info("=" * 80 + "\n")


async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("\n" + "=" * 80)
    logger.info("FACE RECOGNITION SERVICE SHUTDOWN")
    logger.info("=" * 80)
    
    try:
        db = get_db()
        db.close_sync_pool()
        logger.info("✓ Database connection pool closed")
    except Exception as e:
        logger.warning(f"Database cleanup error: {e}")
    
    logger.info("✓ Service shutdown complete")
    logger.info("=" * 80 + "\n")


# ============================================================================
# APPLICATION FACTORY
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    await startup_event()
    yield
    await shutdown_event()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Face Recognition Service",
        description="Face detection, embedding extraction, and recognition using GPU-accelerated ONNX Runtime",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(routes.router)
    
    return app


# Create application instance
app = create_app()


# ============================================================================
# STARTUP EVENTS FOR Uvicorn
# ============================================================================

@app.on_event("startup")
async def on_startup():
    """Called when Uvicorn starts."""
    await startup_event()


@app.on_event("shutdown")
async def on_shutdown():
    """Called when Uvicorn shuts down."""
    await shutdown_event()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
