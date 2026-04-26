"""
FastAPI routes for face recognition service.

Endpoints:
- POST /create_person - Create person record
- POST /add_face_samples/{person_id} - Add multiple face samples
- GET /person/{person_id}/samples - Get person info and sample count
- POST /register_face - Register new person (legacy single-sample endpoint)
- POST /recognize_face - Identify face in image
- GET /health - Health check
- GET /config - Get system configuration
- GET /logs - Get recent recognition logs
"""

import logging
from typing import Optional
import base64
from io import BytesIO
from time import time
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, Path
from pydantic import BaseModel, Field

from .db import get_db
from .face_engine import get_face_engine
from .threshold import get_threshold_manager
from .registration_service import RegistrationService
from .decision_manager import get_decision_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreatePersonRequest(BaseModel):
    """Request model for creating a person."""
    person_name: str = Field(..., min_length=1, max_length=255)


class CreatePersonResponse(BaseModel):
    """Response model for person creation."""
    success: bool
    message: str
    person_id: Optional[int] = None
    person_name: Optional[str] = None


class AddFaceSamplesResponse(BaseModel):
    """Response model for adding face samples."""
    success: bool
    message: str
    person_id: int
    samples_added: int
    total_samples_for_person: Optional[int] = None


class PersonInfoResponse(BaseModel):
    """Response model for person info."""
    success: bool
    message: str
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    total_face_samples: Optional[int] = None
    created_at: Optional[str] = None


class RegisterFaceRequest(BaseModel):
    """Request model for face registration."""
    person_name: str = Field(..., min_length=1, max_length=255)


class RegisterFaceResponse(BaseModel):
    """Response model for face registration."""
    success: bool
    message: str
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    embedding_id: Optional[int] = None


class RecognitionResult(BaseModel):
    """Recognition result for a detected face."""
    decision: str  # "RECOGNIZED" or "UNKNOWN"
    person_name: Optional[str] = None
    similarity: float
    threshold: float
    confidence: str  # "low", "medium", "high"
    bbox: list[int]  # [x1, y1, x2, y2] from InsightFace detection (required, no default)


class RecognizeFaceResponse(BaseModel):
    """Response model for face recognition."""
    success: bool
    message: str
    faces_detected: int
    results: list[RecognitionResult] = []


class ConfigResponse(BaseModel):
    """System configuration response."""
    eer_threshold: float
    embedding_dim: int
    provider: str
    detection_threshold: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy" or "degraded"
    database_ready: bool
    gpu_available: bool
    provider: str
    timestamp: str


class LogEntry(BaseModel):
    """Recognition log entry with extended metadata."""
    id: int
    person_name: Optional[str]
    similarity: float
    confidence: Optional[str] = None
    decision: str
    track_id: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: str


class LogsResponse(BaseModel):
    """Recent logs response."""
    success: bool
    count: int
    logs: list[LogEntry] = []


# ============================================================================
# ENDPOINTS - PERSON & ENROLLMENT MANAGEMENT
# ============================================================================

@router.post("/create_person", response_model=CreatePersonResponse, status_code=201)
async def create_person(request: CreatePersonRequest):
    """
    Create a new person record.
    
    This is the first step in multi-sample enrollment.
    After creating a person, use /add_face_samples/{person_id} to add samples.
    
    Args:
        request: Person name
        
    Returns:
        Created person with ID
    """
    try:
        db = get_db()
        engine = get_face_engine()
        reg_service = RegistrationService(db, engine)
        
        success, message, person_id = reg_service.create_person(request.person_name)
        
        status_code = 201 if success else (409 if "already exists" in message else 400)
        
        if not success:
            # Return error as 4xx
            raise HTTPException(
                status_code=status_code,
                detail=message
            )
        
        return CreatePersonResponse(
            success=True,
            message=message,
            person_id=person_id,
            person_name=request.person_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create person: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/add_face_samples/{person_id}", response_model=AddFaceSamplesResponse)
async def add_face_samples(
    person_id: int = Path(..., gt=0),
    files: list[UploadFile] = File(...)
):
    """
    Add multiple face samples to an existing person.
    
    Validation:
    - Exactly one face per image
    - Maximum 50 images per request
    - Only normalized embeddings stored
    
    Args:
        person_id: Person ID (must exist)
        files: List of image files
        
    Returns:
        Number of samples successfully added and total count
    """
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No image files provided")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 images per request")
        
        db = get_db()
        engine = get_face_engine()
        reg_service = RegistrationService(db, engine)
        
        # Check person exists
        person = db.get_person_by_id(person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        # Read image data from uploaded files
        image_data_list = []
        for file in files:
            image_data = await file.read()
            if image_data:
                image_data_list.append(image_data)
        
        if not image_data_list:
            raise HTTPException(status_code=400, detail="No valid image data")
        
        # Add face samples
        success, message, samples_added, total_samples = reg_service.add_face_samples(
            person_id,
            image_data_list
        )
        
        if not success and samples_added == 0:
            raise HTTPException(status_code=400, detail=message)
        
        return AddFaceSamplesResponse(
            success=success,
            message=message,
            person_id=person_id,
            samples_added=samples_added,
            total_samples_for_person=total_samples
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add face samples: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/person/{person_id}/samples", response_model=PersonInfoResponse)
async def get_person_samples(person_id: int = Path(..., gt=0)):
    """
    Get person information including total face samples.
    
    Args:
        person_id: Person ID
        
    Returns:
        Person info with sample count
    """
    try:
        db = get_db()
        engine = get_face_engine()
        reg_service = RegistrationService(db, engine)
        
        success, message, info = reg_service.get_person_info(person_id)
        
        if not success:
            raise HTTPException(
                status_code=404 if "not found" in message else 500,
                detail=message
            )
        
        return PersonInfoResponse(
            success=True,
            message="OK",
            person_id=info["person_id"],
            person_name=info["person_name"],
            total_face_samples=info["total_face_samples"],
            created_at=info["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get person samples: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/add_gait_samples/{person_id}",
    response_model=AddFaceSamplesResponse,
    deprecated=True
)
async def add_gait_samples(
    person_id: int = Path(..., gt=0),
    files: list[UploadFile] = File(...)
):
    """
    TODO: Add gait samples for a person (not yet implemented).
    
    This endpoint is a placeholder for future gait recognition integration.
    Will support uploading video files for gait-based biometric enrollment.
    
    Args:
        person_id: Person ID
        files: List of gait video files (not yet supported)
        
    Returns:
        Not implemented
    """
    raise HTTPException(
        status_code=501,
        detail="Gait recognition module not yet implemented"
    )


# ============================================================================
# ENDPOINTS - HEALTH & DIAGNOSTICS
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - returns service status and dependencies."""
    try:
        from datetime import datetime
        
        db = get_db()
        engine = get_face_engine()
        
        # Test database connection by checking if schema is initialized
        db_ready = False
        try:
            # Query system_config table to verify database is initialized
            config = db.get_config('eer_threshold')
            db_ready = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            db_ready = False
        
        # Check GPU availability
        gpu_available = 'CUDA' in engine.provider_used
        
        # If database is not ready, return degraded status
        if not db_ready:
            logger.warning("Health check: Database not ready")
            return HealthResponse(
                status='degraded',
                database_ready=False,
                gpu_available=gpu_available,
                provider=engine.provider_used,
                timestamp=datetime.now().isoformat()
            )
        
        return HealthResponse(
            status='healthy',
            database_ready=True,
            gpu_available=gpu_available,
            provider=engine.provider_used,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get system configuration."""
    try:
        db = get_db()
        engine = get_face_engine()
        threshold_mgr = get_threshold_manager()
        
        return ConfigResponse(
            eer_threshold=threshold_mgr.get_threshold(),
            embedding_dim=engine.config.embedding_dim,
            provider=engine.provider_used,
            detection_threshold=engine.config.detection_threshold
        )
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS - RECOGNITION (Legacy & Main)
# ============================================================================



@router.post("/register_face", response_model=RegisterFaceResponse)
async def register_face(
    file: UploadFile = File(...),
    person_name: str = Form(...)
):
    """
    Register a new person with face image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        person_name: Name of person to register
        
    Returns:
        Registration result with person_id and embedding_id
    """
    if not person_name or len(person_name.strip()) == 0:
        raise HTTPException(status_code=400, detail="person_name cannot be empty")
    
    person_name = person_name.strip()
    
    try:
        # Read image file
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Get engines
        db = get_db()
        engine = get_face_engine()
        
        # Check if person already exists
        existing = db.get_person_by_name(person_name)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Person '{person_name}' already registered"
            )
        
        # Process image and detect faces
        try:
            image_bgr, faces = engine.process_image_file(image_data)
        except Exception as e:
            logger.error(f"Face processing failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Could not process image - please provide a clear face image"
            )
        
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected. Using first one.")
        
        # Extract embedding from first face
        embedding = faces[0].embedding
        
        # Normalize embedding
        import numpy as np
        embedding = embedding / np.linalg.norm(embedding)
        
        # Create person in database
        person_id = db.create_person(person_name)
        
        # Store embedding
        embedding_id = db.store_embedding(person_id, embedding)
        
        logger.info(f"Registered {person_name} (id={person_id}) with embedding {embedding_id}")
        
        return RegisterFaceResponse(
            success=True,
            message=f"Successfully registered {person_name}",
            person_id=person_id,
            person_name=person_name,
            embedding_id=embedding_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/recognize_face", response_model=RecognizeFaceResponse)
async def recognize_face(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=100),
    track_id: Optional[str] = Query(None, description="Optional tracker ID from video processor"),
    camera_id: Optional[str] = Query(None, description="Optional camera identifier from video processor"),
):
    """
    Recognize faces in image with per-identity cooldown protection.

    Multi-face detection using InsightFace + RetinaFace. For each detected face:
    - Extract embedding
    - Compare with pgvector DB
    - Return recognition decision
    - Log to DB only if not in cooldown window (30-45 sec per identity)

    Args:
        file: Image file (JPEG, PNG, etc.)
        top_k: Number of top similar faces to return
        camera_id: Optional camera identifier for logging
        track_id: Optional track ID for logging

    Returns:
        Recognition results with matched persons and similarity scores
    """
    try:
        # Read image file
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Get engines
        db = get_db()
        engine = get_face_engine()
        threshold_mgr = get_threshold_manager()
        
        # Process image and detect faces
        try:
            image_bgr, faces = engine.process_image_file(image_data)
        except Exception as e:
            # if processing fails, log and treat as unknown
            logger.error(f"Face processing failed: {e}")
            # log to recognition table with UNKNOWN decision
            try:
                db.log_recognition(
                    person_id=None,
                    similarity=0.0,
                    decision='UNKNOWN',
                    camera_id=camera_id
                )
            except Exception:
                pass
            return RecognizeFaceResponse(
                success=True,
                message="Face processing error; treated as UNKNOWN",
                faces_detected=0,
                results=[]
            )
        
        if not faces:
            return RecognizeFaceResponse(
                success=True,
                message="No faces detected in image",
                faces_detected=0,
                results=[]
            )
        
        results = []
        decision_mgr = get_decision_manager()
        
        # Process each detected face
        for face_idx, face in enumerate(faces):
            # Extract embedding
            embedding = face.embedding

            # Normalize
            import numpy as np
            embedding = embedding / np.linalg.norm(embedding)

            # Extract bounding box from InsightFace detection
            bbox = [int(b) for b in face.bbox]
            logger.debug(f"Face {face_idx}: extracted bbox={bbox}")

            # Prepare default values for logging/response
            person_name: Optional[str]
            similarity: float
            person_id: Optional[int]
            decision_str: str
            threshold_val: float
            confidence_val: str

            # Search for similar embeddings in database
            similar = db.find_similar_embeddings(
                embedding,
                limit=top_k,
                threshold=0.0  # Return all, we'll filter by threshold
            )

            if similar:
                # recognized candidate
                best_match = similar[0]
                person_name = best_match['person_name']
                person_id = best_match['person_id']
                similarity = best_match['similarity']

                thr = threshold_mgr.get_decision(similarity, person_name)
                decision_str = thr['decision']
                threshold_val = thr['threshold']
                confidence_val = thr['confidence']
            else:
                # truly unknown face
                person_name = None
                person_id = None
                similarity = 0.0
                decision_str = 'UNKNOWN'
                threshold_val = threshold_mgr.get_threshold()
                confidence_val = 'low'

            # Apply DecisionManager for stability + cooldown (works for both
            # recognized and unknown events)
            recognition_decision = decision_mgr.process_recognition(
                track_id=track_id,
                person_name=person_name,
                similarity=similarity,
                decision_str=decision_str,
                threshold=threshold_val,
                confidence=confidence_val,
                camera_id=camera_id,
            )

            # Persist log only if DM indicates we should
            if recognition_decision.should_log:
                log_person_id = person_id if recognition_decision.decision == 'RECOGNIZED' else None
                db.log_recognition(
                    person_id=log_person_id,
                    name=recognition_decision.person_name,
                    confidence=recognition_decision.confidence,
                    similarity=similarity,
                    decision=recognition_decision.decision,
                    track_id=track_id,
                    camera_id=camera_id,
                )
            else:
                reason = 'cooldown' if recognition_decision.is_stable else 'unstable'
                name_display = person_name or '<UNKNOWN>'
                logger.debug(
                    f"Skipping DB log: track={track_id}, person={name_display}, "
                    f"reason={reason}"
                )

            # Build API response entry.  We always return the original
            # threshold decision for backward compatibility; stability only
            # affects whether we surface a name or show PROCESSING.
            display_person = recognition_decision.person_name if recognition_decision.is_stable else None
            results.append(RecognitionResult(
                decision=decision_str,
                person_name=display_person,
                similarity=similarity,
                threshold=threshold_val,
                confidence=confidence_val,
                bbox=bbox
            ))
        
        logger.info(
            f"Recognized {len(results)} face(s) from {camera_id or 'unknown-camera'}: "
            f"decisions={[r.decision for r in results]}"
        )
        
        return RecognizeFaceResponse(
            success=True,
            message=f"Processed {len(faces)} face(s)",
            faces_detected=len(faces),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


@router.get("/logs", response_model=LogsResponse)
async def get_logs(limit: int = Query(50, ge=1, le=500)):
    """Get recent recognition logs."""
    try:
        db = get_db()
        logs = db.get_recognition_logs(limit=limit)
        
        log_entries = []
        for log in logs:
            log_entries.append(LogEntry(
                id=log['id'],
                person_name=log.get('person_name') or log.get('name'),
                similarity=log['similarity'],
                confidence=log.get('confidence'),
                decision=log['decision'],
                track_id=log.get('track_id'),
                camera_id=log.get('camera_id'),
                timestamp=log['timestamp'].isoformat() if log['timestamp'] else None,
            ))
        
        return LogsResponse(
            success=True,
            count=len(log_entries),
            logs=log_entries
        )
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
