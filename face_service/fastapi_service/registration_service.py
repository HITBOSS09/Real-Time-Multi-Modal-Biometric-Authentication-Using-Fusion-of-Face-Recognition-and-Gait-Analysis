"""
Registration service for multi-sample face enrollment.

Handles:
- Person creation
- Multi-sample face registration per person
- Embedding extraction and normalization
- Validation (exactly one face per image)
- Error handling with proper logging

Design supports future:
- Gait embeddings
- Other biometric modalities
"""

import logging
from typing import Optional, List, Tuple
import numpy as np

from .db import DatabaseManager
from .face_engine import FaceEngine

logger = logging.getLogger(__name__)


class RegistrationService:
    """
    Service for managing person enrollment and face sample registration.
    
    Workflow:
    1. Create person record
    2. Add multiple face samples to person
    3. Track samples per person
    4. Support future gait/other modalities
    """
    
    def __init__(self, db: DatabaseManager, engine: FaceEngine):
        """
        Initialize registration service.
        
        Args:
            db: Database manager instance
            engine: FaceEngine instance
        """
        self.db = db
        self.engine = engine
    
    # ========================================================================
    # PERSON MANAGEMENT
    # ========================================================================
    
    def create_person(self, person_name: str) -> Tuple[bool, str, Optional[int]]:
        """
        Create a new person record.
        
        Args:
            person_name: Name of the person (must be unique)
            
        Returns:
            (success, message, person_id)
        """
        try:
            # Validate input
            if not person_name or not isinstance(person_name, str):
                return False, "Invalid person name", None
            
            person_name = person_name.strip()
            if len(person_name) < 1 or len(person_name) > 255:
                return False, "Person name must be 1-255 characters", None
            
            # Check if person already exists
            existing = self.db.get_person_by_name(person_name)
            if existing:
                msg = f"Person '{person_name}' already exists (id={existing['id']})"
                logger.warning(msg)
                return False, msg, existing['id']
            
            # Create person
            person_id = self.db.create_person(person_name)
            logger.info(f"Created person: {person_name} (id={person_id})")
            
            return True, f"Person '{person_name}' created successfully", person_id
            
        except Exception as e:
            msg = f"Failed to create person '{person_name}': {str(e)}"
            logger.error(msg)
            return False, msg, None
    
    def get_person_info(self, person_id: int) -> Tuple[bool, str, Optional[dict]]:
        """
        Get person information and sample count.
        
        Args:
            person_id: Person ID
            
        Returns:
            (success, message, person_info_dict)
        """
        try:
            person = self.db.get_person_by_id(person_id)
            if not person:
                return False, f"Person {person_id} not found", None
            
            # Count face samples for this person
            sample_count = self._get_sample_count(person_id)
            
            info = {
                "person_id": person_id,
                "person_name": person["name"],
                "total_face_samples": sample_count,
                "created_at": str(person.get("created_at", ""))
            }
            
            return True, "OK", info
            
        except Exception as e:
            msg = f"Failed to get person info: {str(e)}"
            logger.error(msg)
            return False, msg, None
    
    # ========================================================================
    # FACE SAMPLE REGISTRATION
    # ========================================================================
    
    def add_face_samples(
        self,
        person_id: int,
        image_files: List[bytes]
    ) -> Tuple[bool, str, int, Optional[int]]:
        """
        Add multiple face samples for a person.
        
        Args:
            person_id: Person ID
            image_files: List of image file bytes
            
        Returns:
            (success, message, samples_added, total_samples_for_person)
        """
        try:
            # Validate person exists
            person = self.db.get_person_by_id(person_id)
            if not person:
                return False, f"Person {person_id} not found", 0, None
            
            # Validate input
            if not image_files or len(image_files) == 0:
                return False, "No image files provided", 0, None
            
            if len(image_files) > 50:
                return False, "Maximum 50 images per request", 0, None
            
            # Process each image
            samples_added = 0
            errors = []
            
            for idx, image_data in enumerate(image_files, 1):
                try:
                    # Process image and extract embedding
                    success, msg, embedding = self._process_face_image(image_data)
                    
                    if not success:
                        errors.append(f"Image {idx}: {msg}")
                        continue
                    
                    # Store embedding
                    embedding_id = self.db.store_embedding(person_id, embedding)
                    samples_added += 1
                    logger.info(f"Added sample {idx} for person {person_id} (embedding_id={embedding_id})")
                    
                except Exception as e:
                    error_msg = f"Image {idx}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Get total sample count
            total_samples = self._get_sample_count(person_id)
            
            # Build response message
            msg_parts = [f"{samples_added}/{len(image_files)} samples added"]
            if errors:
                msg_parts.append(f"Errors: {'; '.join(errors[:3])}")
                if len(errors) > 3:
                    msg_parts.append(f"... and {len(errors)-3} more")
            
            message = "; ".join(msg_parts)
            success = samples_added > 0
            
            if success:
                logger.info(f"Successfully added {samples_added} samples for person {person_id}")
            
            return success, message, samples_added, total_samples
            
        except Exception as e:
            msg = f"Failed to add face samples: {str(e)}"
            logger.error(msg)
            return False, msg, 0, None
    
    # ========================================================================
    # VALIDATION & HELPER METHODS
    # ========================================================================
    
    def _process_face_image(self, image_data: bytes) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Process single face image: load, detect, validate, extract embedding.
        
        Validation:
        - Exactly one face per image
        - Face detected with sufficient confidence
        - Valid embedding extracted
        
        Returns:
            (success, message, embedding_normalized)
        """
        try:
            # Load image
            try:
                image_bgr, faces = self.engine.process_image_file(image_data)
            except Exception as e:
                return False, f"Failed to load image: {str(e)}", None
            
            # Validate exactly one face
            if len(faces) == 0:
                return False, "No faces detected in image", None
            
            if len(faces) > 1:
                return False, f"Multiple faces detected ({len(faces)}), expected exactly 1", None
            
            # Extract embedding from the single face
            face = faces[0]
            embedding = face.embedding
            
            if embedding is None:
                return False, "Failed to extract embedding", None
            
            # Normalize embedding
            try:
                embedding = np.asarray(embedding, dtype=np.float64)
                norm = np.linalg.norm(embedding)
                
                if norm < 1e-8:
                    return False, "Embedding norm too small", None
                
                embedding = embedding / norm
                
            except Exception as e:
                return False, f"Failed to normalize embedding: {str(e)}", None
            
            logger.debug(f"Successfully processed face: embedding shape={embedding.shape}")
            return True, "OK", embedding
            
        except Exception as e:
            msg = f"Image processing failed: {str(e)}"
            logger.error(msg)
            return False, msg, None
    
    def _get_sample_count(self, person_id: int) -> int:
        """
        Get total number of face samples for a person.
        
        Args:
            person_id: Person ID
            
        Returns:
            Sample count
        """
        try:
            return self.db.get_face_sample_count(person_id)
        except Exception as e:
            logger.error(f"Failed to get sample count for person {person_id}: {e}")
            return 0
    
    # ========================================================================
    # FUTURE: GAIT EMBEDDING SUPPORT (Placeholder)
    # ========================================================================
    
    def add_gait_samples(
        self,
        person_id: int,
        video_files: List[bytes]
    ) -> Tuple[bool, str, int, Optional[int]]:
        """
        TODO: Add gait samples for a person.
        
        Placeholder for future gait recognition integration.
        
        Args:
            person_id: Person ID
            video_files: List of video file bytes
            
        Returns:
            (success, message, samples_added, total_gait_samples)
        """
        msg = "Gait recognition module not yet implemented"
        logger.info(msg)
        return False, msg, 0, None
    
    def get_gait_sample_count(self, person_id: int) -> int:
        """
        TODO: Get total gait samples for a person.
        
        Placeholder for future gait recognition integration.
        """
        return 0
