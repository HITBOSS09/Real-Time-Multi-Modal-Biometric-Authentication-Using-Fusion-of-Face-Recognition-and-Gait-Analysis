"""
Face recognition engine with GPU/CPU fallback using ONNX Runtime.

Handles:
- Face detection using SCRFD
- Face alignment and preprocessing
- Embedding extraction using InsightFace buffalo_l
- Automatic CUDA→CPU fallback
- Provider diagnostics and logging
"""

import os
import logging
from typing import Optional, Tuple, List
import numpy as np
from datetime import datetime
from io import BytesIO

import cv2
from PIL import Image

try:
    import onnxruntime as rt
except ImportError:  # pragma: no cover
    rt = None

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None

logger = logging.getLogger(__name__)


class FaceEngineConfig:
    """Configuration for face engine."""

    def __init__(self):
        self.use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
        self.device_id = int(os.getenv('DEVICE_ID', '0'))

        self.embedding_dim = 512
        self.detection_threshold = float(os.getenv('DETECTION_THRESHOLD', '0.5'))
        # Allow multiple faces per frame (default 5, configurable via env)
        self.max_faces_per_image = int(os.getenv('MAX_FACES_PER_IMAGE', '5'))

        logger.info(f"FaceEngine config: use_gpu={self.use_gpu}, device_id={self.device_id}")


class FaceEngine:
    """Face detection and embedding extraction engine."""

    def __init__(self, config: FaceEngineConfig = None):
        self.config = config or FaceEngineConfig()
        self.app: Optional[FaceAnalysis] = None
        self.provider_used: str = "unknown"
        self._initialize()

    def _initialize(self):
        """Initialize face analysis app with provider detection.

        Attempts to start with GPU+CPU providers.  If CUDA fails during
        initialization we catch the exception, log a warning and retry
        with CPU only so that the application continues running.
        """
        logger.info("Initializing FaceEngine...")

        if rt is None:
            raise RuntimeError("onnxruntime is required for FaceEngine but not installed")
        if FaceAnalysis is None:
            raise RuntimeError("insightface is required for FaceEngine but not installed")
        available_providers = rt.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        # Decide provider list based on configuration and availability.
        if not self.config.use_gpu:
            # user explicitly disabled GPU (e.g. CPU-only mode)
            providers = ['CPUExecutionProvider']
            self.provider_used = 'CPUExecutionProvider'
        else:
            # attempt GPU+CPU if provider exists, else fall back to CPU only
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.provider_used = 'CUDAExecutionProvider'
            else:
                providers = ['CPUExecutionProvider']
                self.provider_used = 'CPUExecutionProvider'

        # initialize FaceAnalysis with chosen providers
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            # specify ctx_id only if GPU enabled else -1
            ctx = 0 if (self.config.use_gpu and self.provider_used == 'CUDAExecutionProvider') else -1
            self.app.prepare(
                ctx_id=ctx,
                det_size=(640, 640),
                det_thresh=self.config.detection_threshold,
            )
        except Exception as e:
            logger.warning("FaceAnalysis init/prepare failed, retrying CPU only", exc_info=True)
            # guaranteed fallback to CPU
            providers = ['CPUExecutionProvider']
            self.provider_used = 'CPUExecutionProvider'
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.app.prepare(
                ctx_id=-1,
                det_size=(640, 640),
                det_thresh=self.config.detection_threshold,
            )

        logger.info(f"✓ FaceEngine initialized with {self.provider_used}")

    def _resize_image(self, image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        """Resize image if too large."""
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect faces in image."""
        if self.app is None:
            raise RuntimeError("FaceEngine not initialized")

        try:
            image = self._resize_image(image)
            faces = self.app.get(image, max_num=self.config.max_faces_per_image)
            logger.info(f"Detected {len(faces)} face(s)")
            return faces
        except Exception as e:
            logger.error(f"Face detection failed: {e}", exc_info=True)
            raise

    def extract_embedding(self, image: np.ndarray, face: dict = None) -> Optional[np.ndarray]:
        """Extract embedding."""
        if self.app is None:
            raise RuntimeError("FaceEngine not initialized")

        try:
            if face is None:
                faces = self.detect_faces(image)
                if not faces:
                    logger.warning("No face detected in image")
                    return None
                face = faces[0]

            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            logger.info(f"Extracted embedding shape: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}", exc_info=True)
            raise

    def process_image_file(self, image_data: bytes) -> Tuple[np.ndarray, List[dict]]:
        """Load image from bytes and detect faces."""
        try:
            image_pil = Image.open(BytesIO(image_data)).convert("RGB")
            image_array = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            faces = self.detect_faces(image_bgr)

            return image_bgr, faces

        except Exception as e:
            logger.error(f"Image processing failed: {e}", exc_info=True)
            raise

    def get_diagnostics(self) -> dict:
        """Return engine diagnostics."""
        try:
            available_providers = rt.get_available_providers()
            return {
                'initialized': self.app is not None,
                'provider_used': self.provider_used,
                'available_providers': available_providers,
                'embedding_dim': self.config.embedding_dim,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'initialized': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global face engine instance
face_engine: Optional[FaceEngine] = None


def init_face_engine(config: FaceEngineConfig = None) -> FaceEngine:
    global face_engine
    face_engine = FaceEngine(config)
    return face_engine


def get_face_engine() -> FaceEngine:
    if face_engine is None:
        raise RuntimeError("FaceEngine not initialized. Call init_face_engine() first.")
    return face_engine