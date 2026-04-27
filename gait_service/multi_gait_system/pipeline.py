"""
Production Pipeline for Real-Time Multi-Person Gait Recognition
================================================================

Complete pipeline:
Camera → YOLO Detection → ByteTrack → Silhouette Buffer → Gait Embedding → DB Match → Logging

Features:
- Multi-person tracking
- Per-track gait sequence buffer
- Automatic recognition when buffer full
- Event logging with timestamps
- No OpenGait dependencies
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ultralytics import YOLO

from multi_gait_system.models.gait_embedding_model import GaitEmbeddingModel
from multi_gait_system.database.database import GaitDatabase


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Person detection from YOLO"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    track_id: Optional[int] = None
    mask_index: Optional[int] = None


@dataclass
class Track:
    """Person track with gait buffer"""
    track_id: int
    buffer: deque  # Silhouette frames
    bbox_centers: deque  # Bounding box centers for motion tracking
    frame_count: int = 0  # total frames seen for this track
    last_recognition: Optional[float] = None
    last_recognition_frame: Optional[int] = None
    last_identity: Optional[str] = None
    last_similarity: Optional[float] = None
    start_time: Optional[str] = None  # Track start time (ISO format)
    
    def __post_init__(self):
        """Initialize start_time if not provided"""
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()


# ═══════════════════════════════════════════════════════════════════
# SILHOUETTE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

class SilhouetteExtractor:
    """Extract silhouettes from binary masks."""

    def __init__(self):
        self.target_height = 64
        self.target_width = 44

    def extract(self, mask_or_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            mask_or_crop: Either a 2D binary mask (uint8 0/255) or a color/gray
                image crop. Mask input is preferred (full-frame MOG2 should be
                applied in the pipeline and the bbox-intersected mask passed here).

        Returns:
            silhouette: [64, 44] float32 in [0, 1], or None if extraction fails
        """
        if mask_or_crop is None or mask_or_crop.size == 0:
            return None

        try:
            if mask_or_crop.ndim != 2:
                return None

            binary = mask_or_crop.copy()
            if binary.max() <= 1:
                binary = (binary * 255).astype(np.uint8)
            else:
                binary = binary.astype(np.uint8)

            # Find contours and extract largest (person)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            if contour_area <= 200:
                return None

            x, y, w, h = cv2.boundingRect(largest_contour)
            if h < 10:
                return None

            crop = binary[y:y+h, x:x+w]

            # Resize maintaining aspect ratio (target height = 64)
            scale = self.target_height / h
            new_width = int(w * scale)
            new_height = self.target_height

            if new_width > self.target_width:
                new_width = self.target_width

            resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            # --- Centroid-based horizontal alignment ---
            # Compute center of mass of foreground pixels in the resized crop
            fg_coords = np.argwhere(resized > 0)  # (row, col)
            if fg_coords.size > 0:
                centroid_x = fg_coords[:, 1].mean()
                # Desired centroid position: center of the target canvas
                desired_x = self.target_width / 2.0
                # Horizontal shift needed to align centroid with canvas center
                shift = int(round(desired_x - centroid_x))
            else:
                shift = (self.target_width - new_width) // 2

            canvas = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
            # Compute placement with centroid-based shift
            x_offset = max(0, min(shift, self.target_width - new_width))
            y_offset = self.target_height - new_height
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

            silhouette = canvas.astype(np.float32) / 255.0

            assert silhouette.shape == (64, 44)
            assert silhouette.dtype == np.float32

            return silhouette

        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════
# TRACK MANAGER
# ═══════════════════════════════════════════════════════════════════

class TrackManager:
    """Manage per-person tracks and silhouette buffers"""

    def __init__(self, buffer_size: int = 60, min_frames_for_recognition: int = 20,
                 recognition_update_interval: int = 5):
        """
        Args:
            buffer_size: Maximum number of frames kept per gait sequence
            min_frames_for_recognition: Minimum frames required before first recognition
            recognition_update_interval: Minimum frames between recognition updates
        """
        self.buffer_size = buffer_size
        self.min_frames_for_recognition = min_frames_for_recognition
        self.recognition_update_interval = recognition_update_interval
        self.tracks: Dict[int, Track] = {}
    
    def update(self, track_id: int, silhouette: np.ndarray, bbox_center: Tuple[int, int] = None):
        """Add new silhouette frame to track buffer"""
        if track_id not in self.tracks:
            self.tracks[track_id] = Track(
                track_id=track_id,
                buffer=deque(maxlen=self.buffer_size),
                bbox_centers=deque(maxlen=self.buffer_size)
            )
        
        self.tracks[track_id].buffer.append(silhouette)
        if bbox_center is not None:
            self.tracks[track_id].bbox_centers.append(bbox_center)

        # Increment per-track frame counter
        self.tracks[track_id].frame_count += 1
    
    def get_ready_tracks(self) -> List[int]:
        """Get track IDs eligible for recognition based on buffer length and update interval.

        Uses frame-counts to schedule periodic recognition once the minimum frame
        threshold has been met. This reduces latency by allowing early recognition
        when `len(buffer) >= min_frames_for_recognition`.
        """
        ready = []
        for track_id, track in self.tracks.items():
            # Require minimum frames before first recognition
            if len(track.buffer) < self.min_frames_for_recognition:
                continue

            # If never recognized, it's ready
            if track.last_recognition_frame is None:
                ready.append(track_id)
                continue

            # Otherwise require sufficient frames since last recognition
            if (track.frame_count - (track.last_recognition_frame or 0)) >= self.recognition_update_interval:
                ready.append(track_id)

        return ready
    
    def get_buffer(self, track_id: int) -> Optional[np.ndarray]:
        """Get silhouette buffer for track"""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        # Return buffer once we have at least one frame; caller enforces min_frames
        if len(track.buffer) == 0:
            return None
        
        # Convert buffer to numpy array
        buffer = np.array(list(track.buffer))  # [T, H, W]
        return buffer
    
    def update_recognition(self, track_id: int, identity: str, similarity: float):
        """Update track with recognition result"""
        import time
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.last_recognition = time.time()
            track.last_recognition_frame = track.frame_count
            track.last_identity = identity
            track.last_similarity = similarity
    
    def remove_track(self, track_id: int):
        """Remove track (e.g., when person leaves frame)"""
        if track_id in self.tracks:
            del self.tracks[track_id]
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove tracks not in active list"""
        to_remove = [tid for tid in list(self.tracks.keys()) if tid not in active_track_ids]
        for tid in to_remove:
            self.remove_track(tid)

    def get_motion_displacement(self, track_id: int) -> float:
        """
        Calculate total horizontal displacement of bounding box center.
        
        Returns:
            displacement: Total horizontal pixels moved, or -1 if insufficient data
        """
        if track_id not in self.tracks:
            return -1.0

        centers = self.tracks[track_id].bbox_centers
        if len(centers) < 2:
            return -1.0

        # Calculate total horizontal displacement from first to last center
        first_x = centers[0][0]
        last_x = centers[-1][0]
        displacement = abs(last_x - first_x)

        return float(displacement)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

class GaitRecognitionPipeline:
    """
    End-to-end real-time multi-person gait recognition pipeline
    
    Usage:
        pipeline = GaitRecognitionPipeline(
            gait_checkpoint="path/to/checkpoint.pt",
            yolo_model="yolov8n-seg.pt",
            database_path="gait.db"
        )
        
        # Process video
        cap = cv2.VideoCapture("video.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, results = pipeline.process_frame(frame)
            cv2.imshow("Gait Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    """
    
    def __init__(
        self,
        gait_checkpoint: str,
        yolo_model: str = "yolov8n-seg.pt",
        database_path: str = "multi_gait_system/database/gait.db",
        buffer_size: int = 40,
        min_frames_for_recognition: int = 15,
        recognition_update_interval: int = 3,
        recognition_threshold: float = 0.80,
        similarity_margin: float = 0.04,
        motion_energy_threshold: float = 0.008,
        gap_threshold: float = 0.08,
        ema_alpha: float = 0.4,
        embedding_ema_alpha: float = 0.5,
        stability_required: int = 3,
        unlock_threshold: float = 0.60,
        border_margin: int = 20,
        bbox_height_deviation: float = 0.40,
        top_k_scores: int = 3,
        device: str = "cuda:0"
    ):
        """
        Args:
            gait_checkpoint: Path to GaitBase_DA-60000.pt
            yolo_model: YOLO model path
            database_path: SQLite database path
            buffer_size: Frames per gait sequence
            min_frames_for_recognition: Min frames before first recognition
            recognition_update_interval: Min frames between recognition updates
            recognition_threshold: Cosine similarity threshold for matching
            motion_energy_threshold: Min motion energy for valid gait
            gap_threshold: Min gap between best and second-best similarity
            ema_alpha: EMA smoothing factor for similarity scores
            embedding_ema_alpha: EMA smoothing factor for embeddings
            stability_required: Consecutive stable updates to lock identity
            unlock_threshold: Similarity below which a locked identity is released
            border_margin: Pixel margin from frame edge for full-body visibility gate
            bbox_height_deviation: Max relative deviation in bbox height between frames
            top_k_scores: Number of top scores used for track-level aggregation
            device: 'cuda:0' or 'cpu'
        """
        print("Initializing Gait Recognition Pipeline...")
        
        # Load models
        print(f"Loading YOLO model: {yolo_model}")
        # Use the provided YOLO segmentation model path (segmentation-capable and lightweight by default)
        self.yolo = YOLO(yolo_model)
        
        print(f"Loading gait model: {Path(gait_checkpoint).name}")
        self.gait_model = GaitEmbeddingModel(gait_checkpoint, device=device)
        
        # Database
        print(f"Connecting to database: {database_path}")
        self.database = GaitDatabase(database_path)
        
        # Components
        self.silhouette_extractor = SilhouetteExtractor()
        self.track_manager = TrackManager(
            buffer_size=buffer_size,
            min_frames_for_recognition=min_frames_for_recognition,
            recognition_update_interval=recognition_update_interval
        )
        
        # Recognition parameters
        self.recognition_threshold = recognition_threshold
        self.motion_energy_threshold = motion_energy_threshold
        self.gap_threshold = gap_threshold
        self.ema_alpha = ema_alpha
        self.embedding_ema_alpha = embedding_ema_alpha
        self.stability_required = stability_required
        self.unlock_threshold = unlock_threshold
        # similarity margin used to guard against close scores
        self.similarity_margin = similarity_margin

        # Frame quality gate parameters
        self.border_margin = border_margin
        self.bbox_height_deviation = bbox_height_deviation

        # Track-level score aggregation
        self.top_k_scores = top_k_scores

        # Per-track state: smoothed similarity (EMA), embedding EMA, stability counters, and lock state
        self._smoothed_similarity: Dict[int, float] = {}
        # Keep embedding EMA as torch tensors on device to avoid unnecessary CPU transfers
        self._embedding_ema: Dict[int, torch.Tensor] = {}
        # embedding buffer per-track for temporal smoothing (last 5 embeddings)
        self.embedding_buffer: Dict[int, deque] = {}
        self._stability_counters: Dict[int, int] = {}
        self._unlock_counters: Dict[int, int] = {}
        self._locked_identities: Dict[int, Optional[str]] = {}
        # store similarity value when an identity is locked to prevent flip
        self._locked_similarity: Dict[int, float] = {}

        # instantaneous motion: keep previous silhouette and last motion value
        self._prev_silhouette: Dict[int, np.ndarray] = {}
        self._last_motion: Dict[int, float] = {}

        # Frame quality gate: per-track bounding box height history
        self._bbox_height_history: Dict[int, deque] = {}

        # Track-level score aggregation: per-track similarity history
        self._track_score_history: Dict[int, deque] = {}
        # Track-level voting: store identity predictions per track for majority voting
        self.track_votes: Dict[int, List[str]] = defaultdict(list)
        # Debug flags (initialize here to avoid missing-attribute errors)
        self.debug_save_silhouettes: bool = False
        # Performance tuning
        self.max_tracks = 5
        self.device = device
        # Identity gallery caches (centroids and adaptive thresholds)
        self.identity_centroids: Dict[str, np.ndarray] = {}
        self.identity_thresholds: Dict[str, float] = {}
        # Populate initial gallery cache
        self._refresh_gallery()

    def _refresh_gallery(self) -> None:
        """Compute identity centroids and adaptive thresholds from DB templates."""
        identities = self.database.get_all_identities()
        self.identity_centroids = {}
        self.identity_thresholds = {}
        for name, templates in identities:
            try:
                centroid_np = np.mean(np.stack(templates, axis=0), axis=0)
            except Exception:
                centroid_np = templates[0]
            self.identity_centroids[name] = centroid_np

            sims = []
            if len(templates) > 1:
                t_tensors = [torch.from_numpy(t.astype(np.float32)).to(self.device) for t in templates]
                for i in range(len(t_tensors)):
                    for j in range(i + 1, len(t_tensors)):
                        a = t_tensors[i] / (t_tensors[i].norm(p=2) + 1e-12)
                        b = t_tensors[j] / (t_tensors[j].norm(p=2) + 1e-12)
                        sims.append(float(torch.dot(a, b).item()))
            if len(sims) > 0:
                mean_sim = float(np.mean(sims))
                std_sim = float(np.std(sims))
                self.identity_thresholds[name] = max(0.5, mean_sim - 2 * std_sim)
            else:
                self.identity_thresholds[name] = 0.6

    def process_frame(self, frame: np.ndarray):
        """Process a single video frame: detect, extract silhouettes, recognize, and annotate."""
        import time
        start = time.time()

        # Keep original for annotation/display
        orig_frame = frame.copy()

        # 1a. Frame downscaling to reduce GPU load
        try:
            frame = cv2.resize(frame, (960, 540))
        except Exception:
            # if resize fails, continue with original
            pass

        # 1. Detect and track persons (use segmentation-capable YOLO)
        results = self.yolo.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # If the segmentation model didn't produce masks, skip
        if results[0].masks is None:
            return frame, {}

        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Human geometry filter: aspect ratio
                width = x2 - x1
                height = y2 - y1
                if width > 0:
                    ratio = height / width
                    if ratio < 1.3 or ratio > 4.5:
                        continue
                
                # Bounding box size filter: minimum height
                if height < 120:
                    continue
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    track_id=int(track_id),
                    mask_index=int(i)
                ))

        # 2a. Cap maximum active tracks to reduce work — keep largest boxes
        if len(detections) > self.max_tracks:
            detections = sorted(
                detections,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                reverse=True
            )[:self.max_tracks]

        # 2. Extract silhouettes and update buffers using YOLO segmentation masks
        h_frame, w_frame = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # PERFORMANCE: skip very small detections early
            if (y2 - y1) < 40:
                continue

            # --- Frame Quality Gate: full-body visibility ---
            # Reject detections where the bounding box is too close to the frame edge
            margin = self.border_margin
            if x1 < margin or y1 < margin or x2 > w_frame - margin or y2 > h_frame - margin:
                continue

            # --- Frame Quality Gate: bounding box height stability ---
            bbox_height = y2 - y1
            if det.track_id not in self._bbox_height_history:
                self._bbox_height_history[det.track_id] = deque(maxlen=30)
            height_hist = self._bbox_height_history[det.track_id]
            if len(height_hist) >= 3:
                mean_height = np.mean(height_hist)
                if abs(bbox_height - mean_height) / max(mean_height, 1) > self.bbox_height_deviation:
                    # Height fluctuation too large — skip this frame
                    continue
            height_hist.append(bbox_height)

            # Compute bounding box center for motion tracking
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Retrieve corresponding mask from YOLO results
            try:
                mask = results[0].masks.data[det.mask_index]
                mask = mask.cpu().numpy()
            except Exception:
                continue

            # Convert to binary (no interpolation artifacts)
            mask = (mask > 0.5).astype(np.uint8) * 255

            # Ensure mask resolution matches frame
            h_mask, w_mask = mask.shape

            if (h_mask != h_frame) or (w_mask != w_frame):
                mask = cv2.resize(
                    mask,
                    (w_frame, h_frame),
                    interpolation=cv2.INTER_NEAREST
                )

            # Clip bbox safely to frame bounds
            x1_clipped = max(0, min(x1, w_frame - 1))
            x2_clipped = max(0, min(x2, w_frame))
            y1_clipped = max(0, min(y1, h_frame - 1))
            y2_clipped = max(0, min(y2, h_frame))

            if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
                continue

            # Crop mask using corrected coordinates
            mask_crop = mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

            if mask_crop.size == 0:
                continue

            # Clean mask with small morphological ops
            kernel = np.ones((3, 3), np.uint8)
            mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Keep only contours >= 200 area; select largest
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) >= 200]
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            largest_mask = np.zeros_like(mask_crop, dtype=np.uint8)
            cv2.drawContours(largest_mask, [largest], -1, 255, thickness=cv2.FILLED)

            silhouette = self.silhouette_extractor.extract(largest_mask)
            if silhouette is None:
                continue

            # --- Frame Quality Gate: silhouette size constraint ---
            pixel_ratio = silhouette.mean()
            if pixel_ratio < 0.10 or pixel_ratio > 0.70:
                continue

            # DEBUG: Verify silhouette is binary [0, 1] and correct shape (log only first frame per track)
            track = self.track_manager.tracks.get(det.track_id)
            if track is None or len(track.buffer) == 0:
                print(f"[DEBUG Track {det.track_id}] Silhouette: shape={silhouette.shape}, "
                      f"dtype={silhouette.dtype}, min={silhouette.min():.3f}, max={silhouette.max():.3f}, "
                      f"mean={silhouette.mean():.3f}")
            self.track_manager.update(det.track_id, silhouette, bbox_center=bbox_center)
            # update instantaneous motion for this track
            prev = self._prev_silhouette.get(det.track_id)
            if prev is not None:
                motion_val = float(np.mean(np.abs(silhouette - prev)))
                self._last_motion[det.track_id] = motion_val
            self._prev_silhouette[det.track_id] = silhouette.copy()
            # Silhouette quality filter: skip poor-quality silhouettes
            sil_mean = float(silhouette.mean())
            if sil_mean < 0.20:
                print(f"[DEBUG Track {det.track_id}] Low silhouette quality ({sil_mean:.3f}) - SKIPPED")
                continue
            # Optionally save every 10th silhouette per track
            try:
                if self.debug_save_silhouettes:
                    track = self.track_manager.tracks.get(det.track_id)
                    if track is not None and track.frame_count % 10 == 0:
                        # Get the final silhouette stored in buffer
                        img = track.buffer[-1]
                        # Convert to uint8 0-255
                        img_u8 = (img * 255.0).astype(np.uint8)
                        fname = self.debug_silhouettes_dir / f"track_{det.track_id}_frame_{track.frame_count}.png"
                        cv2.imwrite(str(fname), img_u8)
            except Exception as e:
                print(f"Failed to save debug silhouette for track {det.track_id}: {e}")
        
        # 3. Cleanup old tracks and their similarity history
        active_ids = [det.track_id for det in detections]
        
        # Log final results for tracks being removed
        self._log_dead_tracks(active_ids)
        
        self.track_manager.cleanup_old_tracks(active_ids)
        self._clean_similarity_history()
        
        # 4. Run recognition on ready tracks (EMA similarity, adaptive margin, embedding EMA)
        recognition_results = {}
        ready_tracks = self.track_manager.get_ready_tracks()

        for track_id in ready_tracks:
            buffer = self.track_manager.get_buffer(track_id)
            if buffer is None:
                continue

            # Check if person has sufficient gait motion energy (motion filter)
            motion_energy = self._compute_motion_energy(buffer)
            if motion_energy < self.motion_energy_threshold:
                # Skip embedding extraction for insufficient motion
                print(f"[DEBUG Track {track_id}] Motion filter: energy={motion_energy:.6f} "
                      f"(threshold={self.motion_energy_threshold:.6f}) - SKIPPED")
                # Set to Unknown and reset smoothing/locking state
                self._smoothed_similarity.pop(track_id, None)
                self._embedding_ema.pop(track_id, None)
                self._stability_counters.pop(track_id, None)
                self._unlock_counters.pop(track_id, None)
                self._locked_identities.pop(track_id, None)
                self.track_manager.update_recognition(track_id, "Unknown", 0.0)
                continue

            # instantaneous frame-to-frame motion filter
            last_motion = self._last_motion.get(track_id)
            if last_motion is not None and last_motion < 0.02:
                print(f"[DEBUG Track {track_id}] Instant motion low ({last_motion:.4f}) - SKIPPED")
                self.track_manager.update_recognition(track_id, "No Motion", 0.0)
                continue

            # Prepare input: [1, T, 1, H, W] and move to model device
            buffer_tensor = torch.from_numpy(buffer).unsqueeze(0).unsqueeze(2).to(self.device)
            assert buffer_tensor.shape[2] == 1, f"Channel dim must be 1, got {buffer_tensor.shape[2]}"

            # Extract embedding (torch tensor on device)
            embedding = self.gait_model.extract(buffer_tensor)[0]

            # Update per-track embedding buffer for temporal smoothing (keep last 5)
            if track_id not in self.embedding_buffer:
                self.embedding_buffer[track_id] = deque(maxlen=5)
            # store detached float tensor on device
            self.embedding_buffer[track_id].append(embedding.detach().float())

            # compute smoothed embedding as mean of buffer (on device)
            buf = list(self.embedding_buffer[track_id])
            if len(buf) == 0:
                emb_smoothed = embedding.detach().float()
            else:
                emb_smoothed = torch.mean(torch.stack(buf, dim=0), dim=0)

            # keep EMA copy for historical compatibility
            prev_emb = self._embedding_ema.get(track_id)
            if prev_emb is None:
                self._embedding_ema[track_id] = emb_smoothed.detach()
            else:
                self._embedding_ema[track_id] = ((self.embedding_ema_alpha * emb_smoothed) + ((1.0 - self.embedding_ema_alpha) * prev_emb)).detach()

            # Use the smoothed embedding for matching
            emb_ema = emb_smoothed

            # Compute similarities against all identities in DB (multi-template gallery)
            identities = self.database.get_all_identities()  # list of (name, [emb1, emb2, ...])
            if not identities:
                # No registered identities
                final_identity = "Unknown"
                smoothed = self._smoothed_similarity.get(track_id, 0.0)
                self.database.log_entry(final_identity, float(smoothed), "UNKNOWN")
                self.track_manager.update_recognition(track_id, final_identity, float(smoothed))
                recognition_results[track_id] = {'identity': final_identity, 'similarity': float(smoothed), 'status': 'UNKNOWN'}
                continue

            # Compute cosine similarities using hybrid strategy (centroid + templates)
            sims = []
            # normalize query embedding (torch)
            q = emb_ema / (emb_ema.norm(p=2) + 1e-12)
            for name, templates in identities:
                # Use cached centroid if available (computed in _refresh_gallery)
                centroid_np = self.identity_centroids.get(name)
                centroid_sim = -1.0
                if centroid_np is not None:
                    centroid_t = torch.from_numpy(centroid_np.astype(np.float32)).to(self.device)
                    centroid_t = centroid_t / (centroid_t.norm(p=2) + 1e-12)
                    centroid_sim = float(torch.dot(q, centroid_t).item())

                # template-level best similarity
                best_template_sim = 0.0
                for t in templates:
                    t_torch = torch.from_numpy(t.astype(np.float32)).to(self.device)
                    t_torch = t_torch / (t_torch.norm(p=2) + 1e-12)
                    sim = float(torch.dot(q, t_torch).item())
                    if sim > best_template_sim:
                        best_template_sim = sim

                # hybrid: take the best of centroid and template sims
                hybrid_best = max(centroid_sim, best_template_sim)
                sims.append((name, float(hybrid_best)))

            sims.sort(key=lambda x: x[1], reverse=True)
            best_name, best_score = sims[0]
            second_best_score = sims[1][1] if len(sims) > 1 else 0.0

            # --- Track-level score aggregation ---
            if track_id not in self._track_score_history:
                self._track_score_history[track_id] = deque(maxlen=30)
            self._track_score_history[track_id].append(best_score)

            # Aggregated score: mean of top-k similarity scores for this track
            score_hist = sorted(self._track_score_history[track_id], reverse=True)
            top_k = score_hist[:self.top_k_scores]
            aggregated_score = float(np.mean(top_k))

            # EMA smoothing of similarity (uses aggregated score)
            prev_sm = self._smoothed_similarity.get(track_id, aggregated_score)
            smoothed_sim = (self.ema_alpha * aggregated_score) + ((1.0 - self.ema_alpha) * prev_sm)
            self._smoothed_similarity[track_id] = smoothed_sim

            # Decide candidate identity using threshold + margin check (guards against similar scores)
            candidate = "Unknown"
            # baseline recognition threshold
            if (best_score > self.recognition_threshold) and ((best_score - second_best_score) > self.similarity_margin):
                candidate = best_name
            # apply adaptive per-identity threshold with a small tolerance
            if candidate != "Unknown":
                id_thresh = self.identity_thresholds.get(candidate, self.recognition_threshold)
                # allow a 0.03 slack below the computed threshold
                if best_score < (id_thresh - 0.03):
                    candidate = "Unknown"

            # Stability / locking mechanism
            locked = self._locked_identities.get(track_id)
            if locked is not None:
                # currently locked to an identity
                locked_sim = self._locked_similarity.get(track_id, 0.0)
                if candidate != locked:
                    if smoothed_sim < locked_sim + 0.05:
                        # similarity not sufficiently higher -> keep existing lock
                        candidate = locked
                    else:
                        # allow switch: update lock immediately
                        self._locked_identities[track_id] = candidate
                        self._locked_similarity[track_id] = smoothed_sim
                # unlocking logic (unchanged)
                if smoothed_sim < self.unlock_threshold:
                    # count unlock confirmations
                    self._unlock_counters[track_id] = self._unlock_counters.get(track_id, 0) + 1
                    if self._unlock_counters[track_id] >= 2:
                        print(f"[DEBUG Track {track_id}] Unlocking identity {locked} due to low sim {smoothed_sim:.3f}")
                        self._locked_identities[track_id] = None
                        self._stability_counters[track_id] = 0
                        self._unlock_counters[track_id] = 0
                else:
                    # maintain lock counter
                    candidate = self._locked_identities.get(track_id, candidate)
                    self._unlock_counters[track_id] = 0
            else:
                # not locked: update stability counter
                prev_counter = self._stability_counters.get(track_id, 0)
                if candidate != "Unknown" and candidate == self.track_manager.tracks.get(track_id).last_identity:
                    self._stability_counters[track_id] = prev_counter + 1
                elif candidate != "Unknown":
                    # new candidate different from last identity
                    self._stability_counters[track_id] = 1
                else:
                    self._stability_counters[track_id] = 0

                # Lock if stability reached
                if self._stability_counters.get(track_id, 0) >= self.stability_required:
                    print(f"[DEBUG Track {track_id}] Locking identity {candidate} after {self.stability_required} stable updates")
                    self._locked_identities[track_id] = candidate
                    # record similarity at lock time
                    self._locked_similarity[track_id] = smoothed_sim

            # --- Log frame-level prediction ---
            # Determine decision reason for this frame
            if candidate == "Unknown":
                decision = "BELOW_THRESHOLD"
            elif candidate == "No Motion":
                decision = "MOTION_FILTER"
            else:
                decision = "RECOGNIZED"
            
            # Log this frame's prediction with best_score (raw frame similarity)
            self.database.log_frame_prediction(
                track_id=track_id,
                person_name=candidate if candidate != "Unknown" else None,
                similarity=float(best_score),
                decision=decision
            )

            # Track-level voting: accumulate identity predictions
            self.track_votes[track_id].append(candidate)
            # Limit vote history to 10 most recent
            if len(self.track_votes[track_id]) > 10:
                self.track_votes[track_id].pop(0)

            # Apply voting with dominance ratio to decide final identity
            votes = list(self.track_votes[track_id])
            if votes:
                vote_counts = Counter(votes)
                most_common_identity, count = vote_counts.most_common(1)[0]
                vote_ratio = count / float(len(votes))
                top3 = vote_counts.most_common(3)
                print(f"[VOTE] Track {track_id} votes: {votes} top3={top3} ratio={vote_ratio:.2f}")
                # Require dominant vote share (>=70%) and non-Unknown to accept
                if most_common_identity is not None and most_common_identity != "Unknown" and vote_ratio >= 0.7:
                    # If previously locked, require new best_score to exceed locked similarity by margin
                    locked_sim = self._locked_similarity.get(track_id, -1.0)
                    if locked_sim > 0 and best_score <= locked_sim + 0.05:
                        final_identity = self._locked_identities.get(track_id, "Unknown")
                        print(f"[VOTE] Keeping locked identity {final_identity} (locked_sim={locked_sim:.3f} best={best_score:.3f})")
                    else:
                        final_identity = most_common_identity
                        print(f"[VOTE] Final decision (ratio={vote_ratio:.2f}): {final_identity}")
                        # reinforce lock
                        self._locked_identities[track_id] = final_identity
                        self._locked_similarity[track_id] = best_score
                else:
                    final_identity = "Unknown"
                    print(f"[VOTE] Insufficient dominance (ratio={vote_ratio:.2f}): marking as Unknown")
            else:
                final_identity = "Unknown"

            status = "KNOWN" if final_identity != "Unknown" else "UNKNOWN"

            # Log and update track
            self.database.log_entry(final_identity, float(smoothed_sim), status)
            self.track_manager.update_recognition(track_id, final_identity, float(smoothed_sim))

            recognition_results[track_id] = {
                'identity': final_identity,
                'similarity': float(smoothed_sim),
                'status': status,
                'best_score': float(best_score),
                'second_best_score': float(second_best_score)
            }
            print(f"Track {track_id}: {final_identity} (sm={smoothed_sim:.3f}, best={best_score:.3f}, sec={second_best_score:.3f})")
        
        # 5. Annotate frame
        annotated_frame = self._annotate_frame(frame.copy(), detections, recognition_results)
        
        return annotated_frame, recognition_results
    
    def _log_dead_tracks(self, active_track_ids: List[int]) -> None:
        """
        Log final recognition results for tracks that are being removed.
        
        Computes track-level identity based on voting and similarity aggregation,
        then writes to recognition_results table.
        """
        # Identify dead tracks
        dead_track_ids = [tid for tid in list(self.track_manager.tracks.keys()) 
                         if tid not in active_track_ids]
        
        if not dead_track_ids:
            return
        
        print(f"[LOGGING] Finalizing results for {len(dead_track_ids)} dead tracks: {dead_track_ids}")
        
        for track_id in dead_track_ids:
            try:
                track = self.track_manager.tracks.get(track_id)
                if track is None:
                    continue
                
                # Compute final identity using voting
                votes = list(self.track_votes.get(track_id, []))
                final_identity = "Unknown"
                
                if votes:
                    vote_counts = Counter(votes)
                    most_common_identity, count = vote_counts.most_common(1)[0]
                    vote_ratio = count / float(len(votes))
                    
                    # Accept identity if it dominates (>=70%) and is not Unknown
                    if most_common_identity != "Unknown" and vote_ratio >= 0.7:
                        final_identity = most_common_identity
                    
                    print(f"[LOGGING] Track {track_id}: votes={list(vote_counts.items())}, "
                          f"final_identity={final_identity}, ratio={vote_ratio:.2f}")
                
                # Compute average and max similarity for this identity
                avg_similarity = 0.0
                max_similarity = 0.0
                
                if final_identity != "Unknown":
                    # Get all predictions for this track and compute statistics
                    final_identity_computed, avg_sim, max_sim = self.database.compute_track_identity(
                        track_id=track_id,
                        similarity_threshold=0.5,
                        min_occurrences=1  # Use all predictions for aggregation
                    )
                    
                    # Use computed values if available
                    if final_identity_computed is not None:
                        avg_similarity = avg_sim
                        max_similarity = max_sim
                    else:
                        # Fallback: use smoothed similarity
                        avg_similarity = self._smoothed_similarity.get(track_id, 0.0)
                        max_similarity = avg_similarity
                else:
                    # For Unknown, use smoothed similarity
                    avg_similarity = self._smoothed_similarity.get(track_id, 0.0)
                    max_similarity = avg_similarity
                
                # Get track timing
                end_time = datetime.now().isoformat()
                start_time = track.start_time if track.start_time else None
                
                # Log to database
                self.database.log_track_result(
                    track_id=track_id,
                    identity=final_identity,
                    modality="gait",
                    avg_similarity=float(avg_similarity),
                    max_similarity=float(max_similarity),
                    start_time=start_time,
                    end_time=end_time
                )
                
                print(f"[LOGGING] Track {track_id} final result: identity={final_identity}, "
                      f"avg_sim={avg_similarity:.3f}, max_sim={max_similarity:.3f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to log track {track_id}: {e}")
    
    def _clean_similarity_history(self) -> None:
        """Remove per-track EMA, stability, score history, bbox height state, and votes for dead tracks."""
        active = set(self.track_manager.tracks.keys())

        for d in [self._smoothed_similarity, self._embedding_ema, self._stability_counters,
                  self._unlock_counters, self._locked_identities,
                  self._bbox_height_history, self._track_score_history, self.track_votes]:
            for track_id in list(d.keys()):
                if track_id not in active:
                    del d[track_id]
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        recognition_results: Dict
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            track_id = det.track_id
            
            # Get track info
            track = self.track_manager.tracks.get(track_id)
            if track is None:
                continue
            
            # Determine color and label (similarity shown is moving average)
            if track.last_identity is not None:
                identity = track.last_identity
                similarity = track.last_similarity  # moving average over last N
                
                if identity == "Unknown":
                    color = (0, 0, 255)  # Red
                    label = f"ID:{track_id} Unknown (avg={similarity:.2f})"
                elif identity == "No Motion":
                    color = (0, 0, 0)  # Black
                    label = f"ID:{track_id} No Motion"
                else:
                    color = (0, 255, 0)  # Green
                    label = f"ID:{track_id} {identity} (avg={similarity:.2f})"
            else:
                color = (255, 255, 0)  # Cyan
                buffer_size = len(track.buffer)
                label = f"ID:{track_id} Tracking ({buffer_size}/{self.track_manager.buffer_size})"
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )
        
        return frame
    
    def register_person(self, video_paths, person_name: str):
        """
        Register new person using one or more videos. For each provided video
        we attempt to collect silhouette frames (using relaxed filters), sample
        every 3 frames, compute an embedding from the collected frames for that
        video, and store it as a template in the multi-template gallery.

        Args:
            video_paths: single path or list of video paths to use for registration
            person_name: Identity name
        """
        # Allow a single string to be provided
        if isinstance(video_paths, str):
            video_list = [video_paths]
        else:
            video_list = list(video_paths)

        print(f"[REG] Registering: {person_name}")
        collected_templates = 0

        for video_path in video_list:
            print(f"[REG] Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample every 3 frames to avoid near-duplicates
                if frame_idx % 3 != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1

                # Use YOLO single-image inference (segmentation-capable)
                results = self.yolo(frame, classes=[0], verbose=False)
                if results[0].boxes is None or results[0].masks is None:
                    continue

                boxes = results[0].boxes.xyxy.cpu().numpy()
                masks = results[0].masks.data

                if boxes.shape[0] == 0:
                    continue

                # Choose largest box (most likely the person of interest)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = int(areas.argmax())

                x1, y1, x2, y2 = boxes[largest_idx].astype(int)

                mask = masks[largest_idx].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)

                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Safe clip
                h, w = frame.shape[:2]
                x1c = max(0, min(x1, w - 1))
                x2c = max(0, min(x2, w))
                y1c = max(0, min(y1, h - 1))
                y2c = max(0, min(y2, h))

                if x2c <= x1c or y2c <= y1c:
                    continue

                mask_crop = mask[y1c:y2c, x1c:x2c]
                if mask_crop.size == 0:
                    continue

                # Simple morphological cleanup
                kernel = np.ones((3, 3), np.uint8)
                mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, iterations=1)
                mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel, iterations=1)

                # Contour-based selection with relaxed thresholds for registration
                contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = [c for c in contours if cv2.contourArea(c) >= 150]
                if not contours:
                    continue

                largest = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(largest)
                # relaxed bbox height requirement
                if h_box < 20:
                    continue

                largest_mask = np.zeros_like(mask_crop, dtype=np.uint8)
                cv2.drawContours(largest_mask, [largest], -1, 255, thickness=cv2.FILLED)

                silhouette = self.silhouette_extractor.extract(largest_mask)
                # During registration we are permissive: accept any silhouette that
                # the extractor returns (no strict pixel-ratio or border checks)
                if silhouette is None:
                    continue

                frames.append(silhouette)

                # Stop early per-video if we've reached desired buffer_size
                if len(frames) >= self.track_manager.buffer_size:
                    break

            cap.release()

            print(f"[REG] silhouettes collected from video: {len(frames)}")

            if len(frames) == 0:
                print(f"[REG] Warning: no silhouettes found in {video_path}, skipping")
                continue

            # If fewer frames than buffer_size were collected, continue anyway
            if len(frames) < self.track_manager.buffer_size:
                print(f"[REG] Warning: only {len(frames)} frames collected from {video_path}, continuing registration")

            # Build buffer for this video and extract embedding
            print("[REG] Extracting embedding")
            buffer = np.array(frames)
            buffer_tensor = torch.from_numpy(buffer).unsqueeze(0).unsqueeze(2)
            try:
                embedding = self.gait_model.extract(buffer_tensor)[0]
            except Exception as e:
                print(f"[REG] Failed to extract embedding for {video_path}: {e}")
                continue

            # Save template to database with source video metadata
            print("[REG] Storing template in database")
            self.database.register_identity(
                person_name,
                embedding.detach().cpu().numpy(),
                source_video=video_path
            )
            collected_templates += 1

        if collected_templates == 0:
            print(f"[REG] Warning: no templates created for '{person_name}'")
        else:
            print(f"[REG] Completed registration for '{person_name}' - templates stored: {collected_templates}")
    
    def _compute_motion_energy(self, buffer: np.ndarray) -> float:
        """
        Compute gait motion energy as mean absolute frame-to-frame difference.
        
        Args:
            buffer: [T, H, W] silhouette buffer
        
        Returns:
            motion_energy: Mean absolute pixel difference between consecutive frames
        """
        if buffer.shape[0] < 2:
            return 0.0
        
        diffs = np.abs(np.diff(buffer, axis=0))
        return float(diffs.mean())


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════

def main():
    """Example: Run pipeline on video"""
    
    # Initialize pipeline
    pipeline = GaitRecognitionPipeline(
        gait_checkpoint="output/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt",
        yolo_model="yolov8n-seg.pt",
        database_path="multi_gait_system/database/gait.db",
        buffer_size=30,
        recognition_threshold=0.75,
        device="cuda:0"
    )
    
    # Process video
    video_path = "test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    print(f"\nProcessing video: {video_path}")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, results = pipeline.process_frame(frame)
        
        # Display
        cv2.imshow("Gait Recognition", annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Finished processing {frame_count} frames")


if __name__ == "__main__":
    main()
