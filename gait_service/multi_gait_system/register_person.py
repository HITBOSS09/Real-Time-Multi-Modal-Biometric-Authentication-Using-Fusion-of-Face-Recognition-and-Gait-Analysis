import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

from multi_gait_system.pipeline import GaitRecognitionPipeline


def register_person_multi_clip(pipeline, person_name, video_paths):
    """
    Register person from multiple video clips.

    Extract embedding from each clip using YOLOv8-seg masks.
    Average embeddings and L2 normalize before saving.
    """

    print(f"\n{'='*60}")
    print(f"Registering: {person_name}")
    print(f"Videos: {len(video_paths)}")
    print(f"{'='*60}\n")

    embeddings = []

    for idx, video_path in enumerate(video_paths, 1):
        print(f"\n[{idx}/{len(video_paths)}] Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("  ❌ Failed to open video")
            continue

        frames = []
        buffer_size = pipeline.track_manager.buffer_size

        print(f"  Extracting {buffer_size} frames...")

        while len(frames) < buffer_size:
            ret, frame = cap.read()
            if not ret:
                break

            # --- YOLO SEGMENTATION ---
            results = pipeline.yolo(frame, classes=[0], verbose=False)

            if results[0].boxes is None or results[0].masks is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            masks = results[0].masks.data

            if len(boxes) == 0:
                continue

            # Select largest person
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = areas.argmax()

            x1, y1, x2, y2 = boxes[largest_idx].astype(int)

            mask = masks[largest_idx].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            mask_crop = mask[y1:y2, x1:x2]
            if mask_crop.size == 0:
                continue

            # --- CLEAN MASK ---
            kernel = np.ones((3, 3), np.uint8)
            mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Remove small contours
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) >= 200]
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            cleaned = np.zeros_like(mask_crop, dtype=np.uint8)
            cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)

            silhouette = pipeline.silhouette_extractor.extract(cleaned)
            if silhouette is None:
                continue

            frames.append(silhouette)

            if len(frames) % 10 == 0 or len(frames) == buffer_size:
                print(f"    Frames: {len(frames)}/{buffer_size}", end="\r")

        cap.release()

        if len(frames) < buffer_size:
            print(f"\n  ⚠️ Only extracted {len(frames)}/{buffer_size}, skipping clip")
            continue

        print(f"\n  Extracting embedding...")

        buffer = np.array(frames, dtype=np.float32)  # [T, 64, 44]

        buffer_tensor = (
            torch.from_numpy(buffer)
            .unsqueeze(0)  # [1, T, 64, 44]
            .unsqueeze(2)  # [1, T, 1, 64, 44]
        )

        embedding = pipeline.gait_model.extract(buffer_tensor)[0]
        embedding_np = embedding.cpu().numpy()

        print(f"  ✅ Embedding norm: {np.linalg.norm(embedding_np):.6f}")

        embeddings.append(embedding_np)

    if not embeddings:
        print("\n❌ No valid embeddings extracted.")
        return False

    print(f"\nAveraging {len(embeddings)} embedding(s)...")

    avg_embedding = np.mean(embeddings, axis=0)

    norm = np.linalg.norm(avg_embedding)
    normalized_embedding = avg_embedding / max(norm, 1e-12)

    print(f"Final embedding norm: {np.linalg.norm(normalized_embedding):.6f}")

    print("\nSaving to database...")
    pipeline.database.register_identity(person_name, normalized_embedding)

    print(f"\n{'='*60}")
    print(f"✅ {person_name} registered successfully!")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Register person from one or multiple gait videos"
    )

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str,
                        default="output/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt")
    parser.add_argument("--db", type=str,
                        default="multi_gait_system/database/gait.db")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    video_paths = []
    for video_path in args.videos:
        p = Path(video_path)
        if p.exists():
            video_paths.append(str(p.resolve()))
        else:
            print(f"❌ Not found: {video_path}")

    if not video_paths:
        print("❌ No valid videos.")
        return

    print("Initializing pipeline...")

    pipeline = GaitRecognitionPipeline(
        gait_checkpoint=args.checkpoint,
        database_path=args.db,
        buffer_size=60,
        device=args.device
    )

    success = register_person_multi_clip(
        pipeline,
        args.name,
        video_paths
    )

    if success:
        print("\nRegistered identities:")
        identities = pipeline.database.get_all_identities()
        for i, (name, templates) in enumerate(identities, 1):
            print(f"  {i}. {name} ({len(templates)} template(s))")


if __name__ == "__main__":
    main()
