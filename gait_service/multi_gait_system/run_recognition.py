#!/usr/bin/env python3
"""
Production-Ready Video Gait Recognition Runner
===============================================

Processes a video file frame-by-frame using the GaitRecognitionPipeline:
- Multi-person detection and tracking
- Real-time silhouette extraction
- Gait embedding computation
- Database matching and recognition
- Frame-by-frame visualization

Usage:
    python -m multi_gait_system.run_recognition
    python -m multi_gait_system.run_recognition --video path/to/video.mp4
    python -m multi_gait_system.run_recognition --checkpoint path/to/model.pt

No OpenGait framework dependencies. Pure PyTorch + YOLO + SQLite.
"""

import argparse
import sys
import time
import cv2
from pathlib import Path
from typing import Optional

from multi_gait_system.pipeline import GaitRecognitionPipeline


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run gait recognition on a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw_videos/test/Hitanshu_dense_bg.mp4",
        help="Path to video file (default: data/raw_videos/test/Hitanshu_dense_bg.mp4)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt",
        help="Path to gait model checkpoint (default: output/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt)"
    )
    
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model path (default: yolov8n-seg.pt)"
    )
    
    parser.add_argument(
        "--database",
        type=str,
        default="multi_gait_system/database/gait.db",
        help="SQLite database path (default: multi_gait_system/database/gait.db)"
    )
    
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=30,
        help="Number of frames per gait sequence (default: 30)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Recognition cosine similarity threshold (default: 0.75)"
    )
    
    parser.add_argument(
        "--smoothing-frames",
        type=int,
        default=5,
        help="Consecutive frames above threshold required to assign identity (default: 5)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Compute device (default: cuda:0)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable frame visualization"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save annotated video to path (optional)"
    )
    
    return parser.parse_args()


def validate_paths(args) -> None:
    """Validate that required files exist"""
    video_path = args.video
    checkpoint_path = Path(args.checkpoint)
    
    # Allow HTTP/HTTPS/RTSP streams; only check file existence for local paths
    if not (video_path.startswith("http://") or video_path.startswith("https://") or video_path.startswith("rtsp://")):
        if not Path(video_path).exists():
            print(f"❌ Error: Video not found: {video_path}")
            sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)


def run_recognition(args) -> None:
    """Main recognition pipeline"""
    
    print("="*70)
    print("GAIT RECOGNITION - VIDEO PROCESSING")
    print("="*70)
    print(f"\n📹 Video:       {args.video}")
    print(f"🤖 Checkpoint:  {args.checkpoint}")
    print(f"💾 Database:    {args.database}")
    print(f"📊 Buffer Size: {args.buffer_size} frames")
    print(f"📏 Threshold:   {args.threshold}")
    print(f"🔄 Smoothing:   {args.smoothing_frames} frames")
    print(f"🖥️  Device:      {args.device}")
    
    # Initialize pipeline
    print("\n" + "="*70)
    pipeline = GaitRecognitionPipeline(
        gait_checkpoint=args.checkpoint,
        yolo_model=args.yolo_model,
        database_path=args.database,
        buffer_size=args.buffer_size,
        min_frames_for_recognition=15,
        recognition_update_interval=3,
        recognition_threshold=args.threshold,
        device=args.device
    )
    
    # Open video source (MediaMTX RTSP proxy)
    print("\n" + "="*70)
    cap = cv2.VideoCapture(args.video)
    # reduce OpenCV internal buffering to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {args.video}")
        sys.exit(1)
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video loaded")
    print(f"   - Frames: {total_frames}")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - Resolution: {width}x{height}")
    
    # Setup video writer if output specified
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.output,
            fourcc,
            fps,
            (width, height)
        )
        print(f"   - Output: {args.output}")
    
    # Process frames
    print("\n" + "="*70)
    print("PROCESSING VIDEO")
    print("="*70)
    if not args.no_display:
        print("Press 'q' to quit\n")
    
    frame_count = 0
    recognized_count = 0
    
    # FPS limiter parameters
    target_fps = 60
    frame_duration = 1.0 / target_fps

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame
        annotated_frame, results = pipeline.process_frame(frame)

        # Count recognitions
        for track_id, result in results.items():
            if result['status'] == "KNOWN":
                recognized_count += 1

        # Display
        if not args.no_display:
            cv2.imshow("Gait Recognition - Press 'q' to quit", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  User interrupted")
                break

        # Save to output video
        if video_writer:
            video_writer.write(annotated_frame)

        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Recognized: {recognized_count}")

        # FPS limiting sleep
        elapsed = time.time() - start_time
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    
    # Results summary
    print("\n" + "="*70)
    print("RECOGNITION COMPLETE")
    print("="*70)
    print(f"✅ Processed {frame_count} frames")
    print(f"✅ Recognized {recognized_count} person(s)")
    if args.output:
        print(f"✅ Saved annotated video: {args.output}")
    print("="*70 + "\n")


def main():
    """Entry point"""
    args = parse_arguments()
    
    # Validate
    validate_paths(args)
    
    # Run
    try:
        run_recognition(args)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
