import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Camera Mode ("webcam" or "rtsp") --------
camera_mode = "webcam"

# -------- Camera Streams (Direct RTSP) --------

FACE_CAM_RTSP = "rtsp://admin:Admi@321@192.168.1.201:554/cam/realmonitor?channel=1&subtype=0"
GAIT_CAM_RTSP = "rtsp://admin:Admi@321@192.168.1.200:554/cam/realmonitor?channel=1&subtype=0"


def main():
    # -------- Face Service --------
    if camera_mode == "webcam":
        face_cmd = [
            sys.executable,
            "-m",
            "cctv_engine.video_processor",
            "--source",
            "webcam",
        ]
    else:
        face_cmd = [
            sys.executable,
            "-m",
            "cctv_engine.video_processor",
            "--source",
            "rtsp",
            "--rtsp",
            FACE_CAM_RTSP,
        ]
    # -------- Gait Service --------
    gait_cmd = [
        sys.executable,
        "-m",
        "multi_gait_system.run_recognition",
        "--video",
        GAIT_CAM_RTSP,
        "--checkpoint",
        "experiments/supervised/checkpoints/fold_1_seed_42.pt",
        "--device",
        "cuda",
    ]

    # -------- Fusion Engine --------
    fusion_cmd = [
        sys.executable,
        "-m",
        "fusion_engine.fusion_service",
    ]

    print("Starting Face Service...")
    face_proc = subprocess.Popen(face_cmd, cwd=os.path.join(BASE_DIR, "face_service"))

    print("Starting Gait Service...")
    gait_proc = subprocess.Popen(gait_cmd, cwd=os.path.join(BASE_DIR, "gait_service"))

    print("Starting Fusion Engine...")
    fusion_proc = subprocess.Popen(fusion_cmd, cwd=BASE_DIR)

    try:
        face_proc.wait()
        gait_proc.wait()
        fusion_proc.wait()
    except KeyboardInterrupt:
        print("Shutting down services...")
        face_proc.terminate()
        gait_proc.terminate()
        fusion_proc.terminate()


if __name__ == "__main__":
    main()
