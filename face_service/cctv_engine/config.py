"""
Configuration constants for CCTV engine modules.
"""

# Performance / logic parameters
DETECTION_INTERVAL_FRAMES = 5          # run person detection every N frames
COOLDOWN_SECONDS = 45                  # in-memory cooldown per person
STATE_LOCK_CONSECUTIVE = 5             # number of consecutive frames same id required
TRACK_STATE_CLEANUP_SECONDS = 2.0      # remove stale tracks after this inactivity
MAX_TRACKER_DISPLACEMENT = 100.0       # pixels (used by fallback centroid tracker)

# YOLO model selection (legacy – no longer used by in-process engine)
YOLO_MODEL = "yolov8n.pt"            # default YOLOv8 weights for person detection
YOLO_CONF_THRESHOLD = 0.3              # detection confidence filter
YOLO_CLASSES = [0]                     # class 0 = person in COCO

# Recognition client defaults (used only by the now‑deprecated HTTP client)
RECOGNITION_TOP_K = 1                 # only ask for top match
RECOGNITION_TIMEOUT = 5.0             # seconds for HTTP request


# Overlay drawing parameters (cv2 will be imported by modules using these constants)
FONT_SCALE = 0.5
THICKNESS = 1

# Misc
DISPLAY_WINDOW_TITLE = "CCTV Bio Surveillance"