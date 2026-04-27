"""
Visualisation utilities for the real-time display window.
"""

import cv2
import numpy as np
from typing import List
from tracking.tracker import Track


COLOUR_KNOWN   = (0, 220, 0)
COLOUR_UNKNOWN = (0, 0, 220)
COLOUR_PENDING = (200, 200, 0)
FONT           = cv2.FONT_HERSHEY_SIMPLEX


def draw_tracks(
    frame:       np.ndarray,
    tracks:      List[Track],
    identities:  dict,          # track_id → MatchResult | None
    frame_counts: dict,         # track_id → int
    min_frames:  int = 20,
) -> np.ndarray:
    out = frame.copy()

    for track in tracks:
        tid    = track.track_id
        result = identities.get(tid)
        count  = frame_counts.get(tid, 0)
        x1, y1, x2, y2 = track.bbox

        if result is None:
            colour = COLOUR_PENDING
            label  = f"ID:{tid}  [{count}/{min_frames}]"
        elif result.is_known:
            colour = COLOUR_KNOWN
            label  = f"{result.identity_id}  d={result.distance:.2f}"
        else:
            colour = COLOUR_UNKNOWN
            label  = f"Unknown  d={result.distance:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        _put_label(out, label, x1, y1, colour)

    # HUD
    n_known = sum(1 for r in identities.values() if r and r.is_known)
    cv2.putText(out, f"DB: {n_known} matched", (10, 28), FONT, 0.7, (255, 255, 255), 2)
    return out


def _put_label(frame, text, x, y, colour):
    (w, h), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    y0 = max(y - 6, h + 4)
    cv2.rectangle(frame, (x, y0 - h - 4), (x + w + 4, y0 + 2), colour, -1)
    cv2.putText(frame, text, (x + 2, y0), FONT, 0.55, (0, 0, 0), 1)
