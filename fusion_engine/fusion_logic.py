from typing import Optional, Tuple


def hybrid_score(face_score: float, gait_score: float, alpha: float) -> float:
    """Compute weighted hybrid score given blending factor ``alpha``.

    The formula used throughout the project is::

        final_score = alpha * face_score + (1 - alpha) * gait_score
    """
    return alpha * face_score + (1 - alpha) * gait_score


def fuse_identity(
    face_id: Optional[str],
    face_score: float,
    gait_id: Optional[str],
    gait_score: float,
    alpha: float,
) -> Tuple[Optional[str], float]:
    """Apply the rule‑based fusion logic described in the design requirements.

    Parameters
    ----------
    face_id
        Predicted identity from the face subsystem.  ``None`` or the string
        ``"Unknown"`` are treated as unrecognized.
    face_score
        Confidence/similarity score associated with ``face_id`` (0.0-1.0).
    gait_id
        Predicted identity from the gait subsystem.  ``None`` or
        ``"Unknown"`` are treated as unrecognized.
    gait_score
        Confidence/similarity score for the gait prediction.
    alpha
        Weighting factor for score fusion (0 <= alpha <= 1).

    Returns
    -------
    (identity, final_score)
        ``identity`` may be ``None`` when the system could not make a
        definitive determination.  ``final_score`` is either a hybrid score or
        the single-modality score according to the matching rules.
    """
    # normalize unknowns
    if face_id is not None and str(face_id).lower() == "unknown":
        face_id = None
    if gait_id is not None and str(gait_id).lower() == "unknown":
        gait_id = None

    # both modalities recognized
    if face_id and gait_id:
        if face_id == gait_id:
            # agreement: compute hybrid score
            return face_id, hybrid_score(face_score, gait_score, alpha)
        else:
            # conflict: pick the modality with higher score
            if face_score >= gait_score:
                return face_id, face_score
            return gait_id, gait_score

    # single-modality recognition
    if face_id:
        return face_id, face_score
    if gait_id:
        return gait_id, gait_score

    # neither recognized
    return None, 0.0
