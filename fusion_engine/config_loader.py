import yaml
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path


@dataclass
class FusionConfig:
    cameras: Dict[str, str]
    alpha: float
    time_window: float
    gait_db_path: str


def load_config(path: str = None) -> FusionConfig:
    """Load YAML configuration for fusion engine.

    The configuration path defaults to ``fusion_engine/config.yaml`` relative to
    the workspace root.  The returned object exposes camera URLs and fusion
    parameters (alpha, time window in seconds).
    """
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cameras = cfg.get("cameras", {})
    fusion = cfg.get("fusion", {})

    alpha = fusion.get("alpha", 0.65)
    time_window = fusion.get("time_window", 5)
    gait_db_path = fusion.get("gait_db_path", "gait_service/multi_gait_system/database/gait.db")

    return FusionConfig(
        cameras=cameras,
        alpha=alpha,
        time_window=time_window,
        gait_db_path=gait_db_path,
    )
