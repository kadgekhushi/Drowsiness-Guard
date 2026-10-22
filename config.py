"""All tunable settings in one place. Override any of them with a JSON file (--config tuned.json)."""
import dataclasses
import json


@dataclasses.dataclass
class Config:
    # ---- camera ----
    cam_index: int = 0
    width: int = 640
    height: int = 480

    # ---- calibration (per driver) ----
    calib_seconds: float = 20.0     # keep eyes open and look at the camera
    ear_ratio: float = 0.75         # threshold = ear_ratio * driver's open-eye baseline
    ear_min: float = 0.15           # clamp so a bad calibration cannot break detection
    ear_max: float = 0.30
    ear_default: float = 0.22       # used until calibration finishes / if it fails
    smooth_frames: int = 3          # moving average on EAR

    # ---- yawning ----
    mar_thr: float = 0.65
    yawn_min_s: float = 1.5         # mouth must stay open this long to count as a yawn
    yawn_window_s: float = 300.0
    yawns_for_warning: int = 2      # this many yawns inside the window -> warning for yawn_hold_s
    yawn_hold_s: float = 10.0

    # ---- PERCLOS ----
    perclos_window_s: float = 60.0
    perclos_thr: float = 0.18

    # ---- state machine timings (seconds) ----
    warn_closed_s: float = 1.0
    alert_closed_s: float = 2.0
    face_lost_warn_s: float = 3.0
    face_lost_alert_s: float = 6.0
    ack_window_s: float = 5.0       # driver must acknowledge within this time or the vehicle stops
    clear_s: float = 1.0            # evidence must be gone this long before WARNING -> NORMAL

    # ---- head pose (experimental) ----
    use_head_pose: bool = False
    yaw_limit_deg: float = 35.0     # beyond this the driver counts as "looking away"

    # ---- optional CNN ----
    fusion: str = "ear"             # ear | cnn | and | or
    cnn_thr: float = 0.5            # P(closed) above this = closed

    # ---- face model (only needed with the newer mediapipe Tasks API) ----
    face_model_path: str = "models/face_landmarker.task"

    # ---- serial link to Arduino ----
    baud: int = 115200
    heartbeat_s: float = 0.2

    @classmethod
    def load(cls, path=None):
        cfg = cls()
        if path:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg
