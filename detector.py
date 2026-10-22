"""Face landmarks, EAR / MAR, per-driver calibration, PERCLOS and yawn tracking."""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# MediaPipe Face Mesh landmark indices
L_EYE = [362, 385, 387, 263, 373, 380]   # p1..p6 (corner, upper, upper, corner, lower, lower)
R_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_V = (13, 14)                        # inner upper / lower lip
MOUTH_H = (78, 308)                       # mouth corners

# Head pose (experimental): nose, chin, right-eye outer, left-eye outer, mouth corners
POSE_IDX = [1, 152, 33, 263, 61, 291]
POSE_3D = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def eye_aspect_ratio(p) -> float:
    """p: six (x, y) points ordered p1..p6. EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    den = 2.0 * _dist(p[0], p[3])
    if den < 1e-6:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / den


def mouth_aspect_ratio(lm) -> float:
    den = _dist(lm[MOUTH_H[0]], lm[MOUTH_H[1]])
    if den < 1e-6:
        return 0.0
    return _dist(lm[MOUTH_V[0]], lm[MOUTH_V[1]]) / den


def head_pose(lm, w, h):
    """Returns (pitch, yaw) in degrees. Experimental - verify signs on your own camera."""
    img = np.asarray(lm)[POSE_IDX].astype(np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_3D, img, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw = float(angles[0]), float(angles[1])
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw


@dataclass
class Measurement:
    face_found: bool = False
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    looking_away: bool = False
    landmarks: Optional[np.ndarray] = None
    eye_crops: Optional[list] = None      # two 64x64 uint8 grayscale crops (only if want_crops)


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


class FaceAnalyzer:
    """MediaPipe face landmarks. Works with the old `mp.solutions` API (mediapipe 0.10.14) and
    falls back to the newer Tasks API (needs models/face_landmarker.task, see README)."""

    def __init__(self, cfg, want_crops: bool = False):
        import mediapipe as mp   # imported here so the rest of the module works without it
        self.mp, self.cfg, self.want_crops = mp, cfg, want_crops
        self._ts = 0
        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            import os
            from mediapipe.tasks.python import BaseOptions, vision
            if not os.path.exists(cfg.face_model_path):
                raise RuntimeError(
                    f"Your mediapipe version has no mp.solutions. Either `pip install mediapipe==0.10.14` "
                    f"or download the model to {cfg.face_model_path}:\n  {MODEL_URL}")
            self.backend = "tasks"
            self.mesh = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=cfg.face_model_path),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5))

    def _landmarks(self, frame_bgr, w, h):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            pts = res.multi_face_landmarks[0].landmark
        else:
            self._ts += 33                                   # must be strictly increasing (ms)
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.mesh.detect_for_video(img, self._ts)
            if not res.face_landmarks:
                return None
            pts = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in pts])

    @staticmethod
    def _crop(gray, pts, size=64):
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad = 0.35 * max(x1 - x0, 1.0)
        h, w = gray.shape
        xa, xb = int(max(x0 - pad, 0)), int(min(x1 + pad, w))
        ya, yb = int(max(y0 - pad, 0)), int(min(y1 + pad, h))
        if xb - xa < 4 or yb - ya < 4:
            return None
        return cv2.resize(gray[ya:yb, xa:xb], (size, size))

    def process(self, frame_bgr) -> Measurement:
        h, w = frame_bgr.shape[:2]
        lm = self._landmarks(frame_bgr, w, h)
        if lm is None:
            return Measurement()
        m = Measurement(face_found=True, landmarks=lm)
        m.ear = (eye_aspect_ratio(lm[L_EYE]) + eye_aspect_ratio(lm[R_EYE])) / 2.0
        m.mar = mouth_aspect_ratio(lm)
        if self.cfg.use_head_pose:
            m.pitch, m.yaw = head_pose(lm, w, h)
            m.looking_away = abs(m.yaw) > self.cfg.yaw_limit_deg
        if self.want_crops:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            crops = [self._crop(gray, lm[L_EYE]), self._crop(gray, lm[R_EYE])]
            m.eye_crops = [c for c in crops if c is not None] or None
        return m


class Smoother:
    def __init__(self, n: int):
        self.d = collections.deque(maxlen=max(1, n))

    def update(self, v: float) -> float:
        self.d.append(v)
        return sum(self.d) / len(self.d)

    def clear(self):
        self.d.clear()


class EarCalibrator:
    """Measures the driver's normal open-eye EAR and derives a personal threshold."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = []
        self.t0 = None
        self.done = False
        self.baseline = None
        self.threshold = cfg.ear_default

    def start(self, now: float):
        self.samples, self.t0, self.done, self.baseline = [], now, False, None
        self.threshold = self.cfg.ear_default

    def progress(self, now: float) -> float:
        if self.t0 is None:
            return 0.0
        return min(1.0, (now - self.t0) / max(self.cfg.calib_seconds, 1e-6))

    def feed(self, now: float, ear: float, face_ok: bool):
        if self.t0 is None:
            self.start(now)
        if face_ok:
            self.samples.append(ear)
        if now - self.t0 >= self.cfg.calib_seconds:
            self.finish()

    def finish(self):
        if len(self.samples) >= 30:
            s = np.asarray(self.samples)
            open_s = s[s >= np.percentile(s, 20)]          # drop blinks (lowest 20%)
            self.baseline = float(np.median(open_s))
            thr = self.baseline * self.cfg.ear_ratio
            self.threshold = float(min(max(thr, self.cfg.ear_min), self.cfg.ear_max))
        self.done = True


class PerclosTracker:
    """Fraction of time the eyes were closed over a sliding window."""

    def __init__(self, window_s: float):
        self.w = window_s
        self.d = collections.deque()

    def add(self, now: float, closed: bool):
        self.d.append((now, bool(closed)))
        while len(self.d) > 2 and now - self.d[1][0] > self.w:
            self.d.popleft()

    def fraction(self) -> float:
        if len(self.d) < 2:
            return 0.0
        span = self.d[-1][0] - self.d[0][0]
        if span < 0.5 * self.w:           # not enough history yet -> never flag
            return 0.0
        closed, prev = 0.0, None
        for t, c in self.d:
            if prev is not None and prev[1]:
                closed += t - prev[0]
            prev = (t, c)
        return closed / span


class YawnTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._start = None
        self._counted = False
        self.yawning = False
        self.events = collections.deque()

    def update(self, now: float, mouth_open: bool):
        if mouth_open:
            if self._start is None:
                self._start, self._counted = now, False
            self.yawning = (now - self._start) >= self.cfg.yawn_min_s
            if self.yawning and not self._counted:
                self.events.append(now)
                self._counted = True
        else:
            self._start, self._counted, self.yawning = None, False, False
        while self.events and now - self.events[0] > self.cfg.yawn_window_s:
            self.events.popleft()

    def evidence(self, now: float) -> bool:
        if self.yawning:
            return True
        return (len(self.events) >= self.cfg.yawns_for_warning
                and now - self.events[-1] <= self.cfg.yawn_hold_s)
