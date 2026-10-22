#!/usr/bin/env python3
"""Driver drowsiness monitor: camera -> detection -> state machine -> Arduino.

Examples
  python main.py --no-serial                       # software only, webcam, with window
  python main.py --port /dev/ttyACM0               # with the Arduino connected
  python main.py --source clip.mp4 --no-serial     # run on a recorded video
  python main.py --no-display --log logs/run1.csv  # headless on the Raspberry Pi

Keys (window must be focused):  a = acknowledge   r = reset after STOP   c = recalibrate   q = quit
"""
import argparse
import csv
import threading
import time

import cv2
import numpy as np

from config import Config
from detector import (EarCalibrator, FaceAnalyzer, Measurement, PerclosTracker, Smoother, YawnTracker)
from fsm import ALERT, NORMAL, STATE_NAMES, STOP, WARNING, DriverFSM
from serial_link import NullLink, SerialLink

COLORS = {NORMAL: (60, 180, 60), WARNING: (0, 200, 255), ALERT: (0, 0, 255), STOP: (0, 0, 160)}


class VideoSource:
    """Webcam (threaded, always latest frame) or video file (sequential, deterministic time)."""

    def __init__(self, src, w, h):
        self.is_file = isinstance(src, str)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {src!r}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = self._seen = 0
        self._frame = None
        self._run = True
        self._lock = threading.Lock()
        if not self.is_file:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._n = f, self._n + 1
            else:
                time.sleep(0.05)

    def read(self):
        """Returns (ok, frame, timestamp_seconds)."""
        if self.is_file:
            ok, f = self.cap.read()
            self._n += 1
            return ok, f, self._n / self.fps
        deadline = time.monotonic() + 0.5          # camera unplugged -> report failure after 0.5 s
        while self._run and time.monotonic() < deadline:
            with self._lock:
                if self._n > self._seen and self._frame is not None:
                    self._seen = self._n
                    return True, self._frame.copy(), time.monotonic()
            time.sleep(0.002)
        return False, None, time.monotonic()

    def close(self):
        self._run = False
        self.cap.release()


def fuse(mode, ear_closed, cnn_closed):
    if cnn_closed is None or mode == "ear":
        return ear_closed
    if mode == "cnn":
        return cnn_closed
    if mode == "and":
        return ear_closed and cnn_closed
    if mode == "or":
        return ear_closed or cnn_closed
    return ear_closed


def put(frame, text, y, color=(255, 255, 255), scale=0.55):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index or video file path")
    p.add_argument("--config", help="JSON file with overrides (e.g. tuned.json)")
    p.add_argument("--port", help="serial port, e.g. /dev/ttyACM0 or COM3 (default: auto-detect)")
    p.add_argument("--no-serial", action="store_true", help="do not talk to the Arduino")
    p.add_argument("--no-display", action="store_true", help="headless mode (no window)")
    p.add_argument("--mirror", action="store_true", help="flip the image horizontally")
    p.add_argument("--calib-seconds", type=float, help="override calibration length")
    p.add_argument("--skip-calib", action="store_true", help="use the default EAR threshold")
    p.add_argument("--cnn", help="path to eye_cnn.tflite (optional)")
    p.add_argument("--fusion", choices=["ear", "cnn", "and", "or"], help="how to combine EAR and CNN")
    p.add_argument("--head-pose", action="store_true", help="enable experimental head-pose gating")
    p.add_argument("--log", help="write a CSV log to this path")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config.load(a.config)
    if a.calib_seconds is not None:
        cfg.calib_seconds = a.calib_seconds
    if a.head_pose:
        cfg.use_head_pose = True
    cnn = None
    if a.cnn:
        from cnn_eye import EyeCNN
        cnn = EyeCNN(a.cnn)
        cfg.fusion = a.fusion or ("and" if cfg.fusion == "ear" else cfg.fusion)
    elif a.fusion:
        cfg.fusion = a.fusion

    src = VideoSource(int(a.source) if a.source.isdigit() else a.source, cfg.width, cfg.height)
    analyzer = FaceAnalyzer(cfg, want_crops=cnn is not None)
    cal, smoother = EarCalibrator(cfg), Smoother(cfg.smooth_frames)
    perclos, yawn, fsm = PerclosTracker(cfg.perclos_window_s), YawnTracker(cfg), DriverFSM(cfg)
    link = NullLink() if a.no_serial else SerialLink(a.port, cfg.baud, cfg.heartbeat_s)
    link.start()

    calibrating = not a.skip_calib
    logf = writer = None
    if a.log:
        logf = open(a.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "face", "ear", "ear_smooth", "thr", "mar", "perclos", "eyes_closed", "state", "fps"])

    state_time = {s: 0.0 for s in STATE_NAMES}
    prev_t, fps, n_frames, first = None, 0.0, 0, True
    print("Running. Keys: a=ack  r=reset  c=recalibrate  q=quit")
    try:
        while True:
            ok, frame, now = src.read()
            if not ok and src.is_file:
                break
            if ok and a.mirror:
                frame = cv2.flip(frame, 1)
            meas = analyzer.process(frame) if ok else Measurement()
            face_ok = meas.face_found and not meas.looking_away
            ear_s = smoother.update(meas.ear) if face_ok else (smoother.clear() or 1.0)

            if prev_t is not None and now > prev_t:
                fps = 0.9 * fps + 0.1 / (now - prev_t) if fps else 1.0 / (now - prev_t)
                state_time[fsm.state] += now - prev_t
            prev_t = now
            n_frames += 1

            if calibrating:
                if first:
                    cal.start(now)
                cal.feed(now, meas.ear, face_ok)
                if cal.done:
                    calibrating = False
                    print(f"Calibrated: baseline EAR={cal.baseline}, threshold={cal.threshold:.3f}")
                state, eyes_closed = NORMAL, False
            else:
                ear_closed = face_ok and ear_s < cal.threshold
                cnn_closed = None
                if cnn is not None and meas.eye_crops:
                    cnn_closed = float(np.mean([cnn.prob_closed(c) for c in meas.eye_crops])) >= cfg.cnn_thr
                eyes_closed = bool(face_ok and fuse(cfg.fusion, ear_closed, cnn_closed))
                if face_ok:
                    perclos.add(now, eyes_closed)
                yawn.update(now, face_ok and meas.mar > cfg.mar_thr)
                other = yawn.evidence(now) or perclos.fraction() >= cfg.perclos_thr
                acked = link.consume_ack()
                if link.consume_reset():
                    fsm.reset()
                state = fsm.update(now, face_ok, eyes_closed, other, acked)
            first = False
            link.set_state(state)

            key = -1
            if not a.no_display:
                vis = frame if frame is not None else np.zeros((cfg.height, cfg.width, 3), np.uint8)
                put(vis, f"STATE: {STATE_NAMES[state]}", 25, COLORS[state], 0.8)
                if calibrating:
                    put(vis, f"CALIBRATING {int(cal.progress(now) * 100)}%  - eyes open, look at camera", 55, (0, 255, 255))
                else:
                    put(vis, f"EAR {meas.ear:.2f} (thr {cal.threshold:.2f})  MAR {meas.mar:.2f}", 55)
                    put(vis, f"PERCLOS {perclos.fraction():.2f}  yawns {len(yawn.events)}  FPS {fps:.0f}", 80)
                    left = fsm.ack_time_left(now)
                    if left is not None:
                        put(vis, f"PRESS ACK! stop in {left:.1f}s", 110, (0, 0, 255), 0.8)
                    if state == STOP:
                        put(vis, "VEHICLE STOPPED - press r to reset", 110, (0, 0, 255), 0.7)
                if not meas.face_found:
                    put(vis, "NO FACE", 135, (0, 0, 255))
                if state in (ALERT, STOP):
                    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), COLORS[state], 6)
                cv2.imshow("Driver Drowsiness Monitor", vis)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                fsm.ack_latched = fsm.state == ALERT or fsm.ack_latched
            if key == ord("r"):
                fsm.reset()
                link.send_reset()
            if key == ord("c"):
                calibrating, first = True, True

            if writer:
                writer.writerow([f"{now:.3f}", int(face_ok), f"{meas.ear:.4f}", f"{ear_s:.4f}", f"{cal.threshold:.4f}",
                                 f"{meas.mar:.4f}", f"{perclos.fraction():.3f}", int(eyes_closed), state, f"{fps:.1f}"])
    except KeyboardInterrupt:
        pass
    finally:
        link.set_state(fsm.state)
        link.stop()
        src.close()
        if logf:
            logf.close()
        cv2.destroyAllWindows()
        print("\nTime in each state (s):", {STATE_NAMES[k]: round(v, 1) for k, v in state_time.items()})


if __name__ == "__main__":
    main()
