#!/usr/bin/env python3
"""Record labelled EAR / MAR values from your own webcam so you can tune thresholds.

Keys:  o = eyes OPEN (normal)      c = eyes CLOSED (hold them closed while this label is active)
       y = YAWNING                 n = mouth neutral (default)
       q = quit
Typical session (about 3 minutes per person):
  1) press o, look at the camera normally for 20 s (blink naturally)
  2) press c, close your eyes for 10 s, press o, repeat 3 times
  3) press y and yawn widely 5 times (press n between yawns), then press n and talk normally for 20 s
Repeat with glasses, dim light, and with other volunteers (with their consent).
"""
import argparse
import csv
import time

import cv2

from config import Config
from detector import FaceAnalyzer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", required=True, help="CSV file, e.g. data/akshata_normal.csv")
    p.add_argument("--note", default="", help="free text stored with each row, e.g. 'glasses,dim'")
    a = p.parse_args()

    cfg = Config()
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    analyzer = FaceAnalyzer(cfg)
    eye_label, mouth_label = "open", "neutral"
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ear", "mar", "eye_label", "mouth_label", "note"])
        t0 = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            m = analyzer.process(frame)
            if m.face_found:
                w.writerow([f"{time.monotonic() - t0:.3f}", f"{m.ear:.4f}", f"{m.mar:.4f}", eye_label, mouth_label, a.note])
            txt = f"eyes:{eye_label}  mouth:{mouth_label}  EAR {m.ear:.2f} MAR {m.mar:.2f}" + ("" if m.face_found else "  NO FACE")
            cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            eye_label = {ord("o"): "open", ord("c"): "closed"}.get(k, eye_label)
            mouth_label = {ord("y"): "yawn", ord("n"): "neutral"}.get(k, mouth_label)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
