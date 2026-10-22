#!/usr/bin/env python3
"""Summarise a CSV written by  main.py --log  (for the test-matrix results in your report)."""
import sys

import numpy as np
import pandas as pd

NAMES = {0: "NORMAL", 1: "WARNING", 2: "ALERT", 3: "STOP"}
df = pd.read_csv(sys.argv[1])
dt = np.diff(df.t.values, prepend=df.t.values[0])
print(f"duration: {df.t.iloc[-1] - df.t.iloc[0]:.1f} s   frames: {len(df)}   mean FPS: {df.fps.mean():.1f}")
for s, name in NAMES.items():
    print(f"  time in {name:8s}: {dt[df.state.values == s].sum():7.1f} s")
entered = df.state.values[1:][df.state.values[1:] != df.state.values[:-1]]
for s, name in NAMES.items():
    if s:
        print(f"  times {name} was entered: {int((entered == s).sum())}")
print(f"  face lost fraction: {1 - df.face.mean():.2%}   eyes-closed fraction: {df.eyes_closed.mean():.2%}")
