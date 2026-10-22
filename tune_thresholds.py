#!/usr/bin/env python3
"""Find good EAR / MAR thresholds from CSV files made by collect_data.py.

  python tune_thresholds.py data/*.csv --out tuned.json
Then run:  python main.py --config tuned.json
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def best_threshold(y_true, score):
    fpr, tpr, thr = roc_curve(y_true, score)
    j = tpr - fpr                               # Youden's J
    i = int(np.argmax(j))
    return thr[i], tpr[i], fpr[i], roc_auc_score(y_true, score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="tuned.json")
    a = ap.parse_args()
    files = [f for pat in a.csv for f in glob.glob(pat)]
    df = pd.concat([pd.read_csv(f).assign(src=f) for f in files], ignore_index=True)
    print(f"{len(df)} rows from {len(files)} file(s)")
    out = {}

    eyes = df[df.eye_label.isin(["open", "closed"])]
    if eyes.eye_label.nunique() == 2:
        y = (eyes.eye_label == "closed").astype(int)
        thr, tpr, fpr, auc = best_threshold(y, -eyes.ear)       # lower EAR = more closed
        thr = -thr
        print(f"[EAR] AUC={auc:.3f}  best absolute threshold={thr:.3f}  (catches {tpr:.0%} of closed frames, {fpr:.0%} false)")
        ratios = []
        for src, g in eyes[eyes.eye_label == "open"].groupby("src"):
            base = float(np.median(g.ear[g.ear >= g.ear.quantile(0.2)]))
            ratios.append(thr / base)
            print(f"      {src}: open-eye baseline {base:.3f} -> ratio {thr / base:.2f}")
        if ratios:
            out["ear_ratio"] = round(float(np.clip(np.median(ratios), 0.55, 0.9)), 2)
            print(f"      suggested ear_ratio = {out['ear_ratio']}")
    else:
        print("[EAR] need both 'open' and 'closed' labelled data")

    mouth = df[df.mouth_label.isin(["neutral", "yawn"])]
    if mouth.mouth_label.nunique() == 2:
        y = (mouth.mouth_label == "yawn").astype(int)
        thr, tpr, fpr, auc = best_threshold(y, mouth.mar)
        out["mar_thr"] = round(float(thr), 2)
        print(f"[MAR] AUC={auc:.3f}  best threshold={thr:.2f}  (catches {tpr:.0%} of yawn frames, {fpr:.0%} false)")
    else:
        print("[MAR] need both 'neutral' and 'yawn' labelled data")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {a.out}: {out}")


if __name__ == "__main__":
    main()
