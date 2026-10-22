import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from detector import (EarCalibrator, PerclosTracker, YawnTracker, eye_aspect_ratio, mouth_aspect_ratio)


def eye(open_h):
    # corners at x=0 and x=10; lids at +/- open_h around y=0
    return np.array([[0, 0], [3, open_h], [7, open_h], [10, 0], [7, -open_h], [3, -open_h]], float)


class DetectorTests(unittest.TestCase):
    def test_ear_open_vs_closed(self):
        self.assertGreater(eye_aspect_ratio(eye(1.5)), 0.28)
        self.assertLess(eye_aspect_ratio(eye(0.2)), 0.1)

    def test_mar(self):
        lm = np.zeros((478, 2))
        lm[78], lm[308] = (0, 0), (10, 0)
        lm[13], lm[14] = (5, -4), (5, 4)
        self.assertAlmostEqual(mouth_aspect_ratio(lm), 0.8)

    def test_calibrator_ignores_blinks(self):
        cfg = Config(calib_seconds=10)
        cal = EarCalibrator(cfg)
        rng = np.random.default_rng(0)
        for i in range(300):
            v = 0.30 + rng.normal(0, 0.005)
            if i % 40 == 0:
                v = 0.08                        # blink
            cal.feed(i * 10 / 300.0, v, True)
        cal.feed(10.5, 0.3, True)
        self.assertTrue(cal.done)
        self.assertAlmostEqual(cal.baseline, 0.30, delta=0.02)
        self.assertAlmostEqual(cal.threshold, 0.30 * 0.75, delta=0.02)

    def test_perclos(self):
        p = PerclosTracker(60)
        for i in range(0, 121):
            p.add(i * 0.5, closed=(i % 4 == 0 or i % 4 == 1))   # closed half the time
        self.assertAlmostEqual(p.fraction(), 0.5, delta=0.05)

    def test_perclos_needs_history(self):
        p = PerclosTracker(60)
        for i in range(20):
            p.add(i * 0.5, True)
        self.assertEqual(p.fraction(), 0.0)

    def test_yawn(self):
        y = YawnTracker(Config())
        for i in range(10):
            y.update(i * 0.1, True)
        self.assertFalse(y.yawning)             # 1 s < 1.5 s
        for i in range(10, 25):
            y.update(i * 0.1, True)
        self.assertTrue(y.yawning)
        self.assertEqual(len(y.events), 1)


if __name__ == "__main__":
    unittest.main()
