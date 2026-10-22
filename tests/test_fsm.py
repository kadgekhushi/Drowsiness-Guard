import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from fsm import ALERT, NORMAL, STOP, WARNING, DriverFSM


def run(fsm, t0, dur, face=True, closed=False, other=False, ack=False, dt=0.05):
    t = t0
    s = fsm.state
    while t < t0 + dur:
        t += dt
        s = fsm.update(t, face, closed, other, ack)
    return t, s


class FSMTests(unittest.TestCase):
    def setUp(self):
        self.f = DriverFSM(Config())

    def test_normal_blink_ignored(self):
        t, s = run(self.f, 0, 2)
        t, s = run(self.f, t, 0.3, closed=True)
        t, s = run(self.f, t, 2)
        self.assertEqual(s, NORMAL)

    def test_closed_1s_warning_2s_alert(self):
        t, s = run(self.f, 0, 1.2, closed=True)
        self.assertEqual(s, WARNING)
        t, s = run(self.f, t, 1.0, closed=True)
        self.assertEqual(s, ALERT)

    def test_no_ack_leads_to_latched_stop(self):
        t, s = run(self.f, 0, 2.2, closed=True)
        self.assertEqual(s, ALERT)
        t, s = run(self.f, t, 5.2, closed=True)
        self.assertEqual(s, STOP)
        t, s = run(self.f, t, 5, closed=False)      # eyes open again: still latched
        self.assertEqual(s, STOP)
        self.f.reset()
        self.assertEqual(self.f.state, NORMAL)

    def test_ack_with_eyes_open_cancels(self):
        t, s = run(self.f, 0, 2.2, closed=True)
        t, s = run(self.f, t, 1.0, closed=False)
        self.assertEqual(s, ALERT)                   # eyes open alone is not enough
        t, s = run(self.f, t, 0.1, closed=False, ack=True)
        self.assertEqual(s, NORMAL)

    def test_ack_while_closed_then_open(self):
        t, s = run(self.f, 0, 2.2, closed=True)
        t, s = run(self.f, t, 0.1, closed=True, ack=True)
        self.assertEqual(s, ALERT)
        t, s = run(self.f, t, 0.2, closed=False)
        self.assertEqual(s, NORMAL)

    def test_face_lost(self):
        t, s = run(self.f, 0, 3.2, face=False)
        self.assertEqual(s, WARNING)
        t, s = run(self.f, t, 3.0, face=False)
        self.assertEqual(s, ALERT)

    def test_warning_hysteresis(self):
        t, s = run(self.f, 0, 1.2, closed=True)
        self.assertEqual(s, WARNING)
        t, s = run(self.f, t, 0.5)
        self.assertEqual(s, WARNING)                 # still held
        t, s = run(self.f, t, 1.0)
        self.assertEqual(s, NORMAL)

    def test_other_evidence_warns(self):
        t, s = run(self.f, 0, 0.5, other=True)
        self.assertEqual(s, WARNING)


if __name__ == "__main__":
    unittest.main()
