"""Driver-state machine: NORMAL -> WARNING -> ALERT -> STOP (latched)."""
NORMAL, WARNING, ALERT, STOP = 0, 1, 2, 3
STATE_NAMES = {NORMAL: "NORMAL", WARNING: "WARNING", ALERT: "ALERT", STOP: "STOP"}


class DriverFSM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = NORMAL
        self.t_closed = None      # when the eyes were first seen closed
        self.t_lost = None        # when the face was first lost
        self.t_alert = None       # when ALERT began
        self.t_clear = None       # when evidence disappeared while in WARNING
        self.ack_latched = False

    def ack_time_left(self, now):
        if self.state != ALERT or self.t_alert is None:
            return None
        return max(0.0, self.cfg.ack_window_s - (now - self.t_alert))

    def update(self, now, face_found, eyes_closed, other_evidence=False, acked=False):
        """face_found: face visible (and not looking away); eyes_closed: only meaningful if face_found."""
        c = self.cfg
        closed = bool(face_found and eyes_closed)
        self.t_closed = (self.t_closed if self.t_closed is not None else now) if closed else None
        self.t_lost = (self.t_lost if self.t_lost is not None else now) if not face_found else None
        closed_dur = 0.0 if self.t_closed is None else now - self.t_closed
        lost_dur = 0.0 if self.t_lost is None else now - self.t_lost

        if self.state == STOP:                       # latched until reset()
            return self.state

        if self.state == ALERT:
            if acked:
                self.ack_latched = True
            if self.ack_latched and face_found and not closed:
                self.state, self.t_alert, self.ack_latched = NORMAL, None, False
            elif now - self.t_alert >= c.ack_window_s:
                self.state = STOP
            return self.state

        severe = closed_dur >= c.alert_closed_s or lost_dur >= c.face_lost_alert_s
        mild = (closed_dur >= c.warn_closed_s or lost_dur >= c.face_lost_warn_s or bool(other_evidence))
        if severe:
            self.state, self.t_alert, self.ack_latched, self.t_clear = ALERT, now, False, None
        elif mild:
            self.state, self.t_clear = WARNING, None
        elif self.state == WARNING:                  # hysteresis: wait before clearing
            if self.t_clear is None:
                self.t_clear = now
            if now - self.t_clear >= c.clear_s:
                self.state, self.t_clear = NORMAL, None
        return self.state
