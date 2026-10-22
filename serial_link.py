"""Link to the Arduino: sends the driver state as a heartbeat, reads ACK / RST / distance."""
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:            # lets the rest of the project run without pyserial
    serial = None
    list_ports = None


def find_port():
    if list_ports is None:
        return None
    ports = list(list_ports.comports())
    for p in ports:
        text = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(k in text for k in ("arduino", "ch340", "ttyacm", "usb serial", "usbmodem")):
            return p.device
    for p in ports:
        if "usb" in p.device.lower():
            return p.device
    return None


class NullLink:
    """Same API as SerialLink but prints instead of talking to hardware (for software-only testing)."""
    connected = False
    distance_cm = None

    def __init__(self, *a, **k):
        self._last = None
        self._ack = False
        self._rst = False

    def start(self): pass
    def stop(self): pass

    def set_state(self, s):
        if s != self._last:
            print(f"[link] state -> {s}")
            self._last = s

    def send_reset(self): print("[link] reset sent")
    def consume_ack(self): return False
    def consume_reset(self): return False


class SerialLink:
    def __init__(self, port=None, baud=115200, heartbeat_s=0.2):
        if serial is None:
            raise RuntimeError("pyserial is not installed: pip install pyserial")
        self.port, self.baud, self.hb = port, baud, heartbeat_s
        self.connected = False
        self.distance_cm = None
        self._state, self._ack, self._rst, self._send_reset = 0, False, False, False
        self._lock = threading.Lock()
        self._ser, self._buf = None, b""
        self._run, self._thread = False, None

    # ---- public API (called from the main loop) ----
    def start(self):
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._close()

    def set_state(self, s):
        with self._lock:
            self._state = int(s)

    def send_reset(self):
        with self._lock:
            self._send_reset = True

    def consume_ack(self):
        with self._lock:
            v, self._ack = self._ack, False
            return v

    def consume_reset(self):
        with self._lock:
            v, self._rst = self._rst, False
            return v

    # ---- internals ----
    def _open(self):
        port = self.port or find_port()
        if not port:
            return False
        try:
            self._ser = serial.Serial(port, self.baud, timeout=0)
            time.sleep(2.0)                       # Arduino resets when the port opens
            self._buf = b""
            self.connected = True
            print(f"[link] connected on {port}")
            return True
        except (serial.SerialException, OSError) as e:
            print(f"[link] cannot open {port}: {e}")
            self._ser = None
            return False

    def _close(self):
        self.connected = False
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def _read_lines(self):
        data = self._ser.read(256)
        if not data:
            return
        self._buf += data
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip().decode(errors="ignore")
            with self._lock:
                if line == "ACK":
                    self._ack = True
                elif line == "RST":
                    self._rst = True
                elif line.startswith("D,"):
                    try:
                        self.distance_cm = int(line[2:])
                    except ValueError:
                        pass
        if len(self._buf) > 1024:
            self._buf = b""

    def _loop(self):
        last_sent, last_state = 0.0, None
        while self._run:
            if not self.connected and not self._open():
                time.sleep(1.0)
                continue
            try:
                now = time.monotonic()
                with self._lock:
                    st, do_reset = self._state, self._send_reset
                    self._send_reset = False
                if do_reset:
                    self._ser.write(b"R\n")
                if st != last_state or now - last_sent >= self.hb:
                    self._ser.write(f"S{st}\n".encode())
                    last_sent, last_state = now, st
                self._read_lines()
            except (serial.SerialException, OSError) as e:
                print(f"[link] lost connection: {e}")
                self._close()
                last_state = None
            time.sleep(0.01)
