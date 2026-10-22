# Driver Drowsiness Detection & Vehicle Stopping Module - code + build guide

Prototype for a **small motorized test vehicle**. Not a certified safety system; never connect it to the
brakes or controls of a real vehicle.

## What is in this folder

| File | Purpose |
|---|---|
| `main.py` | The live system: camera -> detection -> state machine -> Arduino |
| `detector.py` | MediaPipe landmarks, EAR, MAR, per-driver calibration, PERCLOS, yawn tracking, (experimental) head pose |
| `fsm.py` | NORMAL -> WARNING -> ALERT -> STOP state machine (STOP is latched) |
| `serial_link.py` | Heartbeat + ACK/RST/distance link to the Arduino (`NullLink` = fake link for software-only tests) |
| `config.py` | Every threshold/timing in one place; override with a JSON file |
| `collect_data.py` / `tune_thresholds.py` | Record your own labelled EAR/MAR data, then compute best thresholds (ROC) |
| `train_cnn.py` / `cnn_eye.py` | Optional CNN eye-state model (Kaggle MRL dataset) -> TFLite, and its inference wrapper |
| `summarize_log.py` | Turns a `--log` CSV into numbers for your report |
| `firmware/drowsiness_firmware/*.ino` | Arduino code: buzzer, LED, ramp-down stop, failsafe, ack button, ultrasonic |
| `tests/` | 14 unit tests (state machine + detector maths) - `python -m unittest discover -s tests` |
| `deploy/drowsiness.service` | Autostart on the Raspberry Pi |

## Protocol (Pi <-> Arduino, 115200 baud, newline-terminated)

`S0..S3` Pi->Arduino heartbeat every 200 ms (0 normal, 1 warning, 2 alert, 3 stop) | `R` reset after stop |
`ACK` button pressed in ALERT | `RST` button held 2 s while stopped | `D,<cm>` distance (logging)

---------------------------------------------------------------------------------------------------

# STEP-BY-STEP IMPLEMENTATION

## Step 1 - Install (laptop first)
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests # must say OK (14 tests)
```
**MediaPipe version matters.** `requirements.txt` pins `mediapipe==0.10.14`, which has the `mp.solutions` API this code
uses (needs Python 3.9-3.12 and numpy<2). Very new MediaPipe releases removed `mp.solutions`; the code then falls back to
the Tasks API, which needs a model file: download
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
into `models/face_landmarker.task`. (The Tasks path could not be tested with a real face while writing this - if it
misbehaves, install 0.10.14 in a Python 3.10/3.11 virtual environment.)

## Step 2 - Check the camera and landmarks
```bash
python collect_data.py --out data/test.csv        # shows live EAR/MAR; press q to quit
```
You should see EAR around 0.25-0.35 with eyes open and below ~0.15 when you close them; MAR rises when you open your
mouth wide. If "NO FACE" shows constantly: improve light, face the camera, check `--camera 1`.
Camera placement: 40-70 cm away, slightly below eye level, face filling about a third of the frame.

## Step 3 - Record your own data (this is what makes it accurate)
Run `collect_data.py --out data/<name>_<condition>.csv --note "<name>,glasses"` per session (instructions are in the
file header: o = open, c = closed, y = yawn, n = neutral). Collect at least 5 people, with/without glasses, bright/dim.
Get consent; delete videos/data afterwards unless people agree otherwise.

## Step 4 - Tune thresholds
```bash
python tune_thresholds.py data/*.csv --out tuned.json
```
It prints the ROC-based best EAR threshold, a recommended `ear_ratio` (threshold relative to each driver's calibrated
baseline) and the MAR yawn threshold. Use them with `--config tuned.json`.

## Step 5 - Run the software alone (no hardware)
```bash
python main.py --no-serial --config tuned.json
```
1. First ~20 s = calibration (eyes open, look at camera). 2. Close your eyes 1 s -> WARNING. 3. 2 s -> ALERT and a
5 s countdown. 4. Press **a** (acknowledge) with eyes open -> back to NORMAL; or do nothing -> STOP (latched; press **r**).
5. Cover the camera / look away 3 s -> WARNING, 6 s -> ALERT.
Test on a recorded clip: `python main.py --source clip.mp4 --no-serial --no-display --log logs/clip.csv`.

## Step 6 - (Optional) CNN eye model
Download the Kaggle "MRL Eye Dataset", then on a PC with TensorFlow:
```bash
pip install -r requirements-train.txt
python train_cnn.py --data /path/to/mrlEyes --out models
python main.py --no-serial --cnn models/eye_cnn.tflite --fusion and --config tuned.json
```
`--fusion and` = eyes count as closed only if EAR **and** CNN agree (fewer false alarms, e.g. with glasses);
`or` = more sensitive; `cnn` = CNN only. Compare all modes on your own data and report the difference.
On the Pi install `tflite-runtime` (or `ai-edge-litert`) instead of full TensorFlow; if neither is available for your
Python version use TensorFlow or skip the CNN (EAR alone is the MVP).

## Step 7 - Build and test the hardware (Friend B)
1. Wire exactly as in the PDF plan (Section 5.2) / pins at the top of the `.ino` file. **Common ground** between
   motor battery, driver and Arduino; Pi on its own power bank; kill switch in the motor battery line.
2. Arduino IDE -> open `firmware/drowsiness_firmware/drowsiness_firmware.ino` -> select board/port -> Upload.
3. Open Serial Monitor at 115200, line ending "Newline". Type `S0` (LED off, wheels slow forward once linked),
   `S1` (soft beeps), `S2` (continuous buzzer; press the button -> "ACK" printed), `S3` (3 s ramp to stop, latched).
   Type `R` to unlatch. Stop sending anything for 1 s -> failsafe stop. **Wheels lifted off the ground for all first tests.**
4. Tune `CRUISE_PWM` (slow!) and `RAMP_MS`. Measure stopping distance 10 times and record it.

## Step 8 - Integrate Pi/laptop + Arduino
```bash
python main.py --port /dev/ttyACM0 --config tuned.json     # Windows: --port COM3  (omit to auto-detect)
```
Close your eyes: WARNING (beeps) -> ALERT (loud, press the real button) -> no press = ramp-stop. Hold the button 2 s
(or press **r** in the window) to reset. Unplug the USB cable while driving: the Arduino must stop by itself.
Then floor tests: low speed, clear area, kill switch in hand, a spotter.

## Step 9 - Raspberry Pi deployment
```bash
sudo apt update && sudo apt install -y python3-venv python3-opencv libatlas-base-dev   # 64-bit Raspberry Pi OS
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python main.py --no-display --no-serial --log logs/t.csv     # check FPS in summarize_log.py; aim for 15+
```
If FPS is low: lower `width/height` in `config.py` (e.g. 480x360), skip the CNN, close other programs, or use a
Pi 4/5. Autostart: see `deploy/drowsiness.service`. If `pip install mediapipe` fails on the Pi, confirm you are on
64-bit OS and Python 3.9-3.11.

## Step 10 - Evaluation for the report
* `python main.py --log logs/<test>.csv ...` for each test in the PDF's test matrix, then `python summarize_log.py logs/<test>.csv`.
* Offline accuracy: the CNN script prints a confusion matrix/F1 on unseen subjects; `tune_thresholds.py` prints AUC for EAR/MAR.
* Physical: stopping distance x10, failsafe test (unplug USB), obstacle stop, ack button 10/10.
* False alarms: 30 min of normal behaviour per volunteer (talking, laughing, looking at mirrors) - count WARNING entries.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.14` (Python 3.10/3.11) or use the Tasks model file (Step 1) |
| Constant false WARNING | Recalibrate (press c), raise `warn_closed_s`, check light/glare, try `--fusion and` with the CNN |
| Eyes closed not detected with glasses | Reduce glare (move light), use IR camera, tune with glasses data, use `--fusion or` |
| Arduino resets when Python starts | Normal (DTR reset); the link waits 2 s after opening the port |
| Motors stutter / Pi reboots | Separate supplies, common ground, capacitors across motors |
| Vehicle never moves | It waits for the first heartbeat: run `main.py` with the right `--port` |
| Yawning not detected | Run `collect_data.py` yawn sessions and `tune_thresholds.py`; lower `yawn_min_s` |

## Honest limitations
* The state machine, calibration maths, PERCLOS/yawn logic and the full pipeline are unit/smoke-tested, but the
  landmark detection, CNN training and Arduino firmware could not be run on real hardware while writing this - expect
  to adjust thresholds, pins and PWM values on your setup.
* Head-pose gating (`--head-pose`) is experimental; check the yaw signs on your camera before relying on it.
* Sunglasses block the eyes: the system treats a lost face/eyes as a warning condition, not a perfect detection.
