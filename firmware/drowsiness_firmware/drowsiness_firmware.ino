/*
  Driver Drowsiness Module - actuator controller (Arduino Uno / Nano + L298N)

  Protocol (115200 baud, newline terminated)
    Pi -> Arduino : S0 S1 S2 S3   driver state (heartbeat every ~200 ms)
                    R             reset / unlatch after a stop
    Arduino -> Pi : ACK           driver pressed the button during ALERT
                    RST           button held 2 s while stopped (local reset)
                    D,<cm>        distance from ultrasonic sensor (for logging)

  States: 0 NORMAL (run)  1 WARNING (soft beep)  2 ALERT (loud, press button)  3 STOP (ramp to 0, latched)
  Failsafe: if heartbeats stop after the link was established, the module stops the vehicle by itself.
*/

// ---------- pins ----------
const uint8_t ENA = 5, ENB = 6;                    // motor PWM (must be PWM pins)
const uint8_t IN1 = 7, IN2 = 8, IN3 = 10, IN4 = 11;
const uint8_t BUZ = 9;                             // ACTIVE buzzer (add NPN transistor if weak)
const uint8_t LED = 13;
const uint8_t BTN = 2;                             // to GND, internal pull-up
const uint8_t TRIG = 3, ECHO = 4;                  // HC-SR04

// ---------- tuning ----------
const int           CRUISE_PWM   = 120;            // 0..255, keep slow for tests
const int           MIN_PWM      = 50;             // below this the motors just hum
const unsigned long RAMP_MS      = 3000;           // time to slow from cruise to 0
const unsigned long HB_TIMEOUT   = 1000;           // no heartbeat for this long -> failsafe stop
const unsigned long DIST_EVERY   = 100;
const unsigned long LONGPRESS_MS = 2000;
const int           OBSTACLE_CM  = 15;

// ---------- state ----------
int  piState = 0;
bool linked = false;       // true after the first valid message
bool latched = false;      // STOP is latched until R or long press
unsigned long lastHB = 0, stopT = 0, lastDist = 0, btnDownT = 0, lastAck = 0;
long distCm = -1;
bool btnPrev = false, longSent = false;
char buf[12]; uint8_t n = 0;

void drive(int pwm) {
  if (pwm < MIN_PWM) pwm = 0;
  analogWrite(ENA, pwm);
  analogWrite(ENB, pwm);
}

long readDistanceCm() {
  digitalWrite(TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long us = pulseIn(ECHO, HIGH, 8000UL);  // ~1.3 m max, blocks <= 8 ms
  return us == 0 ? -1 : (long)(us / 58);
}

void handleLine() {
  buf[n] = 0;
  if (n >= 2 && buf[0] == 'S' && buf[1] >= '0' && buf[1] <= '3') {
    piState = buf[1] - '0';
    lastHB = millis();
    linked = true;
  } else if (buf[0] == 'R') {
    latched = false; stopT = 0; piState = 0; lastHB = millis();
  }
  n = 0;
}

void readSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') { if (n) handleLine(); }
    else if (n < sizeof(buf) - 1) buf[n++] = c;
    else n = 0;                                    // overflow: drop garbage
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(BUZ, OUTPUT); pinMode(LED, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  pinMode(BTN, INPUT_PULLUP);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);   // both motors forward
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  drive(0);
  lastHB = millis();
}

void loop() {
  unsigned long t = millis();
  readSerial();

  // ---- effective state (with failsafe + latch) ----
  int state = piState;
  if (linked && (t - lastHB > HB_TIMEOUT)) state = 3;      // lost the Pi
  if (state == 3) latched = true;
  if (latched) state = 3;

  // ---- button: ACK in ALERT, long press = local reset after STOP ----
  bool down = (digitalRead(BTN) == LOW);
  if (down && !btnPrev) { btnDownT = t; longSent = false; }
  if (!down && btnPrev && (t - btnDownT) > 30 && (t - btnDownT) < LONGPRESS_MS && state == 2 && t - lastAck > 300) {
    Serial.println("ACK"); lastAck = t;
  }
  if (down && !longSent && (t - btnDownT) >= LONGPRESS_MS && state == 3) {
    latched = false; stopT = 0; piState = 0; lastHB = t; longSent = true;
    Serial.println("RST");
  }
  btnPrev = down;

  // ---- obstacle sensor ----
  if (t - lastDist >= DIST_EVERY) {
    distCm = readDistanceCm(); lastDist = t;
    Serial.print("D,"); Serial.println(distCm);
  }
  bool blocked = (distCm > 0 && distCm < OBSTACLE_CM);

  // ---- outputs ----
  float spd = CRUISE_PWM;
  switch (state) {
    case 0: digitalWrite(BUZ, LOW); digitalWrite(LED, LOW); stopT = 0; break;
    case 1: digitalWrite(BUZ, (t % 1000) < 150); digitalWrite(LED, HIGH); stopT = 0; break;
    case 2: digitalWrite(BUZ, HIGH); digitalWrite(LED, (t / 150) % 2); stopT = 0; break;
    case 3:
      if (stopT == 0) stopT = t;
      spd = CRUISE_PWM * (1.0f - (float)(t - stopT) / (float)RAMP_MS);
      if (spd < 0) spd = 0;
      digitalWrite(BUZ, (t / 300) % 2); digitalWrite(LED, (t / 300) % 2);
      break;
  }
  if (!linked) { spd = 0; digitalWrite(LED, (t / 1000) % 2); }   // waiting for the Pi: stay still
  drive(blocked ? 0 : (int)spd);
}
