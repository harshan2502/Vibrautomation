#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>

#define MPU_ADDR       0x68
#define WINDOW_SIZE    50
#define MOTOR_PWM      25
#define MOTOR_DIR      26
#define GREEN_LED      2
#define YELLOW_LED     4
#define RED_LED        5
#define BUZZER         18
#define PWM_CHANNEL    0
#define PWM_FREQ       1000
#define PWM_RESOLUTION 8

const char* ssid     = "Harshan's Oneplus";
const char* password = "12345678";
String tsApiKey      = "SKID9I3P6R60DIP3";
const char* flaskURL = "http://10.68.18.6:5001/api/data";

LiquidCrystal_I2C lcd(0x3F, 16, 2);

float vibrationBuffer[WINDOW_SIZE];
int   bufferIndex      = 0;
float baselineRMS      = 0;
float baselineKurtosis = 0;
float currentRMS       = 0;
float currentKurtosis  = 0;
float healthScore      = 100.0;
int   motorSpeed       = 200;
unsigned long lastUpload     = 0;
unsigned long lastBuzz       = 0;
unsigned long lastSpeedCheck = 0;

void setup() {
  Serial.begin(115200);
  pinMode(GREEN_LED,  OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED,    OUTPUT);
  pinMode(BUZZER,     OUTPUT);

  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_PWM, PWM_CHANNEL);
  pinMode(MOTOR_DIR, OUTPUT);
  digitalWrite(MOTOR_DIR, HIGH);
  ledcWrite(PWM_CHANNEL, motorSpeed);

  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Vibrautomation");
  lcd.setCursor(0, 1);
  lcd.print("Health Analyser");
  delay(2000);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  WiFi.begin(ssid, password);
  lcd.clear();
  lcd.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // ── WiFi connected info ──────────────────
  Serial.println("\nWiFi Connected!");
  Serial.print("Network: ");
  Serial.println(ssid);
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Flask URL: ");
  Serial.println(flaskURL);
  Serial.println("Ready to send data!");
  // ─────────────────────────────────────────

  lcd.clear();
  lcd.print("WiFi Connected!");
  lcd.setCursor(0, 1);
  lcd.print(WiFi.localIP());
  delay(2000);

  lcd.clear();
  lcd.print("Calibrating...");
  lcd.setCursor(0, 1);
  lcd.print("Keep motor still");
  calibrateAI();

  lcd.clear();
  lcd.print("System Ready!");
  lcd.setCursor(0, 1);
  lcd.print("Health: 100%");
  delay(1500);
}

void loop() {
  checkSpeed();
  readVibrationAI();
  calculateAIFeatures();
  calculateHealthScore();
  updateLCD();
  updateLEDs();

  Serial.print("RMS: ");     Serial.print(currentRMS, 3);
  Serial.print(" Kurt: ");   Serial.print(currentKurtosis, 2);
  Serial.print(" Health: "); Serial.print(healthScore, 1);
  Serial.println("%");

  if (millis() - lastUpload > 15000) {
    sendToThingSpeak(currentRMS, currentKurtosis, healthScore);
    sendToFlask(currentRMS, currentKurtosis, healthScore);
    lastUpload = millis();
  }

  delay(200);
}

void checkSpeed() {
  if (millis() - lastSpeedCheck < 3000) return;
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  String url = "http://10.68.18.6:5001/api/speed";
  http.begin(url);
  http.setTimeout(3000);
  int code = http.GET();

  if (code == 200) {
    String payload = http.getString();
    Serial.print("Speed payload: "); Serial.println(payload);
    int idx = payload.indexOf(":");
    if (idx != -1) {
      int val = payload.substring(idx + 1).toInt();
      val = constrain(val, 0, 255);
      ledcWrite(PWM_CHANNEL, val);
      motorSpeed = val;
      Serial.print("Speed set to: "); Serial.println(val);
    }
  } else {
    Serial.print("Speed check failed: "); Serial.println(code);
  }
  http.end();
  lastSpeedCheck = millis();
}

void readVibrationAI() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  int16_t ax = Wire.read() << 8 | Wire.read();
  int16_t ay = Wire.read() << 8 | Wire.read();
  int16_t az = Wire.read() << 8 | Wire.read();

  float axg = ax / 16384.0;
  float ayg = ay / 16384.0;
  float azg = az / 16384.0 - 1.0;
  float acc = sqrt(axg*axg + ayg*ayg + azg*azg);

  vibrationBuffer[bufferIndex] = acc;
  bufferIndex = (bufferIndex + 1) % WINDOW_SIZE;
  currentRMS = 0.8 * currentRMS + 0.2 * acc;
}

void calculateAIFeatures() {
  float mean = 0, m2 = 0, m4 = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    mean += vibrationBuffer[i];
    m2   += vibrationBuffer[i] * vibrationBuffer[i];
    m4   += pow(vibrationBuffer[i], 4);
  }
  mean /= WINDOW_SIZE;
  m2   /= WINDOW_SIZE;
  m4   /= WINDOW_SIZE;
  currentKurtosis = (m4 / pow(m2, 2)) - 3.0;
}

void calculateHealthScore() {
  float deviation = (currentRMS - baselineRMS) / baselineRMS * 100.0;
  healthScore = 100.0 - deviation;
  healthScore = constrain(healthScore, 0, 100);
}

void updateLCD() {
  lcd.setCursor(0, 0);
  lcd.print("R:"); lcd.print(currentRMS, 2);
  lcd.setCursor(8, 0);
  lcd.print("K:"); lcd.print(currentKurtosis, 1);
  lcd.setCursor(0, 1);
  lcd.print("H:"); lcd.print(healthScore, 0);
  lcd.print("% ");
  if      (healthScore >= 80) lcd.print("EXCLLNT ");
  else if (healthScore >= 60) lcd.print("GOOD    ");
  else if (healthScore >= 40) lcd.print("FAIR    ");
  else if (healthScore >= 20) lcd.print("POOR    ");
  else                        lcd.print("CRITCAL!");
}

void updateLEDs() {
  digitalWrite(GREEN_LED,  LOW);
  digitalWrite(YELLOW_LED, LOW);
  digitalWrite(RED_LED,    LOW);

  if (healthScore >= 60) {
    digitalWrite(GREEN_LED, HIGH);
  } else if (healthScore >= 30) {
    digitalWrite(YELLOW_LED, HIGH);
    if (millis() - lastBuzz > 5000) {
      tone(BUZZER, 1000, 200);
      lastBuzz = millis();
    }
  } else {
    digitalWrite(RED_LED, HIGH);
    if (millis() - lastBuzz > 1000) {
      tone(BUZZER, 2000, 500);
      lastBuzz = millis();
    }
  }
}

void calibrateAI() {
  for (int i = 0; i < 200; i++) {
    readVibrationAI();
    delay(20);
  }
  float sum = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) sum += vibrationBuffer[i];
  baselineRMS = sum / WINDOW_SIZE;

  float mean = 0, m2 = 0, m4 = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    mean += vibrationBuffer[i];
    m2   += vibrationBuffer[i] * vibrationBuffer[i];
    m4   += pow(vibrationBuffer[i], 4);
  }
  mean /= WINDOW_SIZE; m2 /= WINDOW_SIZE; m4 /= WINDOW_SIZE;
  baselineKurtosis = (m4 / pow(m2, 2)) - 3.0;

  Serial.print("Baseline RMS: ");      Serial.println(baselineRMS);
  Serial.print("Baseline Kurtosis: "); Serial.println(baselineKurtosis);
}

void sendToThingSpeak(float rms, float kurtosis, float health) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = "http://api.thingspeak.com/update?api_key=" + tsApiKey +
               "&field1=" + String(rms) +
               "&field2=" + String(kurtosis) +
               "&field3=" + String(health);
  Serial.println("Sending to ThingSpeak...");
  http.begin(url);
  int code = http.GET();
  Serial.print("ThingSpeak: "); Serial.println(code);
  http.end();
}

void sendToFlask(float rms, float kurtosis, float health) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  http.begin(flaskURL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);

  String status;
  if      (health >= 80) status = "EXCELLENT";
  else if (health >= 60) status = "GOOD";
  else if (health >= 40) status = "FAIR";
  else if (health >= 20) status = "POOR";
  else                   status = "CRITICAL";

  String body = "{\"rms\":"      + String(rms, 3) +
                ",\"kurtosis\":" + String(kurtosis, 3) +
                ",\"health\":"   + String(health, 1) +
                ",\"speed\":"    + String(motorSpeed) +
                ",\"status\":\"" + status + "\"}";

  Serial.print("Sending: "); Serial.println(body);
  int code = http.POST(body);
  Serial.print("Flask: "); Serial.println(code);
  if (code == 200) Serial.println(http.getString());
  http.end();
}
