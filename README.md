# Vibrautomation 🔧
IoT-based Motor Health Analyser

## Overview
Real-time motor fault detection system using vibration analysis,
ML classification and live web dashboard.

## Tech Stack
- Hardware: ESP32 + MPU6050 (accelerometer)
- Signal Processing: RMS, Kurtosis-based health scoring
- ML Model: SVM (cross-validated, highest accuracy)
- Backend: Flask + SocketIO + SQLite
- Frontend: Real-time web dashboard

## Architecture
ESP32 → MPU6050 → I2C → WiFi → Flask API → SVM Model → Dashboard

## Results
- Fault detection accuracy: >95%
- Real-time monitoring with <1s latency
- Classifies: Normal / Bearing Fault / Imbalance
