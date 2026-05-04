## Setup & Run

### Hardware
- ESP32 + MPU6050 connected via I2C (SDA=21, SCL=22)
- Flash `firmware/review1_code.ino` using Arduino IDE

### Backend
```bash
pip install flask scikit-learn pandas numpy
python backend/app_with_ml.py
```

### Access Dashboard
Open browser → `http://localhost:5001`
