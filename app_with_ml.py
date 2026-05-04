from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from datetime import datetime
import sqlite3, joblib, numpy as np, os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vibrautomation2024'
socketio = SocketIO(app, cors_allowed_origins="*")
DB_PATH  = 'vibrautomation.db'

# ── SPEED CONTROL ─────────────────────────────
current_speed = 150  # default speed 0-255

# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, rms REAL, kurtosis REAL,
        speed INTEGER, status TEXT, ml_prediction TEXT, confidence REAL)''')
    conn.commit(); conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_ml():
    try:
        return (joblib.load('best_model.pkl'),
                joblib.load('scaler.pkl'),
                joblib.load('model_name.pkl'))
    except:
        return None, None, None

def ml_predict(rms, kurtosis, speed):
    model, scaler, name = load_ml()
    if model is None:
        return None, None
    rms_kurt_ratio = rms / (kurtosis + 0.001)
    energy = rms ** 2
    X = np.array([[rms, kurtosis, speed, rms_kurt_ratio, energy]])
    if name != 'Random Forest':
        X = scaler.transform(X)
    pred  = model.predict(X)[0]
    conf  = round(float(max(model.predict_proba(X)[0])) * 100, 1)
    return pred, conf

# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── RECEIVE DATA FROM ESP32 ───────────────────
@app.route('/api/data', methods=['POST'])
def receive_data():
    data     = request.get_json()
    rms      = float(data.get('rms', 0))
    kurtosis = float(data.get('kurtosis', 0))
    speed    = int(data.get('speed', 0))
    ts       = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Health-based status
    if rms < 1.3:   status = 'HEALTHY'
    elif rms < 1.8: status = 'WARNING'
    else:           status = 'FAULT'

    # ML prediction
    ml_pred, confidence = ml_predict(rms, kurtosis, speed)

    conn = get_db()
    conn.execute('INSERT INTO readings VALUES (NULL,?,?,?,?,?,?,?)',
                 (ts, rms, kurtosis, speed, status, ml_pred, confidence))
    conn.commit(); conn.close()

    payload = {'timestamp': ts, 'rms': rms, 'kurtosis': kurtosis,
               'speed': speed, 'status': status,
               'ml_prediction': ml_pred, 'confidence': confidence}
    socketio.emit('new_reading', payload)
    return jsonify({'ok': True, 'status': status, 'ml': ml_pred, 'confidence': confidence})

# ── SPEED CONTROL ─────────────────────────────
@app.route('/api/speed', methods=['POST', 'GET'])
def speed():
    global current_speed
    if request.method == 'POST':
        data = request.get_json()
        current_speed = int(data.get('speed', 150))
        current_speed = max(0, min(255, current_speed))
        socketio.emit('speed_update', {'speed': current_speed})
        print(f"Speed updated: {current_speed}")
        return jsonify({'ok': True, 'speed': current_speed})
    else:
        return jsonify({'speed': current_speed})

# ── ML PREDICT ────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    data     = request.get_json()
    rms      = float(data.get('rms', 0))
    kurtosis = float(data.get('kurtosis', 0))
    speed    = int(data.get('speed', 0))
    pred, conf = ml_predict(rms, kurtosis, speed)
    if pred is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first'}), 500
    return jsonify({'prediction': pred, 'confidence': conf})

# ── HISTORY ───────────────────────────────────
@app.route('/api/history')
def history():
    conn = get_db()
    rows = conn.execute('SELECT * FROM readings ORDER BY id DESC LIMIT 50').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ── CHART DATA ────────────────────────────────
@app.route('/api/chart')
def chart_data():
    conn = get_db()
    rows = conn.execute('SELECT timestamp, rms FROM readings ORDER BY id DESC LIMIT 30').fetchall()
    conn.close()
    rows = list(reversed(rows))
    return jsonify({'labels': [r['timestamp'][-8:] for r in rows],
                    'values': [r['rms'] for r in rows]})

# ── STATS ─────────────────────────────────────
@app.route('/api/stats')
def stats():
    conn = get_db()
    row = conn.execute('''SELECT COUNT(*) as total, AVG(rms) as avg_rms,
        MAX(rms) as max_rms,
        SUM(CASE WHEN status="FAULT" THEN 1 ELSE 0 END) as faults,
        SUM(CASE WHEN status="WARNING" THEN 1 ELSE 0 END) as warnings
        FROM readings''').fetchone()
    conn.close()
    return jsonify(dict(row))

# ─────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    model, _, name = load_ml()
    if model:
        print(f"ML model loaded: {name}")
    else:
        print("No ML model found — run train_model.py first")
    print("Server running at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)