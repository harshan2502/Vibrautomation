"""
Vibrautomation - Predict fault from sensor values
Usage: python predict.py
Or import predict_fault() into Flask app
"""

import joblib
import numpy as np

def load_model():
    model      = joblib.load('best_model.pkl')
    scaler     = joblib.load('scaler.pkl')
    features   = joblib.load('features.pkl')
    model_name = joblib.load('model_name.pkl')
    return model, scaler, features, model_name

def predict_fault(rms, kurtosis, speed):
    model, scaler, features, model_name = load_model()

    # Engineer same features as training
    rms_kurt_ratio = rms / (kurtosis + 0.001)
    energy         = rms ** 2

    X = np.array([[rms, kurtosis, speed, rms_kurt_ratio, energy]])

    # Scale if not Random Forest
    if model_name != 'Random Forest':
        X = scaler.transform(X)

    prediction   = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes      = model.classes_

    prob_dict = {c: round(float(p)*100, 1) for c, p in zip(classes, probabilities)}
    confidence = round(float(max(probabilities)) * 100, 1)

    return {
        'prediction':  prediction,
        'confidence':  confidence,
        'model_used':  model_name,
        'probabilities': prob_dict
    }

if __name__ == '__main__':
    print("=" * 45)
    print("  VIBRAUTOMATION — FAULT PREDICTOR")
    print("=" * 45)

    test_cases = [
        {'rms': 1.05, 'kurtosis': 3.1, 'speed': 70,  'expected': 'HEALTHY'},
        {'rms': 1.60, 'kurtosis': 5.9, 'speed': 85,  'expected': 'WARNING'},
        {'rms': 2.20, 'kurtosis': 9.8, 'speed': 92,  'expected': 'FAULT'},
        {'rms': 1.28, 'kurtosis': 4.1, 'speed': 75,  'expected': 'HEALTHY (borderline)'},
        {'rms': 1.85, 'kurtosis': 7.2, 'speed': 88,  'expected': 'FAULT (borderline)'},
    ]

    print(f"\n{'RMS':>6} {'Kurt':>6} {'Spd':>5} | {'Predicted':<10} {'Conf':>6} | Expected")
    print("-" * 60)

    for tc in test_cases:
        result = predict_fault(tc['rms'], tc['kurtosis'], tc['speed'])
        pred   = result['prediction']
        conf   = result['confidence']
        print(f"{tc['rms']:>6.2f} {tc['kurtosis']:>6.1f} {tc['speed']:>5}% | "
              f"{pred:<10} {conf:>5.1f}% | {tc['expected']}")

    print(f"\nModel used: {result['model_used']}")
    print("\nSample probability output (last test):")
    for label, prob in result['probabilities'].items():
        bar = '#' * int(prob / 5)
        print(f"  {label:<10} {prob:>5.1f}%  {bar}")
