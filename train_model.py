"""
Vibrautomation - ML Model Training
Trains Random Forest, SVM, and Neural Network
Picks the best model and saves it for Flask integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# 1. GENERATE SIMULATED TRAINING DATA
# ─────────────────────────────────────────────
np.random.seed(42)

def generate_data(n=500):
    rows = []
    for _ in range(n):
        label = np.random.choice(['HEALTHY', 'WARNING', 'FAULT'],
                                  p=[0.5, 0.3, 0.2])
        if label == 'HEALTHY':
            rms      = np.random.normal(1.05, 0.08)
            kurtosis = np.random.normal(3.2,  0.4)
            speed    = np.random.randint(60, 80)
        elif label == 'WARNING':
            rms      = np.random.normal(1.55, 0.12)
            kurtosis = np.random.normal(5.8,  0.6)
            speed    = np.random.randint(78, 90)
        else:
            rms      = np.random.normal(2.15, 0.20)
            kurtosis = np.random.normal(9.4,  1.0)
            speed    = np.random.randint(85, 100)

        # Extra engineered features
        rms_kurt_ratio = rms / (kurtosis + 0.001)
        energy         = rms ** 2
        rows.append([rms, kurtosis, speed, rms_kurt_ratio, energy, label])

    df = pd.DataFrame(rows, columns=['rms','kurtosis','speed',
                                      'rms_kurt_ratio','energy','label'])
    return df

print("=" * 55)
print("  VIBRAUTOMATION — ML MODEL TRAINER")
print("=" * 55)

print("\n[1/6] Generating training data...")
df = generate_data(1000)
df.to_csv('training_data.csv', index=False)
print(f"      {len(df)} samples generated")
print(df['label'].value_counts().to_string())

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────────
print("\n[2/6] Preparing features...")
features = ['rms', 'kurtosis', 'speed', 'rms_kurt_ratio', 'energy']
X = df[features].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. TRAIN ALL 3 MODELS
# ─────────────────────────────────────────────
print("\n[3/6] Training models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42),
    'SVM': SVC(
        kernel='rbf', C=10, gamma='scale',
        probability=True, random_state=42),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500,
        random_state=42, early_stopping=False)
}

results = {}
for name, model in models.items():
    # Use scaled data for SVM and NN, raw for RF
    Xtr = X_train_s if name != 'Random Forest' else X_train
    Xte = X_test_s  if name != 'Random Forest' else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, Xtr, y_train, cv=5).mean()

    results[name] = {
        'model':  model,
        'acc':    acc,
        'cv':     cv,
        'y_pred': y_pred,
        'Xte':    Xte
    }
    print(f"      {name:<18} Accuracy: {acc*100:.1f}%  CV: {cv*100:.1f}%")

# ─────────────────────────────────────────────
# 4. PICK BEST MODEL
# ─────────────────────────────────────────────
print("\n[4/6] Selecting best model...")
best_name = max(results, key=lambda k: results[k]['cv'])
best      = results[best_name]
print(f"      Winner: {best_name} (CV accuracy: {best['cv']*100:.1f}%)")

# ─────────────────────────────────────────────
# 5. SAVE MODEL + SCALER
# ─────────────────────────────────────────────
print("\n[5/6] Saving model...")
joblib.dump(best['model'], 'best_model.pkl')
joblib.dump(scaler,        'scaler.pkl')
joblib.dump(features,      'features.pkl')
joblib.dump(best_name,     'model_name.pkl')
print(f"      Saved: best_model.pkl ({best_name})")
print(f"      Saved: scaler.pkl")

# ─────────────────────────────────────────────
# 6. GENERATE REPORT + CHARTS
# ─────────────────────────────────────────────
print("\n[6/6] Generating report and charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Vibrautomation — ML Training Report', fontsize=14, fontweight='bold')

# Chart 1: Model comparison bar chart
names  = list(results.keys())
accs   = [results[n]['acc']*100 for n in names]
cvs    = [results[n]['cv']*100  for n in names]
x      = np.arange(len(names))
w      = 0.35
ax     = axes[0, 0]
bars1  = ax.bar(x - w/2, accs, w, label='Test Accuracy', color='#2563eb', alpha=0.85)
bars2  = ax.bar(x + w/2, cvs,  w, label='CV Accuracy',   color='#16a34a', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(80, 102); ax.set_ylabel('Accuracy (%)')
ax.set_title('Model comparison'); ax.legend(fontsize=8)
for b in bars1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                         f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
for b in bars2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                         f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

# Chart 2: Confusion matrix of best model
cm     = confusion_matrix(y_test, best['y_pred'],
                           labels=['HEALTHY','WARNING','FAULT'])
ax2    = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['HEALTHY','WARNING','FAULT'],
            yticklabels=['HEALTHY','WARNING','FAULT'], ax=ax2)
ax2.set_title(f'Confusion matrix — {best_name}')
ax2.set_ylabel('Actual'); ax2.set_xlabel('Predicted')

# Chart 3: RMS distribution by class
ax3 = axes[1, 0]
colors = {'HEALTHY':'#16a34a','WARNING':'#ca8a04','FAULT':'#dc2626'}
for label, color in colors.items():
    subset = df[df['label']==label]['rms']
    ax3.hist(subset, bins=30, alpha=0.6, label=label, color=color)
ax3.axvline(1.3, color='orange', linestyle='--', linewidth=1.5, label='Warning threshold')
ax3.axvline(1.8, color='red',    linestyle='--', linewidth=1.5, label='Fault threshold')
ax3.set_xlabel('RMS (g)'); ax3.set_ylabel('Count')
ax3.set_title('RMS distribution by class'); ax3.legend(fontsize=8)

# Chart 4: Feature importance (RF only, else kurtosis scatter)
ax4 = axes[1, 1]
if best_name == 'Random Forest':
    imp   = best['model'].feature_importances_
    fidx  = np.argsort(imp)[::-1]
    ax4.bar(range(len(features)), imp[fidx], color='#7c3aed', alpha=0.85)
    ax4.set_xticks(range(len(features)))
    ax4.set_xticklabels([features[i] for i in fidx], rotation=20, fontsize=8)
    ax4.set_title('Feature importance (Random Forest)')
    ax4.set_ylabel('Importance')
else:
    for label, color in colors.items():
        sub = df[df['label']==label]
        ax4.scatter(sub['rms'], sub['kurtosis'], alpha=0.4,
                    label=label, color=color, s=15)
    ax4.set_xlabel('RMS (g)'); ax4.set_ylabel('Kurtosis')
    ax4.set_title('RMS vs Kurtosis by class'); ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig('training_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: training_report.png")

# Print final classification report
print("\n" + "=" * 55)
print(f"  BEST MODEL: {best_name}")
print(f"  TEST ACCURACY: {best['acc']*100:.1f}%")
print(f"  CV ACCURACY:   {best['cv']*100:.1f}%")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['HEALTHY','WARNING','FAULT']))
print("Files saved:")
print("  best_model.pkl  — trained model")
print("  scaler.pkl      — feature scaler")
print("  features.pkl    — feature list")
print("  training_data.csv")
print("  training_report.png")
print("\nDone! Run predict.py to test predictions.")
