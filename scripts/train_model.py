import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT = os.path.join(ROOT, 'paysim_engineered.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, 'rf_paysim.joblib')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.joblib')
META_FILE = os.path.join(MODEL_DIR, 'metadata.json')
METRICS_FILE = os.path.join(MODEL_DIR, 'train_metrics.json')

if not os.path.exists(INPUT):
    print('Engineered CSV not found at', INPUT)
    raise SystemExit(1)

print('Loading', INPUT)
df = pd.read_csv(INPUT)
print('Rows:', len(df))

# Drop identifier columns
drop_cols = ['nameOrig','nameDest','device_id','ip_addr']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Target
if 'isFraud' not in df.columns:
    print('No isFraud column in engineered CSV')
    raise SystemExit(1)

y = df['isFraud']

# Categorical features to one-hot
cat_cols = [c for c in ['type','merchant_cat','country','isFlaggedFraud','transaction_channel'] if c in df.columns]
num_cols = [c for c in df.columns if c not in cat_cols + ['isFraud']]
# remove any non-numeric leftovers
num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]

print('Numeric cols count:', len(num_cols), 'Categorical cols:', cat_cols)

# One-hot encode small categorical columns
if cat_cols:
    df_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
else:
    df_cat = pd.DataFrame(index=df.index)

X_num = df[num_cols].fillna(0)

# Scale numeric
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

X = np.hstack([X_num_scaled, df_cat.values])
feature_names = list(X_num.columns) + list(df_cat.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print('Train size', X_train.shape, 'Test size', X_test.shape)

# Train model
clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1] if hasattr(clf, 'predict_proba') else None
report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, probs) if probs is not None else None
cm = confusion_matrix(y_test, y_pred).tolist()
print('AUC:', auc)
print('Classification report:\n', classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(clf, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
meta = {'feature_names': feature_names}
with open(META_FILE, 'w') as f:
    json.dump(meta, f)
metrics = {'auc': auc, 'report': report, 'confusion_matrix': cm}
with open(METRICS_FILE, 'w') as f:
    json.dump(metrics, f)

print('Saved model to', MODEL_FILE)
print('Saved scaler to', SCALER_FILE)
print('Saved metadata to', META_FILE)
print('Saved metrics to', METRICS_FILE)
