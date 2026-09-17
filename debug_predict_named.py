import json
from app import FraudDetectionSystem
from fastapi.testclient import TestClient
import api

fs = FraudDetectionSystem()
print('load_from_artifacts', fs.load_from_artifacts(model_dir='models'))
print('models', list(fs.models.keys()))
print('feature_names count', len(fs.feature_names))

api.fraud_system = fs
client = TestClient(api.app)
payload = {
    'step':1,
    'time_hour':12,
    'amount':200,
    'oldbalanceOrg':1000,
    'newbalanceOrig':800,
    'oldbalanceDest':0,
    'newbalanceDest':0,
    'orig_amount_mean':120,
    'orig_amount_std':20,
    'time_since_prev':5,
    'tx_count_24h':1,
    'amount_z':0.5,
    'is_unusual_amount':0,
    'merchant_risk':0.2,
    'account_age_days':365,
    'auth_failures':0,
    'type_PAYMENT':1,
    'type_CASH_OUT':0,
    'type_DEBIT':0,
    'type_TRANSFER':0,
    'merchant_cat_grocery':0,
    'merchant_cat_restaurant':0,
    'merchant_cat_retail':0,
    'merchant_cat_travel':0,
    'merchant_cat_utilities':0,
    'country_CN':0,
    'country_DE':0,
    'country_FR':0,
    'country_GB':0,
    'country_IN':0,
    'country_JP':0,
    'country_NG':0,
    'country_RU':0,
    'country_US':1,
}
print('payload keys', list(payload.keys())[:5], '...', len(payload))
resp = client.post('/predict_named', json=payload)
print('status', resp.status_code)
print(resp.text)
