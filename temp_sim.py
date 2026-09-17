import numpy as np
import pandas as pd
import random
from app import FraudDetectionSystem

df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
model = FraudDetectionSystem()
model.fit(X, y)
print('trained')
for i in range(10):
    amt = round(random.uniform(2000, 5000), 2)
    ts = random.uniform(0, 172792)
    features = np.zeros((1, 30), dtype=float)
    features[0, 0] = ts
    features[0, -1] = amt
    scores = model.get_real_time_score(features)
    fraud_probability = scores.get('soft_voting', 0.5)
    print(i+1, amt, fraud_probability, scores)
