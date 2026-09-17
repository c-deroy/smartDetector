import pandas as pd
from feature_engineer import engineer_chunk
import os

INPUT = os.path.join(os.path.dirname(__file__), '..', 'paysim.csv')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'paysim_engineered_sample.csv')

chunk = pd.read_csv(INPUT, nrows=1000)
eng = engineer_chunk(chunk)
eng.to_csv(OUTPUT, index=False)
print('Wrote sample to', OUTPUT)
