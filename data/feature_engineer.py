import pandas as pd
import numpy as np
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COUNTRY_COORDINATES = {
    'US': (39.8283, -98.5795),
    'GB': (55.3781, -3.4360),
    'NG': (9.0820, 8.6753),
    'IN': (20.5937, 78.9629),
    'BR': (-14.2350, -51.9253),
    'DE': (51.1657, 10.4515),
    'FR': (46.2276, 2.2137),
    'CN': (35.8617, 104.1954),
    'JP': (36.2048, 138.2529),
    'RU': (61.5240, 105.3188),
}
HIGH_RISK_COUNTRIES = {'NG', 'RU'}

INPUT = os.path.join(os.path.dirname(__file__), '..', 'paysim.csv')
OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'paysim_engineered.csv')

CHUNK_SIZE = 100000

def synthesize_fields(df):
    # Synthesize location (country code), device id, merchant category, IP
    n = len(df)
    countries = ['US','GB','NG','IN','BR','DE','FR','CN','JP','RU']
    merchant_cats = ['retail','grocery','electronics','travel','utilities','restaurant']
    df['country'] = np.random.choice(countries, size=n, p=None)
    df['location_available'] = 1
    df['location_latitude'] = df['country'].map(lambda country: COUNTRY_COORDINATES[country][0])
    df['location_longitude'] = df['country'].map(lambda country: COUNTRY_COORDINATES[country][1])
    df['high_risk_location_flag'] = df['country'].isin(HIGH_RISK_COUNTRIES).astype(int)
    df['device_id'] = np.random.randint(100000, 200000, size=n).astype(str)
    df['ip_addr'] = ['192.168.%d.%d' % (x%255, (x*3)%255) for x in np.random.randint(1,255,size=n)]
    df['merchant_cat'] = np.random.choice(merchant_cats, size=n)
    # merchant risk score 0-1
    df['merchant_risk'] = np.where(df['merchant_cat'].isin(['electronics','travel']), np.random.beta(2,5,size=n), np.random.beta(1,10,size=n))
    # account age in days (synthesize per customer)
    unique_orig = df['nameOrig'].unique()
    age_map = {k: np.random.randint(30,2000) for k in unique_orig}
    df['account_age_days'] = df['nameOrig'].map(age_map)
    # authentication failures (synth)
    df['auth_failures'] = np.random.poisson(0.02, size=n)
    return df


def engineer_chunk(df):
    # basic mapping from PaySim
    # step: time in hours
    df['time_hour'] = df['step'] % 24

    # amount statistics per origin account (cumulative)
    df = df.sort_values(['nameOrig','step'])
    grp = df.groupby('nameOrig')
    # historical mean/std excluding current txn: use expanding then shift
    df['orig_amount_mean'] = grp['amount'].transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    df['orig_amount_std'] = grp['amount'].transform(lambda x: x.expanding().std().shift(1).fillna(0))

    # time since previous transaction for origin
    df['prev_step'] = grp['step'].shift(1)
    df['time_since_prev'] = df['step'] - df['prev_step']
    df['time_since_prev'] = df['time_since_prev'].fillna(-1)

    # frequency: number of txns in last 24 hours (approx using step hours)
    # For efficiency on chunks, compute simple count over group window of last 24 steps
    df['tx_count_24h'] = grp['step'].transform(lambda s: s.rolling(window=24, min_periods=1).count())

    # amount z-score relative to historical mean/std
    df['amount_z'] = (df['amount'] - df['orig_amount_mean']) / (df['orig_amount_std'].replace(0, np.nan))
    df['amount_z'] = df['amount_z'].replace(np.nan, 0)
    df['is_unusual_amount'] = (df['amount_z'].abs() > 3).astype(int)

    # relation/network patterns: same device/ip used by multiple accounts
    # device_id and ip_addr must exist
    df = synthesize_fields(df)

    # Derived behavioural and transaction-channel signals.
    channel_map = {
        'CASH_OUT': 'ATM',
        'CASH_IN': 'BANK_TRANSFER',
        'TRANSFER': 'BANK_TRANSFER',
        'DEBIT': 'POS',
        'PAYMENT': 'ONLINE',
    }
    df['transaction_channel'] = df['type'].map(channel_map).fillna('OTHER')
    df['device_account_count'] = df.groupby('device_id')['nameOrig'].transform('nunique')
    df['ip_account_count'] = df.groupby('ip_addr')['nameOrig'].transform('nunique')
    df['beneficiary_tx_count'] = df.groupby('nameDest')['nameOrig'].transform('count')
    df['merchant_tx_count'] = df.groupby('nameDest')['nameOrig'].transform('count')
    df['is_new_device'] = (df.groupby('device_id').cumcount() == 0).astype(int)
    df['is_new_ip'] = (df.groupby('ip_addr').cumcount() == 0).astype(int)
    df['is_new_beneficiary'] = (df.groupby('nameDest').cumcount() == 0).astype(int)
    df['is_new_merchant'] = df['is_new_beneficiary']
    df['rapid_withdrawal_count'] = (
        df.assign(_withdrawal=(df['type'] == 'CASH_OUT').astype(int))
        .groupby('nameOrig')['_withdrawal']
        .transform(lambda values: values.rolling(window=3, min_periods=1).sum())
    )
    df['customer_category_deviation'] = (
        df.groupby('nameOrig')['merchant_cat']
        .transform(lambda values: (values != values.shift(1)).astype(int))
    )

    # Synthetic PaySim geography is country-level only. These fields provide
    # the same schema as live GPS data without claiming PaySim has real GPS.
    df['previous_latitude'] = df.groupby('nameOrig')['location_latitude'].shift(1)
    df['previous_longitude'] = df.groupby('nameOrig')['location_longitude'].shift(1)
    latitude_delta = np.radians(df['location_latitude'] - df['previous_latitude'].fillna(df['location_latitude']))
    longitude_delta = np.radians(df['location_longitude'] - df['previous_longitude'].fillna(df['location_longitude']))
    haversine_a = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(np.radians(df['previous_latitude'].fillna(df['location_latitude'])))
        * np.cos(np.radians(df['location_latitude']))
        * np.sin(longitude_delta / 2) ** 2
    )
    df['location_change_distance_km'] = 6371 * 2 * np.arcsin(np.sqrt(haversine_a))
    df['impossible_travel_flag'] = (
        (df['location_change_distance_km'] > 500) & (df['time_since_prev'].between(0, 6))
    ).astype(int)

    return df


def main():
    first = True
    reader = pd.read_csv(INPUT, chunksize=CHUNK_SIZE)
    for chunk in reader:
        eng = engineer_chunk(chunk)
        cols = ['step','time_hour','type','amount','nameOrig','nameDest','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud',
                'orig_amount_mean','orig_amount_std','time_since_prev','tx_count_24h','amount_z','is_unusual_amount','country','device_id','ip_addr','merchant_cat','merchant_risk','account_age_days','auth_failures',
                'transaction_channel','device_account_count','ip_account_count','beneficiary_tx_count','merchant_tx_count','is_new_device','is_new_ip','is_new_beneficiary','is_new_merchant','rapid_withdrawal_count','customer_category_deviation',
                'location_available','location_latitude','location_longitude','high_risk_location_flag','location_change_distance_km','impossible_travel_flag']
        # ensure cols exist
        cols = [c for c in cols if c in eng.columns]
        if first:
            eng.to_csv(OUTPUT, index=False, columns=cols, mode='w')
            first = False
        else:
            eng.to_csv(OUTPUT, index=False, columns=cols, mode='a', header=False)
    print('Engineered data written to', OUTPUT)

if __name__ == '__main__':
    main()
