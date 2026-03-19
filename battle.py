import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
print("Loading 57k rows of data...")
df = pd.read_csv('OrderFlowData.csv', sep=';')

# 2. CREATE INDICATORS
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# RSI 14
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Order Flow Features
df['TotalVol'] = df['BuyVol'] + df['SellVol']
df['Delta_Percent'] = df['Delta'] / df['TotalVol']
df['RVol'] = df['TotalVol'] / df['TotalVol'].rolling(window=20).mean()

# CVD Trend
if 'CVD' in df.columns:
    df['CVD_Trend'] = df['CVD'] - df['CVD'].rolling(window=20).mean()
else:
    df['CVD_Trend'] = 0

df = df.dropna()

# SETTINGS
tick_size = 0.25
target_ticks = 4
stop_ticks = 6
target_dist = tick_size * target_ticks
stop_dist = tick_size * stop_ticks

# Future Lookups
df['Future_High'] = df['High'].shift(-1)
df['Future_Low'] = df['Low'].shift(-1)

# ==============================================================================
# STRATEGY 1: STANDARD TREND PULLBACK (LONG)
# Logic: Price > SMA (Uptrend) + Red Candle. We BUY.
# ==============================================================================
print("\n--- STRATEGY 1: STANDARD TREND PULLBACK (Long) ---")
s1_data = df[(df['Close'] > df['SMA_50']) & (df['Close'] < df['Open'])].copy()

# Win = Hit Target UP, didn't hit Stop DOWN
hit_stop = s1_data['Future_Low'] <= (s1_data['Close'] - stop_dist)
hit_target = s1_data['Future_High'] >= (s1_data['Close'] + target_dist)
s1_data['Target'] = np.where((hit_target) & (~hit_stop), 1, 0)

if len(s1_data) > 100:
    X = s1_data[['Delta_Percent', 'RVol', 'RSI', 'CVD_Trend']]
    y = s1_data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results = pd.DataFrame({'Actual': y_test, 'Signal': preds})
    buys = results[results['Signal'] == 1]
    
    if len(buys) > 0:
        win_rate = len(buys[buys['Actual'] == 1]) / len(buys)
        print(f"Trades Taken: {len(buys)}")
        print(f"Win Rate: {win_rate*100:.2f}%")
    else:
        print("AI refused to trade.")
else:
    print("Not enough data.")

# ==============================================================================
# STRATEGY 2: FAILED TREND (SHORT)
# Logic: Price > SMA (Uptrend) + Red Candle. We SHORT (Betting the trend is breaking).
# ==============================================================================
print("\n--- STRATEGY 2: FAILED TREND (Short) ---")
# Same entry filter, inverted target
s2_data = s1_data.copy() 

# Win = Hit Target DOWN, didn't hit Stop UP
hit_stop_short = s2_data['Future_High'] >= (s2_data['Close'] + stop_dist)
hit_target_short = s2_data['Future_Low'] <= (s2_data['Close'] - target_dist)
s2_data['Target'] = np.where((hit_target_short) & (~hit_stop_short), 1, 0)

X = s2_data[['Delta_Percent', 'RVol', 'RSI', 'CVD_Trend']]
y = s2_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model.fit(X_train, y_train)
preds = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Signal': preds})
shorts = results[results['Signal'] == 1]

if len(shorts) > 0:
    win_rate = len(shorts[shorts['Actual'] == 1]) / len(shorts)
    print(f"Trades Taken: {len(shorts)}")
    print(f"Win Rate: {win_rate*100:.2f}%")
else:
    print("AI refused to trade.")

# ==============================================================================
# STRATEGY 3: MEAN REVERSION (SHORT)
# Logic: RSI > 70. We SHORT.
# ==============================================================================
print("\n--- STRATEGY 3: RSI MEAN REVERSION (Short) ---")
s3_data = df[df['RSI'] > 70].copy()

hit_stop_short = s3_data['Future_High'] >= (s3_data['Close'] + stop_dist)
hit_target_short = s3_data['Future_Low'] <= (s3_data['Close'] - target_dist)
s3_data['Target'] = np.where((hit_target_short) & (~hit_stop_short), 1, 0)

if len(s3_data) > 100:
    X = s3_data[['Delta_Percent', 'RVol', 'RSI', 'CVD_Trend']]
    y = s3_data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Signal': preds})
    shorts = results[results['Signal'] == 1]
    
    if len(shorts) > 0:
        win_rate = len(shorts[shorts['Actual'] == 1]) / len(shorts)
        print(f"Trades Taken: {len(shorts)}")
        print(f"Win Rate: {win_rate*100:.2f}%")
    else:
        print("AI refused to trade.")
else:
    print("Not enough Overbought conditions.")