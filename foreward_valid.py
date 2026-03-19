import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

# 1. LOAD DATA
print("Loading data for Walk-Forward Analysis...")
df = pd.read_csv('OrderFlowData.csv', sep=';', dtype={
    'DurationSec': 'float32',
    'VWAP_Dist': 'float32', 
    'Delta': 'float32',
    'CVD': 'float32'
})
df = df.dropna()

# 2. FEATURE ENGINEERING (Same as before)
df['TotalVol'] = df['BuyVol'] + df['SellVol']
df['Delta_Percent'] = df['Delta'] / (df['TotalVol'] + 1)
df['Is_HFT_Speed'] = (df['DurationSec'] < 0.3).astype(int) # The "Secret Sauce" threshold
df['Abs_VWAP_Dist'] = df['VWAP_Dist'].abs()
df['Is_Breakout'] = np.where((df['SessionHigh_Dist'] < 0) | (df['SessionLow_Dist'] < 0), 1, 0)

# Target: Trend Continuation
df['Bar_Dir'] = np.where(df['Close'] > df['Open'], 1, -1)
df['Next_Dir'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
df['Target'] = np.where(df['Bar_Dir'] == df['Next_Dir'], 1, 0)
df = df.dropna()

# 3. DEFINE FEATURES
features = ['Delta_Percent', 'DurationSec', 'VWAP_Dist', 'SessionHigh_Dist', 'CVD']
X = df[features]
y = df['Target']

# 4. RUN WALK-FORWARD VALIDATION
# We split data into 5 segments.
# Iteration 1: Train on first 20%, Test on next 20%
# Iteration 2: Train on first 40%, Test on next 20%
# ... and so on.
tscv = TimeSeriesSplit(n_splits=5)

print("\n--- WALK-FORWARD RESULTS ---")
print(f"{'Split':<6} | {'Train Rows':<12} | {'Test Rows':<12} | {'Win Rate (New Data)'}")
print("-" * 65)

fold = 1
win_rates = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the Model (HistGradientBoosting is fast)
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on NEW data only
    # We apply the specific filter we found: HFT Speed + Breakout
    test_df = X_test.copy()
    test_df['Actual'] = y_test
    test_df['Predicted'] = model.predict(X_test)
    
    # Filter for our Specific Strategy Condition (Duration < 0.3s)
    # The AI might have general accuracy, but we only care about the HFT Setup
    hft_trades = test_df[test_df['DurationSec'] < 0.3]
    
    if len(hft_trades) > 0:
        # Calculate Win Rate on this specific "Future" chunk
        win_rate = hft_trades['Actual'].mean()
        win_rates.append(win_rate)
        
        print(f"#{fold:<5} | {len(X_train):<12} | {len(X_test):<12} | {win_rate*100:.2f}%")
    else:
        print(f"#{fold:<5} | {len(X_train):<12} | {len(X_test):<12} | No Trades Found")
    
    fold += 1

print("-" * 65)
avg_wr = np.mean(win_rates)
print(f"AVERAGE WALK-FORWARD WIN RATE: {avg_wr*100:.2f}%")

if avg_wr > 0.55:
    print(">>> VERDICT: ROBUST. The strategy holds up on unseen future data.")
else:
    print(">>> VERDICT: FRAGILE. The strategy degrades over time.")