import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
print("Loading data...")
df = pd.read_csv('OrderFlowData.csv', sep=';')

# 2. FEATURE ENGINEERING (Standard)
df['TotalVol'] = df['BuyVol'] + df['SellVol']
df['Delta_Percent'] = df['Delta'] / df['TotalVol']
df['Vol_SMA_20'] = df['TotalVol'].rolling(window=20).mean()
df['RVol'] = df['TotalVol'] / df['Vol_SMA_20']
df['Price_ROC'] = df['Close'].pct_change()
df['Volatility'] = df['Close'].rolling(window=20).std()

# CVD Trend
if 'CVD' in df.columns:
    df['CVD_Trend'] = df['CVD'] - df['CVD'].rolling(window=20).mean()
else:
    df['CVD_Trend'] = 0

df = df.dropna()

# 3. DEFINE "SWING" GRID (The Big Numbers)
# We are testing 2.5 points up to 10 points
tick_size = 0.25
targets = [10, 20, 30, 40] 
stops =   [10, 20, 30, 40]

features = ['Delta_Percent', 'RVol', 'Price_ROC', 'Volatility', 'CVD_Trend']
X_base = df[features]

print(f"{'Target':<6} | {'Stop':<5} | {'Win Rate':<8} | {'Trades':<6} | {'Expectancy'}")
print("-" * 60)

# 4. TEST LOOP
for t in targets:
    for s in stops:
        
        target_dist = tick_size * t
        stop_dist = tick_size * s
        
        # We need to look further into the future for big moves
        # Shift -20 bars (approx 10 mins on 30s chart) to give trade time to play out
        future_high = df['High'].rolling(window=20).max().shift(-20)
        future_low = df['Low'].rolling(window=20).min().shift(-20)
        entry = df['Close']
        
        # Long Logic
        hit_stop = future_low <= (entry - stop_dist)
        hit_target = future_high >= (entry + target_dist)
        
        # Win = Hit Target AND did NOT hit Stop
        y = np.where((hit_target) & (~hit_stop), 1, 0)
        
        # Skip if target is impossible (never happened in dataset)
        if sum(y) < 10: continue

        # Train Model
        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        results = pd.DataFrame({'Actual': y_test, 'Signal': preds})
        trades = results[results['Signal'] == 1]
        
        if len(trades) > 5:
            wins = trades[trades['Actual'] == 1]
            win_rate = len(wins) / len(trades)
            
            # Expectancy
            ev = (win_rate * t) - ((1 - win_rate) * s)
            
            # Highlight huge winners
            prefix = ">>> " if ev > 2.0 else "    "
            print(f"{prefix}{t:<6} | {s:<5} | {win_rate*100:.1f}%    | {len(trades):<6} | {ev:.2f}")

print("-" * 60)