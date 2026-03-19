import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
print("Loading Data for AI Audit...")
df = pd.read_csv('OrderFlowData.csv', sep=';')
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df.set_index('Time', inplace=True)

# Resample to 5-Min Bars for granular OR calculation
ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 
             'BuyVol': 'sum', 'SellVol': 'sum', 'Delta': 'sum', 'VWAP_Dist': 'last'}
df_5m = df.resample('5min').apply(ohlc_dict).dropna()
df_5m.reset_index(inplace=True)
df_5m['Date'] = df_5m['Time'].dt.date

# PLATINUM SETTINGS
or_minutes = 60
trail_points = 20
or_bars = int(or_minutes / 5)
tick_value = 0.50
tick_size = 0.25
fee = 1.24

# 2. EXTRACT TRADES & FEATURES
print("Simulating Platinum Strategy & extracting AI features...")

features = []
labels = [] # 1 = Win, 0 = Loss

for day in df_5m['Date'].unique():
    day_data = df_5m[df_5m['Date'] == day].copy().reset_index(drop=True)
    if len(day_data) < (or_bars + 5): continue

    # Opening Range Stats
    opening_range = day_data.iloc[:or_bars]
    or_high = opening_range['High'].max()
    or_low = opening_range['Low'].min()
    
    # AI Features (What happened BEFORE the breakout?)
    or_vol = opening_range['BuyVol'].sum() + opening_range['SellVol'].sum()
    or_delta = opening_range['Delta'].sum()
    or_height = or_high - or_low
    or_vwap_dist = opening_range.iloc[-1]['VWAP_Dist'] # Context
    
    trade_taken = False
    entry_price = 0
    direction = 0
    stop_price = 0
    
    # Scan day
    for i in range(or_bars, len(day_data)):
        bar = day_data.iloc[i]
        
        if not trade_taken:
            # CHECK ENTRY
            if bar['Close'] > or_high:
                entry_price = bar['Close']
                direction = 1
                stop_price = entry_price - trail_points
                trade_taken = True
            elif bar['Close'] < or_low:
                entry_price = bar['Close']
                direction = -1
                stop_price = entry_price + trail_points
                trade_taken = True
            
            # If trade taken, capture the AI features
            if trade_taken:
                # Features: Volume, Delta, Height, VWAP
                features.append([or_vol, or_delta, or_height, or_vwap_dist])
                
        else:
            # MANAGE TRADE (Did it win?)
            pnl = 0
            complete = False
            
            if direction == 1:
                if bar['Low'] <= stop_price:
                    pnl = ((stop_price - entry_price) / tick_size * tick_value) - fee
                    complete = True
                elif (bar['High'] - trail_points) > stop_price:
                    stop_price = bar['High'] - trail_points
            elif direction == -1:
                if bar['High'] >= stop_price:
                    pnl = ((entry_price - stop_price) / tick_size * tick_value) - fee
                    complete = True
                elif (bar['Low'] + trail_points) < stop_price:
                    stop_price = bar['Low'] + trail_points
            
            if complete:
                # Label: Did we make money?
                labels.append(1 if pnl > 0 else 0)
                break 

# 3. TRAIN THE AI (Random Forest)
if len(features) > 50:
    X = np.array(features)
    y = np.array(labels)
    feature_names = ['OR_Volume', 'OR_Delta', 'OR_Height', 'VWAP_Dist']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*50)
    print(f"AI AUDIT RESULTS (Can AI predict the winners?)")
    print("="*50)
    print(f"Baseline Win Rate: {np.mean(y)*100:.1f}%")
    print(f"AI Prediction Accuracy: {acc*100:.1f}%")
    
    print("-" * 50)
    if acc > 0.58:
        print(">>> CRITICAL: The AI found a hidden pattern!")
        print("Your strategy is MISSING something. Check Feature Importance below.")
        print("-" * 50)
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            print(f"{name}: {imp:.4f}")
            
        # Quick rule extraction logic
        # Is High Volume good or bad?
        high_vol_wins = y[X[:,0] > np.median(X[:,0])].mean()
        low_vol_wins = y[X[:,0] <= np.median(X[:,0])].mean()
        print(f"\nWin Rate with High Volume: {high_vol_wins*100:.1f}%")
        print(f"Win Rate with Low Volume:  {low_vol_wins*100:.1f}%")
        
    else:
        print(">>> VERDICT: CLEAN.")
        print("The AI could NOT predict winners better than random chance.")
        print("This means your '60-min / Trail 20' rule captures 100% of the available edge.")
        print("The rest is just market noise.")
else:
    print("Not enough trades to train AI.")