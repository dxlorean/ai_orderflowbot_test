import pandas as pd
import numpy as np

# 1. LOAD DATA
print("Loading MNQ Data...")
df = pd.read_csv('OrderFlowData.csv', sep=';')
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df.set_index('Time', inplace=True)

# PARAMS
tick_value = 0.50
fee = 1.24
tick_size = 0.25

# 2. DEFINE THE OPTIMIZER FUNCTION
def test_parameters(range_minutes, trail_points):
    # Resample to the specific Opening Range size (e.g., 15min or 30min bars)
    rule = f'{range_minutes}min'
    
    # We need to aggregate properly
    ohlc = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    try:
        df_resampled = df.resample(rule).apply(ohlc).dropna()
        df_resampled.reset_index(inplace=True)
        df_resampled['Date'] = df_resampled['Time'].dt.date
    except:
        return -9999 # Fail safe

    unique_days = df_resampled['Date'].unique()
    total_pnl = 0
    trades = 0

    for day in unique_days:
        day_data = df_resampled[df_resampled['Date'] == day].copy().reset_index(drop=True)
        # We need at least 1 bar for OR, plus trading bars
        if len(day_data) < 5: continue

        # Opening Range is strictly the FIRST BAR of this timeframe
        # e.g. if range_minutes=30, the first bar IS the 30min range.
        or_high = day_data.iloc[0]['High']
        or_low = day_data.iloc[0]['Low']
        
        trade_taken = False
        entry_price = 0
        direction = 0
        stop_price = 0
        
        # Scan subsequent bars
        for i in range(1, len(day_data)):
            bar = day_data.iloc[i]
            
            if not trade_taken:
                # ENTRY
                if bar['Close'] > or_high:
                    entry_price = bar['Close']
                    direction = 1
                    stop_price = entry_price - trail_points
                    trade_taken = True
                    trades += 1
                elif bar['Close'] < or_low:
                    entry_price = bar['Close']
                    direction = -1
                    stop_price = entry_price + trail_points
                    trade_taken = True
                    trades += 1
            else:
                # TRAILING STOP
                if direction == 1:
                    if bar['Low'] <= stop_price: # Stopped out
                        pnl = ((stop_price - entry_price) / tick_size * tick_value) - fee
                        total_pnl += pnl
                        break
                    if (bar['High'] - trail_points) > stop_price:
                        stop_price = bar['High'] - trail_points
                        
                elif direction == -1:
                    if bar['High'] >= stop_price:
                        pnl = ((entry_price - stop_price) / tick_size * tick_value) - fee
                        total_pnl += pnl
                        break
                    if (bar['Low'] + trail_points) < stop_price:
                        stop_price = bar['Low'] + trail_points
    
    return total_pnl, trades

# 3. RUN THE GRID SEARCH
print(f"{'OR Time':<10} | {'Trail Pts':<10} | {'Trades':<8} | {'Total Profit'}")
print("-" * 50)

best_pnl = -99999
best_params = ""

# Test OR durations: 15 min, 30 min, 60 min
# Test Trails: 20 to 100 points
for orb_time in [15, 30, 60]:
    for trail in [20, 30, 40, 50, 60, 80]:
        pnl, count = test_parameters(orb_time, trail)
        
        # Only show decent results
        if pnl > 0:
            print(f"{orb_time} min     | {trail:<10} | {count:<8} | ${pnl:.2f}")
            
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = f"OR: {orb_time} mins, Trail: {trail} pts"

print("-" * 50)
print(f"BEST SETTINGS: {best_params} -> ${best_pnl:.2f}")