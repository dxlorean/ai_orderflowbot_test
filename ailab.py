import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_FILE = 'OrderFlowData.csv'
MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE = 0.50
MNQ_POINT_VALUE = 2.0
ROUND_TRIP_FEE = 1.24           # $1.24 per trade

# Hyperparameters
WINDOW_SIZE = 10                # AI sees last 10 bars
INITIAL_BALANCE = 10000.0       # Sim account size
TRAIN_SPLIT = 0.7               # 70% Train, 30% Test

# ==========================================
# 2. CUSTOM TRADING ENVIRONMENT
# ==========================================
class MNQTradingEnv(gym.Env):
    """
    Custom Gym Environment for MNQ Trading.
    Actions: 0=FLAT (Close), 1=LONG (Buy), 2=SHORT (Sell)
    """
    def __init__(self, df):
        super(MNQTradingEnv, self).__init__()
        self.df = df
        self.max_steps = len(df) - 1
        
        # Action Space: 0=Flat, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: [Features] * Window + [Position Status]
        self.feature_cols = ['Close_Pct', 'Delta_Norm', 'CVD_Norm', 'VWAP_Dist', 'SessionHigh_Dist']
        n_features = len(self.feature_cols)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(WINDOW_SIZE * n_features + 2,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = WINDOW_SIZE
        self.balance = INITIAL_BALANCE
        self.position = 0       # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.equity_curve = [INITIAL_BALANCE]
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Get window of data
        window = self.df.iloc[self.current_step - WINDOW_SIZE : self.current_step]
        obs = window[self.feature_cols].values.flatten().astype(np.float32)
        
        # Append Position Info (State Awareness)
        pos_info = np.array([self.position, self.entry_price], dtype=np.float32)
        return np.concatenate((obs, pos_info))

    def step(self, action):
        # 1. Get Market Data
        current_price = self.df.iloc[self.current_step]['Close']
        prev_price = self.df.iloc[self.current_step - 1]['Close']
        
        reward = 0
        fee_cost = 0
        
        # 2. Execute Action
        # Action 0: FLAT
        if action == 0:
            if self.position != 0:
                fee_cost = ROUND_TRIP_FEE / 2
                self.position = 0
                
        # Action 1: LONG
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                fee_cost = ROUND_TRIP_FEE / 2
            elif self.position == -1:
                self.position = 1
                self.entry_price = current_price
                fee_cost = ROUND_TRIP_FEE

        # Action 2: SHORT
        elif action == 2:
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                fee_cost = ROUND_TRIP_FEE / 2
            elif self.position == 1:
                self.position = -1
                self.entry_price = current_price
                fee_cost = ROUND_TRIP_FEE

        # 3. Calculate PnL
        step_pnl = 0
        if self.position == 1:
            step_pnl = (current_price - prev_price) * MNQ_POINT_VALUE
        elif self.position == -1:
            step_pnl = (prev_price - current_price) * MNQ_POINT_VALUE
            
        step_profit = step_pnl - fee_cost
        self.balance += step_profit
        self.equity_curve.append(self.balance)
        
        # ------------------------------------------
        # 4. SCALED REWARD SHAPING (The Fix)
        # ------------------------------------------
        # We scale rewards down so the Neural Network doesn't explode.
        # $20 profit = +1.0 reward.
        REWARD_SCALE = 20.0 
        reward = step_profit / REWARD_SCALE

        # Living Cost (Scaled)
        # Force the AI to move. Staying flat costs "points".
        if self.position == 0:
            reward -= 0.05 

        # Asymmetric Risk (Fear Factor)
        # Losing feels worse than winning feels good.
        if step_profit < 0:
            reward *= 1.2

        # 5. Termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Kill switch if account blows up (saves training time)
        if self.balance < INITIAL_BALANCE * 0.7:
            terminated = True
            reward -= 10 # Massive penalty
        
        info = {'balance': self.balance, 'position': self.position}
        
        return self._get_observation(), reward, terminated, truncated, info

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
def load_and_process_data(filepath):
    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';', decimal='.')
        df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
    except:
        # Dummy data generator for testing if file missing
        print("Warning: CSV not found. Generating dummy data.")
        dates = pd.date_range(start='2024-01-01', periods=2000, freq='15min')
        df = pd.DataFrame({'Time': dates, 'Close': np.random.normal(15000, 50, 2000).cumsum()})
        df['Open'] = df['Close'] + np.random.normal(0, 5, 2000)
        df['High'] = df['Close'] + 10
        df['Low'] = df['Close'] - 10
        df['Delta'] = np.random.normal(0, 200, 2000)
        df['CVD'] = df['Delta'].cumsum()
        df['VWAP_Dist'] = np.random.normal(0, 5, 2000)
        df['SessionHigh_Dist'] = np.random.normal(0, 10, 2000)

    # Feature Engineering (Z-Scores)
    df['Close_Pct'] = df['Close'].pct_change().fillna(0) * 100
    df['Delta_Norm'] = (df['Delta'] - df['Delta'].rolling(100).mean()) / (df['Delta'].rolling(100).std() + 1e-5)
    df['CVD_Norm'] = (df['CVD'] - df['CVD'].rolling(100).mean()) / (df['CVD'].rolling(100).std() + 1e-5)
    df.fillna(0, inplace=True)
    
    return df

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # A. Load Data
    full_df = load_and_process_data(CSV_FILE)
    
    # Split Train/Test
    split_idx = int(len(full_df) * TRAIN_SPLIT)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    test_df = full_df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Training on {len(train_df)} bars. Testing on {len(test_df)} bars.")
    
    # B. Setup Training Environment
    train_env = DummyVecEnv([lambda: MNQTradingEnv(train_df)])
    
    # C. Initialize AI (PPO Agent)
    # ent_coef=0.05 forces exploration (prevents early collapse to "Flat")
    print("--- INITIALIZING PPO AGENT ---")
    model = PPO("MlpPolicy", train_env, verbose=1, 
                learning_rate=0.0003, 
                ent_coef=0.05, 
                batch_size=128)
    
    # D. Train
    print("--- STARTING AI TRAINING (This may take a while) ---")
    model.learn(total_timesteps=50000) 
    print("--- TRAINING COMPLETE ---")
    
    # E. Test Suite (With Visual Heartbeat)
    print("\n--- RUNNING ROBUST TEST SUITE ---")
    test_env = MNQTradingEnv(test_df)
    obs, _ = test_env.reset()
    done = False
    
    bar_count = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        bar_count += 1
        
        # Heartbeat: Print status every 100 bars so you know it's working
        if bar_count % 100 == 0:
            print(f"Simulating Bar {bar_count}... Balance: ${info['balance']:.2f} | Pos: {info['position']}")
            
    # F. Generate Report
    equity = np.array(test_env.equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    total_return = (equity[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252*96) 
    max_drawdown = np.max(np.maximum.accumulate(equity) - equity) / np.max(np.maximum.accumulate(equity)) * 100
    
    print("\n" + "="*40)
    print("      AI PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Final Balance:    ${equity[-1]:.2f}")
    print(f"Total Return:     {total_return:.2f}%")
    print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {max_drawdown:.2f}%")
    print("="*40)
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(equity, label="AI Equity Curve")
    plt.title("AI Trading Performance (Out-of-Sample)")
    plt.xlabel("Trades / Bars")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.show()