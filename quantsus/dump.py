### 20260319

def open_margin_position(signal, cash, price, 
                  invest_fraction = INVEST_FRACTION,
                  leverage=LEVERAGE,
                  entry_cost=ENTRY_COST,
                  min_unit=MIN_UNIT):
    exec_price = price + signal * entry_cost
    margin_per_unit = exec_price / leverage
    max_units = cash / margin_per_unit
    invest_fraction = min(invest_fraction, 1)
    units = np.floor(max_units*invest_fraction / min_unit)*min_unit*signal
    margin_used = abs(units) * margin_per_unit
    return cash, units, exec_price


def close_margin_position(prev_spot, cash, price, units,
                 exit_cost = EXIT_COST):
    signal = np.sign(units)
    exit_spread = signal*exit_cost
    exec_price = price - exit_spread
    pnl = units * (exec_price - prev_spot)
    cash += pnl
    return cash, 0, 0


def is_margin_safe(prev_spot, cash, price, units,
                   leverage=LEVERAGE,
                   liquidation_level=LIQUIDATION_LEVEL):
    if units == 0:
        return True

    equity = cash + units * (price - prev_spot)
    initial_margin = abs(units) * prev_spot / leverage
    
    if (equity / initial_margin >= liquidation_level):
        return True
    else:
        print('Liquidation happened!')
        return False



### 20260324
START_DATE = '2025-01-01'
END_DATE  = '2025-04-01'
FINAL_DATE = '2025-06-01'

# TICK_FREQ = '1h'
FEATURE_WINDOW_SIZE = 5
INITIAL_CASH = 1e5


path_dict = {
    "XAUUSD": "data/YF_XAUUSD_20250101_20260331_1h.csv"
}

loader = SusLoadCsvs(path_dict)
df_dict = loader.load()
data = SusMarketData(df_dict)
data.align()

Xtrain = data.slice(start = START_DATE, end = END_DATE)
Xtest  = data.slice(start = END_DATE, end = FINAL_DATE)

assets = [
    SusAssetParameters("XAUUSD", multiplier=100, min_unit=0.01, tx_cost_bp=0.5, slippage_bp=0, margin_rate=0.05)
]
n_assets = len(assets)

account = SusAccount()
account.reset(n_assets)

exec_engine = SusExecutionEngine(assets, account)

features = {}

features["vwap_spread"] = qs.factors.vwap_signal(Xtrain.high, Xtrain.low, Xtrain.close, Xtrain.volume)

features["log_ret"] = np.log(Xtrain.close / Xtrain.close.shift(1)).fillna(0)

log_vol = np.log(Xtrain.volume + 1)
log_vol_mean = log_vol.rolling(FEATURE_WINDOW_SIZE, min_periods = 1).mean()
features["norm_log_vol"] = (log_vol - log_vol_mean).fillna(0)

hours = Xtrain.close.index.hour
time_sin = np.sin(2 * np.pi * hours / 24)

features["time_sin"] = pd.DataFrame(
    np.tile(time_sin.values.reshape(-1, 1), (1, Xtrain.close.shape[1])),
    index=Xtrain.close.index,
    columns=Xtrain.close.columns
)

feature_names = [
    "vwap_spread",
    "log_ret",
    "norm_log_vol",
    "time_sin"
]

feature_engine = SusFeatureEngine(
    features=features,
    feature_names=feature_names,
    window_size=FEATURE_WINDOW_SIZE
)

env = SusTradingEnv(
    data=Xtrain,
    exec_engine=exec_engine,
    feature_engine=feature_engine,
    position_change_penalty = 0.5
)

env.reset()
state = env.get_state()
n_states = state.shape[0]

agent = SACAgent(state_dim=n_states, action_dim=n_assets, device = 'mps')
analyzer = SusPerformanceAnalyzer()

training_rewards_ary = []
sharpe_ary = []
mdd_ary = []
volatility_ary = []
calmar_ary = []
win_ratio_ary = []
pnl_ary = []

episodes = 1

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    episode_returns = []
    positions = []

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, trade = env.step(action)
        
        episode_returns.append(trade['pct_pnl'])
        positions.append(trade['positions'])

        agent.buffer.add(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        total_reward += reward

    metrics = analyzer.evaluate(episode_returns)
    pnl = account.cash / account.initial_cash

    print(
        f"Episode {ep}: {pnl:.2%} \n"
        f"    -Reward: {total_reward:.2f} \n"
        f"    -Sharpe: {metrics['sharpe']:.3f} \n"
        f"    -MaxDrawDown: {metrics['max_drawdown']:.2%} \n"
        f"    -Calmar Ratio: {metrics['calmar_ratio']:.2f} \n"
        f"    -Volatility: {metrics['volatility']:.2f} \n"
        f"    -WinRate: {metrics['win_rate']:.2%} \n"
        f"-------------------------------------"
    )

    training_rewards_ary.append(total_reward)
    sharpe_ary.append(metrics['sharpe'])
    mdd_ary.append(metrics['max_drawdown'])
    volatility_ary.append(metrics['volatility'])
    calmar_ary.append(metrics['calmar_ratio'])
    win_ratio_ary.append(metrics['win_rate'])
    pnl_ary.append(pnl)


## v2

def make_trading_env(
    data,
    assets: list,
    feature_specs: List[Tuple[str, Callable]],
    feature_window_size: int = 5,
    initial_cash: float = 1e5,
    position_penalty: float = 0.5
):
    """
    Build a complete trading environment with account, execution, features, and env.
    
    Returns:
        env: SusTradingEnv
        n_states: int (state dimension)
        account: SusAccount
    """
    # --- Account & Execution Engine ---
    account = SusAccount()
    account.reset(len(assets))
    
    exec_engine = SusExecutionEngine(assets, account)
    
    features = qs.features.factors.build_features(data, feature_specs)
    feature_names = [name for name, _ in feature_specs]
    
    feature_engine = SusFeatureEngine(
        features=features,
        feature_names=feature_names,
        window_size=feature_window_size
    )
    
    env = SusTradingEnv(
        data=data,
        exec_engine=exec_engine,
        feature_engine=feature_engine,
        position_change_penalty=position_penalty
    )
    
    env.reset()
    state = env.get_state()
    n_states = state.shape[0]
    
    return env, n_states, account



START_DATE = '2025-01-01'
END_DATE  = '2025-04-01'
FINAL_DATE = '2025-06-01'

# TICK_FREQ = '1h'
FEATURE_WINDOW_SIZE = 5
INITIAL_CASH = 1e5


path_dict = {
    "XAUUSD": "data/YF_XAUUSD_20250101_20260331_1h.csv"
}

loader = SusLoadCsvs(path_dict)
df_dict = loader.load()
data = SusMarketData(df_dict)
data.align()
assets = [
    SusAssetParameters("XAUUSD", multiplier=100, min_unit=0.01, tx_cost_bp=0.5, slippage_bp=0, margin_rate=0.05)
]
n_assets = len(assets)

feature_specs = [
    ("vwap_spread", partial(qs.features.factors.vwap, window=FEATURE_WINDOW_SIZE)),
    ("log_ret", qs.features.factors.log_return),
    ("norm_log_vol", partial(qs.features.factors.normalized_log_volume, window=FEATURE_WINDOW_SIZE)),
    ("time_sin", qs.features.factors.time_sin_hour),
]

env_train, n_states, account_train = make_trading_env(
    Xtrain, assets, feature_specs, feature_window_size=FEATURE_WINDOW_SIZE
)

env_test, _, account_test = make_trading_env(
    Xtest, assets, feature_specs, feature_window_size=FEATURE_WINDOW_SIZE
)

# Initialize agent and analyzer
agent = SACAgent(state_dim=n_states, action_dim=n_assets, device='mps')
analyzer = SusPerformanceAnalyzer()

# Number of training epochs
epochs = 1

# Storage for metrics
metrics_history = {
    "train": {
        "reward": [], "sharpe": [], "max_drawdown": [],
        "volatility": [], "calmar_ratio": [], "win_rate": [], "pnl": []
    },
    "test": {
        "reward": [], "sharpe": [], "max_drawdown": [],
        "volatility": [], "calmar_ratio": [], "win_rate": [], "pnl": []
    }
}

for ep in range(epochs):
    # ------------------------
    # 1.TRAIN on Xtrain
    # ------------------------
    state = env_train.reset()
    done = False
    total_reward = 0
    episode_returns = []

    while not done:
        action = agent.select_action(state)  # normal SAC policy
        next_state, reward, done, trade = env_train.step(action)

        # store in replay buffer & update agent
        agent.buffer.add(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        total_reward += reward
        episode_returns.append(trade['pct_pnl'])

    # Train metrics
    train_metrics = analyzer.evaluate(episode_returns)
    train_pnl = env_train.exec_engine.account.cash / env_train.exec_engine.account.initial_cash

    # Store
    metrics_history["train"]["reward"].append(total_reward)
    metrics_history["train"]["sharpe"].append(train_metrics["sharpe"])
    metrics_history["train"]["max_drawdown"].append(train_metrics["max_drawdown"])
    metrics_history["train"]["volatility"].append(train_metrics["volatility"])
    metrics_history["train"]["calmar_ratio"].append(train_metrics["calmar_ratio"])
    metrics_history["train"]["win_rate"].append(train_metrics["win_rate"])
    metrics_history["train"]["pnl"].append(train_pnl)

    print(f"Epoch {ep} TRAIN | PnL: {train_pnl:.2%} | Reward: {total_reward:.2f} | Sharpe: {train_metrics['sharpe']:.3f}")

    # ------------------------
    # 2.EVALUATE on Xtest
    # ------------------------
    state = env_test.reset()
    done = False
    total_reward = 0
    episode_returns = []

    while not done:
        # deterministic policy: no exploration, no learning
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, trade = env_test.step(action)

        state = next_state
        total_reward += reward
        episode_returns.append(trade['pct_pnl'])

    test_metrics = analyzer.evaluate(episode_returns)
    test_pnl = env_test.exec_engine.account.cash / env_test.exec_engine.account.initial_cash

    # Store
    metrics_history["test"]["reward"].append(total_reward)
    metrics_history["test"]["sharpe"].append(test_metrics["sharpe"])
    metrics_history["test"]["max_drawdown"].append(test_metrics["max_drawdown"])
    metrics_history["test"]["volatility"].append(test_metrics["volatility"])
    metrics_history["test"]["calmar_ratio"].append(test_metrics["calmar_ratio"])
    metrics_history["test"]["win_rate"].append(test_metrics["win_rate"])
    metrics_history["test"]["pnl"].append(test_pnl)

    print(f"Epoch {ep} TEST  | PnL: {test_pnl:.2%} | Sharpe: {test_metrics['sharpe']:.3f}")
    print("-" * 50)