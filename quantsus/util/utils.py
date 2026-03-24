from typing import List, Tuple, Callable
import pandas as pd
import numpy as np
from ..agents.env import SusTradingEnv
from ..core.feature_engine import SusFeatureEngine
from ..core.exec_engine import SusExecutionEngine
from ..core.account import SusAccount
from ..metrics.analyzer import SusPerformanceAnalyzer

from ..features import *


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
        analyzer: SusPerformanceAnalyzer
    """
    # --- Account & Execution Engine ---
    account = SusAccount()
    account.reset(len(assets))
    
    exec_engine = SusExecutionEngine(assets, account)
    
    features = build_features(data, feature_specs)
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
    analyzer = SusPerformanceAnalyzer()
    
    return env, n_states, account, analyzer