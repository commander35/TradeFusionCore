import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
BACKTEST_DIR = BASE_DIR / "backtests"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, BACKTEST_DIR]:
    directory.mkdir(exist_ok=True)

# Trading configuration
TRADING_CONFIG = {
    "assets": ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "SOL/USD"],
    "timeframes": ["1h", "4h", "1d"],
    "risk_per_trade": 0.02,  # 2% max risk per trade
    "max_open_positions": 3,
    "retraining_hours": 4,  # Retrain models every 4 hours
}

# Technical indicators configuration
INDICATOR_CONFIG = {
    "rsi": {"length": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"length": 20, "std": 2},
    "pivots": {"types": ["standard", "fibonacci"]},
}

# KNN Configuration
KNN_CONFIG = {
    "n_neighbors": 5,
    "distance_metric": "euclidean",
    "weights": "distance"
}

# LSTM Model Configuration
LSTM_CONFIG = {
    "units": 128,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10,
}

# CNN Configuration
CNN_CONFIG = {
    "filters": [64, 128, 256],
    "kernel_sizes": [3, 3, 3],
    "pool_sizes": [2, 2, 2],
    "dropout": 0.2,
    "learning_rate": 0.001,
}

# XGBoost Configuration
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
}

# Risk Management Configuration
RISK_CONFIG = {
    "volatility_lookback": 20,
    "atr_multiplier": 2.5,
    "min_profit_ratio": 1.5,  # Minimum reward:risk ratio
    "liquidity_threshold": 0.3,  # Minimum required liquidity as % of position
    "max_adverse_timeframes": 1,  # Maximum number of timeframes with contrary signals
}

# API configurations
TRADINGVIEW_CONFIG = {
    "username": os.environ.get("TRADINGVIEW_USERNAME", ""),
    "password": os.environ.get("TRADINGVIEW_PASSWORD", ""),
    "token": os.environ.get("TRADINGVIEW_TOKEN", ""),
    "signature": os.environ.get("TRADINGVIEW_SIGNATURE", ""),
}

# News and social media API configs
NEWS_API_CONFIG = {
    "api_key": os.environ.get("NEWS_API_KEY", ""),
    "crypto_keywords": ["bitcoin", "ethereum", "crypto", "blockchain", "altcoin"],
}

TWITTER_API_CONFIG = {
    "api_key": os.environ.get("TWITTER_API_KEY", ""),
    "api_secret": os.environ.get("TWITTER_API_SECRET", ""),
    "bearer_token": os.environ.get("TWITTER_BEARER_TOKEN", ""),
}

REDDIT_API_CONFIG = {
    "client_id": os.environ.get("REDDIT_CLIENT_ID", ""),
    "client_secret": os.environ.get("REDDIT_CLIENT_SECRET", ""),
    "user_agent": "TradingBot/1.0"
}

# Notification configurations
TELEGRAM_CONFIG = {
    "token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
}

WHATSAPP_CONFIG = {
    "token": os.environ.get("WHATSAPP_API_TOKEN", ""),
    "to_phone": os.environ.get("WHATSAPP_TO_PHONE", ""),
}

# MongoDB configuration
MONGO_CONFIG = {
    "uri": os.environ.get("MONGO_URI", "mongodb://localhost:27017/"),
    "db_name": "crypto_trading_bot",
}

# DeepSeek AI Configuration
DEEPSEEK_CONFIG = {
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "model": "deepseek-coder-33b-instruct",
}

# Ensemble Model Weights
ENSEMBLE_WEIGHTS = {
    "price_prediction": 0.4,
    "technical_indicators": 0.3,
    "sentiment_analysis": 0.2,
    "market_depth": 0.1,
}

# Output Signal Thresholds
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.75,
    "buy": 0.6,
    "neutral": 0.4,
    "sell": 0.3,
    "strong_sell": 0.25,
} 