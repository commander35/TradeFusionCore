# AI Trading Bot User Guide

## Introduction

This guide explains how to set up and use the Multi-Timeframe AI Trading Bot system that integrates TradingView data with machine learning for generating trading signals. The system is designed to provide comprehensive analysis and trading recommendations for cryptocurrency markets.

## Getting Started

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB of free space
- **Optional**: NVIDIA GPU for faster model training

### Installation Process

1. **Install Python** (if not already installed):
   Download and install from [python.org](https://www.python.org/downloads/)

2. **Clone or download the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-trading-bot.git
   cd ai-trading-bot
   ```

3. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.sample .env
   ```
   Then edit the `.env` file with your credentials.

### TradingView API Setup

The system requires TradingView API access:

1. **Create a TradingView account** if you don't have one
2. **Get your session token and signature**:
   - Log in to TradingView in your browser
   - Open Developer Tools (F12)
   - Go to Application tab → Cookies
   - Find and copy the values for `sessionid` and `signature`
3. **Add these to your `.env` file**:
   ```
   TRADINGVIEW_TOKEN=your_session_token
   TRADINGVIEW_SIGNATURE=your_signature_cookie
   ```

### Setting Up Notifications

For trading alerts:

1. **Telegram**:
   - Create a bot using [@BotFather](https://t.me/botfather) on Telegram
   - Get your bot token and chat ID
   - Add to `.env`:
     ```
     TELEGRAM_BOT_TOKEN=your_bot_token
     TELEGRAM_CHAT_ID=your_chat_id
     ```

2. **WhatsApp** (optional):
   - Register with a WhatsApp API provider
   - Get API credentials
   - Add to `.env`:
     ```
     WHATSAPP_API_TOKEN=your_api_token
     WHATSAPP_TO_PHONE=your_phone_number
     ```

## Basic Usage

### Running the Bot

1. **Start the bot**:
   ```bash
   python trading_bot.py
   ```

2. **Stop the bot**: Press `Ctrl+C` in the terminal

### Configuration

Modify `config.py` to change:

- **Assets to trade**: Edit `TRADING_CONFIG["assets"]`
- **Timeframes to analyze**: Edit `TRADING_CONFIG["timeframes"]`
- **Risk parameters**: Edit `RISK_CONFIG`
- **Model parameters**: Edit model-specific configurations

Example configuration change:
```python
# To change the risk per trade from 2% to 1%
TRADING_CONFIG = {
    # ... other settings ...
    "risk_per_trade": 0.01,  # Changed from 0.02
    # ... other settings ...
}
```

### Backtesting

To test the system on historical data:

```bash
python trading_bot.py --backtest --start 2023-01-01 --end 2023-03-31
```

After backtesting, review the report in the `backtests/` directory.

## Advanced Features

### Model Retraining

Models are automatically retrained every 4 hours, but you can manually trigger retraining:

```python
from trading_bot import TradingBot

bot = TradingBot()
bot._retrain_models()  # Triggers manual retraining
```

### Custom Indicators

To add custom indicators:

1. Edit `pine_analysis.py`
2. Implement your indicator calculation
3. Add it to the analysis process in the `process_data` method

### Risk Management Customization

Adjust risk parameters in `config.py`:

```python
RISK_CONFIG = {
    "volatility_lookback": 20,  # Number of periods for ATR calculation
    "atr_multiplier": 2.5,      # Multiplier for stop-loss distance
    "min_profit_ratio": 1.5,    # Minimum reward:risk ratio
    # ... other settings ...
}
```

## Understanding Outputs

### Signal Format

The system generates trading signals in this format:

```json
{
  "timestamp": "2024-03-15 14:30:00",
  "asset": "BTC/USD",
  "composite_score": 0.92,
  "price_prediction": {
    "1h": 51200,
    "4h": 52500,
    "24h": 54100
  },
  "sentiment_impact": 0.78,
  "recommended_action": "LONG_LIMITED_RISK",
  "risk_parameters": {
    "max_size": "1.8%",
    "dynamic_sl": 50500,
    "take_profit": 53000,
    "pyramiding_levels": [50800, 51500],
    "confidence_interval": [49600, 52800]
  },
  "signal_components": {
    "price_prediction_1h": 0.65,
    "price_prediction_4h": 0.58,
    "rsi_divergence_1h": 0.5,
    "macd_divergence_1h": 0.7,
    "sentiment_impact": 0.65
  }
}
```

### Interpreting the Signal

- **composite_score**: Overall signal strength (0-1, higher is more bullish)
- **recommended_action**: Trading recommendation (LONG/SHORT/NEUTRAL + risk level)
- **risk_parameters**: Stop-loss, take-profit and position sizing information
- **signal_components**: Individual factors contributing to the signal

### Action Types

The system generates these action recommendations:

- **LONG_LIMITED_RISK**: Strong buy signal with low risk
- **LONG_MODERATE_RISK**: Buy signal with moderate risk
- **LONG_HIGH_RISK**: Buy signal with high risk
- **SHORT_LIMITED_RISK**: Strong sell signal with low risk
- **SHORT_MODERATE_RISK**: Sell signal with moderate risk
- **SHORT_HIGH_RISK**: Sell signal with high risk
- **NEUTRAL**: No clear signal to trade

## Performance Monitoring

### Log Files

- **trading_bot.log**: Main trading bot activity
- **data_collector.log**: Data collection process
- **pine_analysis.log**: Technical analysis results
- **price_prediction.log**: ML model predictions
- **sentiment_analysis.log**: Sentiment analysis results

Access logs in the `logs/` directory.

### Backtest Reports

Backtest reports include:

- **Profit/Loss**: Overall performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest decline
- **Win Rate**: Percentage of profitable trades

## Troubleshooting

### Common Issues

1. **TradingView API Connection Issues**:
   - Verify your session token and signature are correct
   - Check if your IP is being blocked
   - Ensure you're not exceeding API rate limits

2. **Model Training Errors**:
   - Ensure you have enough historical data
   - Check for missing features or NaN values
   - Verify GPU configuration if using CUDA

3. **Missing Dependencies**:
   - Run `pip install -r requirements.txt` again
   - Check for version conflicts using `pip check`

4. **Out of Memory Errors**:
   - Reduce batch sizes in model configurations
   - Limit the number of assets or timeframes
   - Use data sampling for long historical periods

### Getting Help

If you encounter issues:

1. Check the log files for error messages
2. Review documentation for configuration guidance
3. Inspect signal outputs for anomalies
4. Verify data quality and availability

## Best Practices

1. **Start Small**: Begin with 1-2 assets before scaling up
2. **Use Paper Trading**: Test the system before using real funds
3. **Monitor Regularly**: Check performance and signals daily
4. **Retrain Periodically**: Update models when market conditions change
5. **Backup Your Data**: Regularly backup model files and configurations

## Disclaimer

This trading bot is for educational and research purposes only. Always perform thorough testing before using with real funds. The system does not guarantee profitable results, and trading financial markets involves substantial risk.

## Support

For additional help and updates, contact the developer through the repository issues page or email.

Happy Trading! 