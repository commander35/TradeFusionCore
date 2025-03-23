# TradingView API Integration Guide

This guide explains how to set up and use the TradingView API with our trading bot system. The system leverages this API to access real-time market data, technical indicators, and chart patterns from TradingView.

## Introduction to the TradingView API

The TradingView API used in this project is an unofficial API implementation that allows programmatic access to TradingView's data and features. This library (`@mathieuc/tradingview`) provides the capability to:

1. Get real-time market data
2. Access built-in indicators (RSI, MACD, Bollinger Bands, etc.)
3. Use custom Pine Script indicators
4. Retrieve historical data for backtesting
5. Access technical analysis recommendations

## Setup and Authentication

### Prerequisites

- Node.js installed (for the TradingView API library)
- Valid TradingView account (free or paid)

### Installation

The TradingView API library is already included in the project, but if you need to install it separately:

```bash
npm install @mathieuc/tradingview
```

### Getting Authentication Credentials

To authenticate with TradingView, you need to obtain your session token and signature:

1. Log in to TradingView in your web browser
2. Open Developer Tools (F12 or right-click → Inspect)
3. Go to the Application tab
4. Select Cookies in the sidebar
5. Find the following cookies for the tradingview.com domain:
   - `sessionid` - This is your TOKEN
   - `signature` - This is your SIGNATURE

### Setting Up Environment Variables

Add your credentials to the `.env` file:

```
TRADINGVIEW_TOKEN=your_session_token
TRADINGVIEW_SIGNATURE=your_signature_cookie
```

## Using the API in the Trading Bot

### Data Collection

The `data_collector.py` module handles all interactions with the TradingView API. Key functionalities include:

1. **Connecting to TradingView**:
   ```python
   self.client = Client({
       "token": TRADINGVIEW_CONFIG["token"],
       "signature": TRADINGVIEW_CONFIG["signature"]
   })
   ```

2. **Creating Chart Sessions**:
   ```python
   chart = self.client.Session.Chart()
   chart.setMarket('BINANCE:BTCUSD', {
       "timeframe": "60",  # 1-hour timeframe
   })
   ```

3. **Adding Technical Indicators**:
   ```python
   rsi_indicator = BuiltInIndicator("RSI@tv-basicstudies")
   rsi_indicator.setOption("length", 14)
   rsi_study = chart.Study(rsi_indicator)
   ```

4. **Retrieving Price Data**:
   ```python
   # When price updates, this function is called
   chart.onUpdate(lambda: self._on_chart_update(chart, asset, timeframe))
   
   # Inside the update handler
   latest_period = chart.periods[0]
   data_point = {
       "timestamp": latest_period.time,
       "open": latest_period.open,
       "high": latest_period.max,
       "low": latest_period.min,
       "close": latest_period.close,
       "volume": latest_period.volume
   }
   ```

### Available Indicators

The system is configured to use these indicators by default:

1. **RSI (Relative Strength Index)**
   ```python
   rsi_indicator = BuiltInIndicator("RSI@tv-basicstudies")
   ```

2. **MACD (Moving Average Convergence Divergence)**
   ```python
   macd_indicator = BuiltInIndicator("MACD@tv-basicstudies")
   ```

3. **Bollinger Bands**
   ```python
   bb_indicator = BuiltInIndicator("BB@tv-basicstudies")
   ```

4. **Pivot Points** (calculated from OHLC data)

### Adding Custom Indicators

To add a custom indicator:

1. Find the indicator ID in TradingView:
   - Open the indicator in TradingView
   - Use Developer Tools to inspect network requests
   - Look for the study ID in the request payload

2. Add the indicator to your chart:
   ```python
   custom_indicator = BuiltInIndicator("YourIndicator@tv-basicstudies")
   custom_indicator.setOption("param1", value1)
   custom_indicator.setOption("param2", value2)
   chart.Study(custom_indicator)
   ```

3. Extract data in the update handler:
   ```python
   if custom_study.periods and len(custom_study.periods) > 0:
       custom_value = custom_study.periods[0].get("plot_0", None)
   ```

## Using Premium Indicators

For premium or invite-only indicators, you need:

1. A TradingView account with access to the indicator
2. Valid session and signature cookies
3. Use the `PineIndicator` class instead of `BuiltInIndicator`

Example:
```python
from main import PineIndicator, PinePermManager

# Get private indicators
indic_list = await getPrivateIndicators(session_token)

for indic in indic_list:
    private_indic = await indic.get()
    indicator = chart.Study(private_indic)
```

## Handling Data Storage

The system stores collected data in CSV files:

1. **File Organization**: Data is stored in the `data/` directory, organized by asset and timeframe
2. **File Naming**: Files follow the pattern `{timeframe}_{date}.csv` (e.g., `1h_2023-03-15.csv`)
3. **Data Format**: CSV files contain columns for timestamp, OHLCV data, and indicator values

## Troubleshooting TradingView API Issues

### Common Issues

1. **Authentication Errors**:
   - Cookies may have expired (typically valid for 2 weeks)
   - Account may be logged out or restricted
   - IP address may be blocked

   Solution: Refresh your cookies by logging in again to TradingView and updating your `.env` file.

2. **Rate Limiting**:
   - TradingView limits the number of requests
   - Too many connections may be blocked

   Solution: Implement rate limiting in your code, reduce the number of assets or timeframes.

3. **Indicator Data Not Available**:
   - Incorrect indicator ID
   - Indicator requires premium account
   - Market data not available for the symbol

   Solution: Verify indicator IDs, check TradingView account permissions, ensure market is available.

### Debugging Tips

1. Enable verbose logging in the data collector:
   ```python
   logging.getLogger("DataCollector").setLevel(logging.DEBUG)
   ```

2. Inspect raw API responses:
   ```python
   client.onData(lambda d: print("Raw response:", d))
   ```

3. Test individual components:
   ```bash
   python -c "from data_collector import TradingViewDataCollector; \
   collector = TradingViewDataCollector(['BTC/USD'], ['1h']); \
   collector.start_data_collection(); \
   import time; time.sleep(10); \
   collector.stop_data_collection()"
   ```

## Best Practices

1. **Session Management**:
   - Renew your session tokens regularly
   - Implement automatic reconnection logic

2. **Error Handling**:
   - Always use try/except blocks when calling the API
   - Implement exponential backoff for retries

3. **Data Validation**:
   - Verify data integrity before using it
   - Check for missing values or anomalies

4. **Resource Optimization**:
   - Only collect data for the assets and timeframes you need
   - Batch requests where possible
   - Use efficient storage formats for historical data

## API Limitations

Be aware of these limitations:

1. **Unofficial API**: This is not an official TradingView API, so it may change without notice
2. **Rate Limits**: Excessive usage may lead to account restrictions
3. **Data Accuracy**: Always verify critical data points
4. **Terms of Service**: Ensure your usage complies with TradingView's terms of service

## Example: Complete API Workflow

Here's a complete example of collecting data from TradingView:

```python
from main import Client, BuiltInIndicator
import time

# 1. Connect to TradingView
client = Client({
    "token": "your_session_token",
    "signature": "your_signature_cookie"
})

# 2. Create a chart
chart = client.Session.Chart()

# 3. Set the market and timeframe
chart.setMarket('BINANCE:BTCUSD', {
    "timeframe": "60",  # 1-hour
})

# 4. Add RSI indicator
rsi = BuiltInIndicator("RSI@tv-basicstudies")
rsi.setOption("length", 14)
rsi_study = chart.Study(rsi)

# 5. Set up data handlers
def on_update():
    if not chart.periods or len(chart.periods) == 0:
        return
        
    # Get price data
    latest = chart.periods[0]
    print(f"Price: {latest.close}")
    
    # Get RSI data
    if rsi_study.periods and len(rsi_study.periods) > 0:
        rsi_value = rsi_study.periods[0].get("plot_0", None)
        print(f"RSI: {rsi_value}")

# 6. Register the update handler
chart.onUpdate(on_update)

# 7. Wait for data
print("Waiting for data...")
time.sleep(30)

# 8. Clean up
chart.delete()
client.end()
```

By following this guide, you should be able to effectively use the TradingView API within the trading bot system to collect and analyze market data for your trading strategies. 