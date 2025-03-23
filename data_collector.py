import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import json
from pathlib import Path

# Import the TradingView API library
import sys
sys.path.append(".")  # Add current directory to path
from config import TRADING_CONFIG, TRADINGVIEW_CONFIG, DATA_DIR, INDICATOR_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollector")

class TradingViewDataCollector:
    """
    Class responsible for collecting data from TradingView using the TradingView API library
    """
    def __init__(self, assets=None, timeframes=None):
        """
        Initialize the data collector
        
        Args:
            assets (list): List of assets to collect data for (e.g. ["BTC/USD"])
            timeframes (list): List of timeframes (e.g. ["1h", "4h", "1d"])
        """
        self.assets = assets or TRADING_CONFIG["assets"]
        self.timeframes = timeframes or TRADING_CONFIG["timeframes"]
        self.data_queue = queue.Queue()
        self.client = None
        self.charts = {}
        self.running = False
        
        # Create directories for data storage
        for asset in self.assets:
            asset_dir = DATA_DIR / asset.replace("/", "_")
            asset_dir.mkdir(exist_ok=True)
            
        self._connect_tradingview()
    
    def _connect_tradingview(self):
        """Connect to the TradingView API"""
        try:
            # Import TradingView module within the function to allow for dependency injection for testing
            from main import Client
            
            logger.info("Connecting to TradingView API...")
            
            # Configure the client with authentication if available
            if TRADINGVIEW_CONFIG["token"] and TRADINGVIEW_CONFIG["signature"]:
                self.client = Client({
                    "token": TRADINGVIEW_CONFIG["token"],
                    "signature": TRADINGVIEW_CONFIG["signature"]
                })
            else:
                self.client = Client()
                
            # Set up error handling
            self.client.onError(self._handle_error)
            logger.info("TradingView API client initialized")
            
        except Exception as e:
            logger.error(f"Failed to connect to TradingView API: {e}")
            raise
    
    def _handle_error(self, *args):
        """Handle errors from the TradingView API"""
        error_msg = " ".join(str(arg) for arg in args)
        logger.error(f"TradingView API error: {error_msg}")
        
        # Attempt reconnection if client is disconnected
        if "disconnect" in error_msg.lower():
            logger.info("Attempting to reconnect to TradingView API...")
            self._connect_tradingview()
    
    def _create_chart_session(self, asset, timeframe):
        """
        Create a chart session for a specific asset and timeframe
        
        Args:
            asset (str): Asset symbol (e.g. "BTC/USD")
            timeframe (str): Timeframe (e.g. "1h", "4h", "1d")
        
        Returns:
            Chart session object
        """
        try:
            # Convert asset format for TradingView (BTC/USD -> BITSTAMP:BTCUSD)
            tv_symbol = self._convert_to_tv_symbol(asset)
            
            # Convert timeframe format for TradingView (1h -> 60, 4h -> 240, 1d -> D)
            tv_timeframe = self._convert_to_tv_timeframe(timeframe)
            
            # Create chart session
            chart = self.client.Session.Chart()
            
            # Set up error handling
            chart.onError(lambda *err: logger.error(f"Chart error for {asset} {timeframe}: {err}"))
            
            # Set up market and timeframe
            chart.setMarket(tv_symbol, {
                "timeframe": tv_timeframe,
            })
            
            # Set up indicators
            self._add_indicators_to_chart(chart, asset, timeframe)
            
            # Set up data collection on updates
            chart.onUpdate(lambda: self._on_chart_update(chart, asset, timeframe))
            
            logger.info(f"Created chart session for {asset} {timeframe}")
            return chart
            
        except Exception as e:
            logger.error(f"Failed to create chart session for {asset} {timeframe}: {e}")
            raise
    
    def _convert_to_tv_symbol(self, asset):
        """
        Convert asset symbol to TradingView format
        
        Args:
            asset (str): Asset symbol (e.g. "BTC/USD")
        
        Returns:
            str: TradingView symbol (e.g. "BITSTAMP:BTCUSD")
        """
        # Simple conversion - in a real implementation, you would have a mapping of assets to exchanges
        exchange = "BINANCE"
        base, quote = asset.split("/")
        return f"{exchange}:{base}{quote}"
    
    def _convert_to_tv_timeframe(self, timeframe):
        """
        Convert timeframe to TradingView format
        
        Args:
            timeframe (str): Timeframe (e.g. "1h", "4h", "1d")
        
        Returns:
            str: TradingView timeframe
        """
        # Mapping of timeframes
        timeframe_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "D",
            "1w": "W",
            "1M": "M"
        }
        
        return timeframe_map.get(timeframe, "D")  # Default to daily if not found
    
    def _add_indicators_to_chart(self, chart, asset, timeframe):
        """
        Add technical indicators to the chart
        
        Args:
            chart: Chart session object
            asset (str): Asset symbol
            timeframe (str): Timeframe
        """
        try:
            from main import BuiltInIndicator
            
            # Add RSI
            rsi_config = INDICATOR_CONFIG["rsi"]
            rsi_indicator = BuiltInIndicator("RSI@tv-basicstudies")
            rsi_indicator.setOption("length", rsi_config["length"])
            self.charts.setdefault(asset, {}).setdefault(timeframe, {})["rsi"] = chart.Study(rsi_indicator)
            
            # Add MACD
            macd_config = INDICATOR_CONFIG["macd"]
            macd_indicator = BuiltInIndicator("MACD@tv-basicstudies")
            macd_indicator.setOption("fast_length", macd_config["fast"])
            macd_indicator.setOption("slow_length", macd_config["slow"])
            macd_indicator.setOption("signal_length", macd_config["signal"])
            self.charts.setdefault(asset, {}).setdefault(timeframe, {})["macd"] = chart.Study(macd_indicator)
            
            # Add Bollinger Bands
            bb_config = INDICATOR_CONFIG["bollinger"]
            bb_indicator = BuiltInIndicator("BB@tv-basicstudies")
            bb_indicator.setOption("length", bb_config["length"])
            bb_indicator.setOption("std", bb_config["std"])
            self.charts.setdefault(asset, {}).setdefault(timeframe, {})["bollinger"] = chart.Study(bb_indicator)
            
            logger.info(f"Added indicators for {asset} {timeframe}")
            
        except Exception as e:
            logger.error(f"Failed to add indicators for {asset} {timeframe}: {e}")
            raise
    
    def _on_chart_update(self, chart, asset, timeframe):
        """
        Handle chart updates - collect the data and put it in the queue
        
        Args:
            chart: Chart session object
            asset (str): Asset symbol
            timeframe (str): Timeframe
        """
        try:
            # Ensure we have data
            if not chart.periods or len(chart.periods) == 0:
                return
            
            # Get the latest period data
            latest_period = chart.periods[0]
            
            # Get indicator data if available
            indicators = {}
            
            # RSI data
            if "rsi" in self.charts.get(asset, {}).get(timeframe, {}):
                rsi_study = self.charts[asset][timeframe]["rsi"]
                if rsi_study.periods and len(rsi_study.periods) > 0:
                    indicators["rsi"] = rsi_study.periods[0].get("plot_0", None)
            
            # MACD data
            if "macd" in self.charts.get(asset, {}).get(timeframe, {}):
                macd_study = self.charts[asset][timeframe]["macd"]
                if macd_study.periods and len(macd_study.periods) > 0:
                    indicators["macd"] = {
                        "macd": macd_study.periods[0].get("plot_0", None),
                        "signal": macd_study.periods[0].get("plot_1", None),
                        "histogram": macd_study.periods[0].get("plot_2", None)
                    }
            
            # Bollinger Bands data
            if "bollinger" in self.charts.get(asset, {}).get(timeframe, {}):
                bb_study = self.charts[asset][timeframe]["bollinger"]
                if bb_study.periods and len(bb_study.periods) > 0:
                    indicators["bollinger"] = {
                        "middle": bb_study.periods[0].get("plot_0", None),
                        "upper": bb_study.periods[0].get("plot_1", None),
                        "lower": bb_study.periods[0].get("plot_2", None)
                    }
            
            # Construct the data point
            data_point = {
                "asset": asset,
                "timeframe": timeframe,
                "timestamp": latest_period.time * 1000,  # Convert to milliseconds
                "price": {
                    "open": latest_period.open,
                    "high": latest_period.max,
                    "low": latest_period.min,
                    "close": latest_period.close,
                    "volume": latest_period.volume
                },
                "indicators": indicators
            }
            
            # Add to queue
            self.data_queue.put(data_point)
            
            # Also save to file
            self._save_data_point(data_point)
            
            logger.debug(f"Collected data for {asset} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error processing chart update for {asset} {timeframe}: {e}")
    
    def _save_data_point(self, data_point):
        """
        Save data point to file
        
        Args:
            data_point (dict): Data point to save
        """
        try:
            asset = data_point["asset"]
            timeframe = data_point["timeframe"]
            date = datetime.fromtimestamp(data_point["timestamp"] / 1000)
            
            # Create file path
            asset_dir = DATA_DIR / asset.replace("/", "_")
            file_path = asset_dir / f"{timeframe}_{date.strftime('%Y-%m-%d')}.csv"
            
            # Convert to DataFrame row
            df_row = {
                "timestamp": data_point["timestamp"],
                "open": data_point["price"]["open"],
                "high": data_point["price"]["high"],
                "low": data_point["price"]["low"],
                "close": data_point["price"]["close"],
                "volume": data_point["price"]["volume"]
            }
            
            # Add indicators if available
            for indicator, value in data_point["indicators"].items():
                if isinstance(value, dict):
                    for key, val in value.items():
                        df_row[f"{indicator}_{key}"] = val
                else:
                    df_row[indicator] = value
            
            # Create or append to DataFrame
            df = pd.DataFrame([df_row])
            
            # Check if file exists, append if it does
            if file_path.exists():
                existing_df = pd.read_csv(file_path)
                
                # Check if this timestamp already exists
                if data_point["timestamp"] not in existing_df["timestamp"].values:
                    updated_df = pd.concat([existing_df, df])
                    updated_df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
                
        except Exception as e:
            logger.error(f"Failed to save data point: {e}")
    
    def start_data_collection(self):
        """Start data collection for all assets and timeframes"""
        if self.running:
            logger.warning("Data collection is already running")
            return
        
        self.running = True
        logger.info("Starting data collection...")
        
        try:
            # Create chart sessions for each asset and timeframe
            for asset in self.assets:
                for timeframe in self.timeframes:
                    if asset not in self.charts:
                        self.charts[asset] = {}
                    
                    # Create chart session
                    self.charts[asset][timeframe] = {}
                    self.charts[asset][timeframe]["session"] = self._create_chart_session(asset, timeframe)
            
            logger.info("Data collection started for all assets and timeframes")
        
        except Exception as e:
            self.running = False
            logger.error(f"Failed to start data collection: {e}")
            raise
    
    def stop_data_collection(self):
        """Stop data collection"""
        if not self.running:
            logger.warning("Data collection is not running")
            return
        
        self.running = False
        logger.info("Stopping data collection...")
        
        try:
            # Close all chart sessions
            for asset in self.charts:
                for timeframe in self.charts[asset]:
                    if "session" in self.charts[asset][timeframe]:
                        self.charts[asset][timeframe]["session"].delete()
            
            # Close client connection
            if self.client:
                self.client.end()
                
            logger.info("Data collection stopped")
        
        except Exception as e:
            logger.error(f"Error stopping data collection: {e}")
            raise
    
    def get_historical_data(self, asset, timeframe, start_date, end_date=None):
        """
        Get historical data for an asset and timeframe
        
        Args:
            asset (str): Asset symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime, optional): End date. Defaults to None (current time).
        
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            end_date = end_date or datetime.now()
            
            # Convert dates to string format
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Create asset directory path
            asset_dir = DATA_DIR / asset.replace("/", "_")
            
            # Get all CSV files for the timeframe
            csv_files = list(asset_dir.glob(f"{timeframe}_*.csv"))
            
            # Read and combine all data within the date range
            dfs = []
            for file_path in csv_files:
                # Extract date from filename
                file_date_str = file_path.stem.split("_")[1]
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                
                # Check if file is within the date range
                if start_date.date() <= file_date.date() <= end_date.date():
                    df = pd.read_csv(file_path)
                    
                    # Convert timestamp to datetime
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Filter by date range
                    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
                    
                    dfs.append(df)
            
            # Combine all dataframes
            if dfs:
                combined_df = pd.concat(dfs)
                
                # Sort by timestamp
                combined_df = combined_df.sort_values('timestamp')
                
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                
                return combined_df
            else:
                logger.warning(f"No data found for {asset} {timeframe} from {start_str} to {end_str}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {asset} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_pivot_points(self, data, pivot_type="standard"):
        """
        Calculate pivot points from OHLC data
        
        Args:
            data (pd.DataFrame): OHLC data
            pivot_type (str): Pivot point type ('standard' or 'fibonacci')
        
        Returns:
            pd.DataFrame: Data with pivot points
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure we have the necessary columns
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.error("Cannot calculate pivot points: missing required columns")
            return df
        
        # Shift data to get previous period values
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        # Calculate pivot point
        df['pivot'] = (prev_high + prev_low + prev_close) / 3
        
        if pivot_type == "standard":
            # Standard pivot points
            df['r1'] = 2 * df['pivot'] - prev_low
            df['s1'] = 2 * df['pivot'] - prev_high
            df['r2'] = df['pivot'] + (prev_high - prev_low)
            df['s2'] = df['pivot'] - (prev_high - prev_low)
            df['r3'] = df['pivot'] + 2 * (prev_high - prev_low)
            df['s3'] = df['pivot'] - 2 * (prev_high - prev_low)
            
        elif pivot_type == "fibonacci":
            # Fibonacci pivot points
            df['r1'] = df['pivot'] + 0.382 * (prev_high - prev_low)
            df['s1'] = df['pivot'] - 0.382 * (prev_high - prev_low)
            df['r2'] = df['pivot'] + 0.618 * (prev_high - prev_low)
            df['s2'] = df['pivot'] - 0.618 * (prev_high - prev_low)
            df['r3'] = df['pivot'] + 1.0 * (prev_high - prev_low)
            df['s3'] = df['pivot'] - 1.0 * (prev_high - prev_low)
        
        else:
            logger.warning(f"Unknown pivot type: {pivot_type}")
        
        return df
    
    def get_data_with_pivots(self, asset, timeframe, start_date, end_date=None, pivot_types=None):
        """
        Get historical data with pivot points
        
        Args:
            asset (str): Asset symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime, optional): End date. Defaults to None.
            pivot_types (list, optional): List of pivot types. Defaults to None.
        
        Returns:
            pd.DataFrame: Historical data with pivot points
        """
        # Get base data
        df = self.get_historical_data(asset, timeframe, start_date, end_date)
        
        if df.empty:
            return df
        
        # Calculate pivot points for each type
        pivot_types = pivot_types or ["standard", "fibonacci"]
        
        for pivot_type in pivot_types:
            pivot_df = self.calculate_pivot_points(df, pivot_type)
            
            # Add suffix to pivot columns
            pivot_cols = [col for col in pivot_df.columns if col.startswith(('pivot', 'r', 's'))]
            pivot_df = pivot_df.rename(columns={col: f"{col}_{pivot_type}" for col in pivot_cols})
            
            # Update the original dataframe with the new pivot columns
            for col in [c for c in pivot_df.columns if c.endswith(f"_{pivot_type}")]:
                df[col] = pivot_df[col]
        
        return df


# Example usage
if __name__ == "__main__":
    collector = TradingViewDataCollector()
    
    try:
        # Start data collection
        collector.start_data_collection()
        
        # Run for a while
        print("Data collection running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stopping data collection...")
    
    finally:
        # Stop data collection
        collector.stop_data_collection() 