import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import json

from config import DATA_DIR, KNN_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pine_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PineAnalysis")

class PineScriptAnalysisEngine:
    """
    PineScript Analysis Engine that processes market data and generates trading signals
    using algorithms similar to those found in PineScript.
    """
    def __init__(self):
        """Initialize the Pine Script Analysis Engine"""
        self.knn_models = {}
        self.scalers = {}
        
    def calculate_divergence(self, price_data, indicator_data, window=14):
        """
        Calculate divergence between price and indicator
        
        Args:
            price_data (pd.Series): Price data series
            indicator_data (pd.Series): Indicator data series
            window (int): Window size for divergence calculation
            
        Returns:
            pd.Series: Divergence series (1 for bullish, -1 for bearish, 0.5 for hidden bullish, -0.5 for hidden bearish, 0 for none)
        """
        # Make a copy to avoid modifying the original
        price = price_data.copy()
        indicator = indicator_data.copy()
        
        # Calculate local minima and maxima for price
        price_min = price.rolling(window=window, center=True).min()
        price_max = price.rolling(window=window, center=True).max()
        
        # Calculate local minima and maxima for indicator
        indicator_min = indicator.rolling(window=window, center=True).min()
        indicator_max = indicator.rolling(window=window, center=True).max()
        
        # Initialize divergence series
        divergence = pd.Series(0, index=price.index)
        
        # Regular bullish divergence: Price makes lower low, indicator makes higher low
        bullish_div = (price == price_min) & (price < price.shift(window//2)) & (indicator > indicator.shift(window//2))
        divergence[bullish_div] = 1
        
        # Regular bearish divergence: Price makes higher high, indicator makes lower high
        bearish_div = (price == price_max) & (price > price.shift(window//2)) & (indicator < indicator.shift(window//2))
        divergence[bearish_div] = -1
        
        # Hidden bullish divergence: Price makes higher low, indicator makes lower low
        hidden_bullish = (price > price.shift(window//2)) & (indicator < indicator.shift(window//2)) & (indicator == indicator_min)
        divergence[hidden_bullish] = 0.5
        
        # Hidden bearish divergence: Price makes lower high, indicator makes higher high
        hidden_bearish = (price < price.shift(window//2)) & (indicator > indicator.shift(window//2)) & (indicator == indicator_max)
        divergence[hidden_bearish] = -0.5
        
        return divergence
    
    def analyze_multi_timeframe_divergences(self, data_dict):
        """
        Analyze divergences across multiple timeframes
        
        Args:
            data_dict (dict): Dictionary with timeframes as keys and dataframes as values
            
        Returns:
            dict: Divergence analysis results for each timeframe
        """
        results = {}
        
        for timeframe, df in data_dict.items():
            timeframe_results = {}
            
            # RSI Divergence
            if 'rsi' in df.columns and 'close' in df.columns:
                timeframe_results['rsi_divergence'] = self.calculate_divergence(df['close'], df['rsi'])
            
            # MACD Divergence
            if 'macd_macd' in df.columns and 'close' in df.columns:
                timeframe_results['macd_divergence'] = self.calculate_divergence(df['close'], df['macd_macd'])
            
            # Bollinger Bands %B - distance from lower band in % of total band width
            if all(col in df.columns for col in ['close', 'bollinger_upper', 'bollinger_lower']):
                bb_width = df['bollinger_upper'] - df['bollinger_lower']
                bb_percent_b = (df['close'] - df['bollinger_lower']) / bb_width
                timeframe_results['bb_percent_b'] = bb_percent_b
                
                # Bollinger squeeze - band width normalized by price
                bb_squeeze = bb_width / df['close']
                timeframe_results['bb_squeeze'] = bb_squeeze
                
                # Bollinger divergence
                timeframe_results['bb_divergence'] = self.calculate_divergence(df['close'], bb_percent_b)
            
            results[timeframe] = timeframe_results
        
        return results
    
    def consolidate_signals(self, signals_dict, weighting=None):
        """
        Consolidate signals from multiple timeframes
        
        Args:
            signals_dict (dict): Dictionary with timeframes as keys and signal data as values
            weighting (dict, optional): Dictionary with timeframes as keys and weights as values
            
        Returns:
            dict: Consolidated signals
        """
        # Default weighting - higher weights for longer timeframes
        if weighting is None:
            weighting = {
                "1h": 1.0,
                "4h": 2.0,
                "1d": 3.0
            }
        
        # Normalize weights
        total_weight = sum(weighting.values())
        normalized_weights = {tf: w/total_weight for tf, w in weighting.items()}
        
        # Initialize consolidated signals
        consolidated = {
            "rsi_divergence": 0,
            "macd_divergence": 0,
            "bb_divergence": 0,
            "bb_squeeze": 0,
            "pivot_breakout": 0,
            "combined_signal": 0
        }
        
        # Combine signals with weights
        for timeframe, signals in signals_dict.items():
            if timeframe not in normalized_weights:
                continue
                
            weight = normalized_weights[timeframe]
            
            # Add weighted signals
            for signal_type in consolidated.keys():
                if signal_type in signals:
                    consolidated[signal_type] += signals[signal_type] * weight
        
        # Calculate combined signal
        signal_weights = {
            "rsi_divergence": 0.3,
            "macd_divergence": 0.3,
            "bb_divergence": 0.2,
            "bb_squeeze": 0.1,
            "pivot_breakout": 0.1
        }
        
        combined_signal = 0
        for signal_type, weight in signal_weights.items():
            combined_signal += consolidated[signal_type] * weight
        
        consolidated["combined_signal"] = combined_signal
        
        return consolidated
    
    def train_knn_pivot_breakout(self, data, asset, timeframe, n_neighbors=5, retrain=False):
        """
        Train KNN model for pivot breakout detection
        
        Args:
            data (pd.DataFrame): Historical data with pivot points
            asset (str): Asset symbol
            timeframe (str): Timeframe
            n_neighbors (int): Number of neighbors for KNN
            retrain (bool): Whether to retrain the model if it already exists
            
        Returns:
            sklearn.neighbors.KNeighborsClassifier: Trained KNN model
        """
        model_key = f"{asset}_{timeframe}"
        
        # Check if model already exists and we don't want to retrain
        if model_key in self.knn_models and not retrain:
            logger.info(f"Using existing KNN model for {asset} {timeframe}")
            return self.knn_models[model_key]
        
        logger.info(f"Training KNN pivot breakout model for {asset} {timeframe}")
        
        try:
            # Prepare features - using pivot points and technical indicators
            pivot_cols = [col for col in data.columns if 'pivot' in col or col.startswith(('r', 's')) and '_' in col]
            indicator_cols = [col for col in data.columns if any(ind in col for ind in ['rsi', 'macd', 'bollinger'])]
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            
            feature_cols = pivot_cols + indicator_cols
            
            if not feature_cols:
                logger.error(f"No feature columns found for {asset} {timeframe}")
                return None
            
            # Drop rows with NaN values
            clean_data = data.dropna(subset=feature_cols + price_cols).copy()
            
            if len(clean_data) < 30:
                logger.warning(f"Not enough data to train KNN model for {asset} {timeframe}")
                return None
            
            # Create features
            X = clean_data[feature_cols].values
            
            # Create labels - 1 for uptrend (close > pivot), -1 for downtrend (close < pivot)
            # Looking at next period's close for prediction
            y = np.zeros(len(clean_data))
            
            # Find closest pivot level
            for i in range(len(clean_data) - 1):  # Skip last row as we need the next period's close
                row = clean_data.iloc[i]
                next_close = clean_data.iloc[i+1]['close']
                
                # Get pivot levels
                pivot_levels = {}
                for col in pivot_cols:
                    pivot_levels[col] = row[col]
                
                # Find closest pivot level that was broken
                closest_level = None
                min_distance = float('inf')
                
                for level_name, level_value in pivot_levels.items():
                    if pd.isna(level_value):
                        continue
                        
                    distance = abs(row['close'] - level_value)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_level = (level_name, level_value)
                
                if closest_level:
                    level_name, level_value = closest_level
                    
                    # Check if the next close breaks through this level
                    if "r" in level_name and next_close > level_value:  # Resistance breakout
                        y[i] = 1
                    elif "s" in level_name and next_close < level_value:  # Support breakdown
                        y[i] = -1
                    elif "pivot" in level_name:
                        if next_close > level_value and row['close'] < level_value:  # Pivot breakout
                            y[i] = 1
                        elif next_close < level_value and row['close'] > level_value:  # Pivot breakdown
                            y[i] = -1
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train KNN model
            knn = KNeighborsClassifier(
                n_neighbors=KNN_CONFIG["n_neighbors"],
                weights=KNN_CONFIG["weights"],
                metric=KNN_CONFIG["distance_metric"]
            )
            
            # Fit to all but the last data point (which has no label)
            knn.fit(X_scaled[:-1], y[:-1])
            
            # Save model and scaler
            self.knn_models[model_key] = knn
            self.scalers[model_key] = scaler
            
            logger.info(f"KNN model trained for {asset} {timeframe}")
            return knn
            
        except Exception as e:
            logger.error(f"Error training KNN model for {asset} {timeframe}: {e}")
            return None
    
    def detect_pivot_breakout(self, data, asset, timeframe):
        """
        Detect pivot breakouts using KNN model
        
        Args:
            data (pd.DataFrame): Current market data
            asset (str): Asset symbol
            timeframe (str): Timeframe
            
        Returns:
            float: Breakout signal (1 for bullish breakout, -1 for bearish breakdown, 0 for none)
        """
        model_key = f"{asset}_{timeframe}"
        
        # Check if model exists
        if model_key not in self.knn_models or model_key not in self.scalers:
            logger.warning(f"No KNN model found for {asset} {timeframe}")
            return 0
        
        try:
            # Get model and scaler
            knn = self.knn_models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare features
            pivot_cols = [col for col in data.columns if 'pivot' in col or col.startswith(('r', 's')) and '_' in col]
            indicator_cols = [col for col in data.columns if any(ind in col for ind in ['rsi', 'macd', 'bollinger'])]
            
            feature_cols = pivot_cols + indicator_cols
            
            if not feature_cols:
                logger.error(f"No feature columns found for {asset} {timeframe}")
                return 0
            
            # Get latest data point
            latest_data = data.iloc[-1]
            
            # Extract features
            X = np.array([latest_data[feature_cols].values])
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Predict
            probabilities = knn.predict_proba(X_scaled)
            
            # Convert to signal (-1 to 1)
            # Assuming classes are [0, -1, 1] or [0, 1, -1] - adjust based on your model
            if len(knn.classes_) == 2:
                # Binary classification
                prediction = knn.predict(X_scaled)[0]
                signal = prediction  # Assuming prediction is already -1 or 1
            else:
                # Multi-class with probabilities
                class_dict = {c: i for i, c in enumerate(knn.classes_)}
                
                if 1 in class_dict and -1 in class_dict:
                    bullish_prob = probabilities[0][class_dict[1]]
                    bearish_prob = probabilities[0][class_dict[-1]]
                    
                    # Convert to signal (-1 to 1)
                    signal = bullish_prob - bearish_prob
                else:
                    # Fallback to prediction
                    prediction = knn.predict(X_scaled)[0]
                    signal = prediction
            
            logger.info(f"Pivot breakout signal for {asset} {timeframe}: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error detecting pivot breakout for {asset} {timeframe}: {e}")
            return 0
    
    def analyze_volume_weighted_pivots(self, data, volume_window=5):
        """
        Analyze volume-weighted pivot points
        
        Args:
            data (pd.DataFrame): Market data with pivot points
            volume_window (int): Window size for volume weighting
            
        Returns:
            pd.DataFrame: Data with volume-weighted pivot confirmations
        """
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure we have the necessary columns
            if 'volume' not in df.columns:
                logger.error("Cannot analyze volume-weighted pivots: missing volume data")
                return df
            
            # Get pivot columns
            pivot_cols = [col for col in df.columns if 'pivot' in col or col.startswith(('r', 's')) and '_' in col]
            
            if not pivot_cols:
                logger.error("Cannot analyze volume-weighted pivots: no pivot columns found")
                return df
            
            # Calculate volume relative to moving average
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=volume_window).mean()
            
            # Initialize pivot confirmation columns
            for col in pivot_cols:
                df[f"{col}_vol_conf"] = 0
            
            # Check for price crossing pivot levels with high volume
            for i in range(1, len(df)):
                curr_row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                for col in pivot_cols:
                    pivot_value = prev_row[col]
                    
                    if pd.isna(pivot_value):
                        continue
                    
                    # Check if price crossed the pivot level
                    if (prev_row['close'] < pivot_value and curr_row['close'] > pivot_value) or \
                       (prev_row['close'] > pivot_value and curr_row['close'] < pivot_value):
                        
                        # Weight by volume ratio
                        volume_weight = curr_row['volume_ratio']
                        
                        # Sign of confirmation (1 for bullish, -1 for bearish)
                        direction = 1 if curr_row['close'] > pivot_value else -1
                        
                        # Set confirmation value
                        df.at[i, f"{col}_vol_conf"] = direction * volume_weight
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing volume-weighted pivots: {e}")
            return data
    
    def process_data(self, data_dict, asset):
        """
        Process market data and generate trading signals
        
        Args:
            data_dict (dict): Dictionary with timeframes as keys and dataframes as values
            asset (str): Asset symbol
            
        Returns:
            dict: Trading signals
        """
        signals = {}
        
        # Process each timeframe
        for timeframe, data in data_dict.items():
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure we have enough data
            if len(df) < 30:
                logger.warning(f"Not enough data for {asset} {timeframe}")
                continue
            
            # 1. Train KNN model for pivot breakout detection (if needed)
            self.train_knn_pivot_breakout(df, asset, timeframe)
            
            # 2. Detect pivot breakouts
            pivot_breakout_signal = self.detect_pivot_breakout(df, asset, timeframe)
            
            # 3. Analyze volume-weighted pivots
            df_with_vol_pivots = self.analyze_volume_weighted_pivots(df)
            
            # 4. Calculate divergences for technical indicators
            divergences = {}
            
            # RSI Divergence
            if 'rsi' in df.columns and 'close' in df.columns:
                divergences['rsi_divergence'] = self.calculate_divergence(df['close'], df['rsi']).iloc[-1]
            
            # MACD Divergence
            if 'macd_macd' in df.columns and 'close' in df.columns:
                divergences['macd_divergence'] = self.calculate_divergence(df['close'], df['macd_macd']).iloc[-1]
            
            # Bollinger Bands divergence and squeeze
            if all(col in df.columns for col in ['close', 'bollinger_upper', 'bollinger_lower']):
                bb_width = df['bollinger_upper'] - df['bollinger_lower']
                bb_percent_b = (df['close'] - df['bollinger_lower']) / bb_width
                
                # Bollinger divergence
                divergences['bb_divergence'] = self.calculate_divergence(df['close'], bb_percent_b).iloc[-1]
                
                # Bollinger squeeze - band width normalized by price
                bb_squeeze = bb_width / df['close']
                
                # Detect squeeze (low volatility)
                squeeze_percentile = bb_squeeze.iloc[-1] / bb_squeeze.rolling(window=50).max().iloc[-1]
                divergences['bb_squeeze'] = 1 if squeeze_percentile < 0.3 else 0
            
            # 5. Volume-weighted pivot confirmations
            vol_conf_cols = [col for col in df_with_vol_pivots.columns if col.endswith('_vol_conf')]
            if vol_conf_cols:
                # Get the latest values
                vol_confirmations = df_with_vol_pivots[vol_conf_cols].iloc[-1].values
                
                # Calculate average confirmation value (positive = bullish, negative = bearish)
                avg_confirmation = np.nanmean(vol_confirmations) if vol_confirmations.size > 0 else 0
                divergences['pivot_vol_conf'] = avg_confirmation
            
            # Store signals for this timeframe
            signals[timeframe] = {
                "pivot_breakout": pivot_breakout_signal,
                **divergences
            }
        
        # Consolidate signals from all timeframes
        consolidated_signals = self.consolidate_signals(signals)
        
        return {
            "per_timeframe": signals,
            "consolidated": consolidated_signals
        }


# Example usage
if __name__ == "__main__":
    from data_collector import TradingViewDataCollector
    from datetime import datetime, timedelta
    
    # Create data collector
    collector = TradingViewDataCollector(assets=["BTC/USD"], timeframes=["1h", "4h", "1d"])
    
    # Get data for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data_dict = {}
    for timeframe in collector.timeframes:
        data = collector.get_data_with_pivots("BTC/USD", timeframe, start_date, end_date)
        if not data.empty:
            data_dict[timeframe] = data
    
    # Create analysis engine
    engine = PineScriptAnalysisEngine()
    
    # Process data
    signals = engine.process_data(data_dict, "BTC/USD")
    
    # Print results
    print(json.dumps(signals, indent=2)) 