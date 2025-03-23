import logging
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import json
import scipy.stats as stats
from scipy.fft import fft, ifft, fftfreq

from config import RISK_CONFIG, TRADING_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/risk_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RiskManagement")

class RiskManager:
    """
    Risk management system for the trading bot
    """
    def __init__(self):
        """Initialize the risk manager"""
        self.atr_multiplier = RISK_CONFIG.get("atr_multiplier", 2.5)
        self.min_profit_ratio = RISK_CONFIG.get("min_profit_ratio", 1.5)
        self.liquidity_threshold = RISK_CONFIG.get("liquidity_threshold", 0.3)
        self.max_adverse_timeframes = RISK_CONFIG.get("max_adverse_timeframes", 1)
        self.volatility_lookback = RISK_CONFIG.get("volatility_lookback", 20)
        self.max_risk_per_trade = TRADING_CONFIG.get("risk_per_trade", 0.02)  # 2% of portfolio
        
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            data (pd.DataFrame): OHLC data
            period (int): Period for ATR calculation
            
        Returns:
            pd.Series: ATR values
        """
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure we have the necessary columns
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.error("Cannot calculate ATR: missing required columns")
                return pd.Series()
            
            # Calculate True Range
            df['tr0'] = df['high'] - df['low']
            df['tr1'] = abs(df['high'] - df['close'].shift())
            df['tr2'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            
            # Calculate ATR
            df['atr'] = df['tr'].rolling(period).mean()
            
            return df['atr']
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    def calculate_stop_loss(self, data, entry_price, direction, price_key='close'):
        """
        Calculate dynamic stop loss based on ATR
        
        Args:
            data (pd.DataFrame): OHLC data
            entry_price (float): Entry price
            direction (int): Trade direction (1 for long, -1 for short)
            price_key (str): Price column to use
            
        Returns:
            float: Stop loss price
        """
        try:
            # Calculate ATR
            atr = self.calculate_atr(data, self.volatility_lookback)
            
            if atr.empty:
                logger.warning("ATR calculation failed, using fixed stop loss")
                # Fallback to fixed percentage stop loss (2%)
                return entry_price * (1 - 0.02 * direction)
            
            # Get latest ATR value
            latest_atr = atr.iloc[-1]
            
            # Calculate stop loss based on ATR and direction
            if direction > 0:  # Long position
                stop_loss = entry_price - (latest_atr * self.atr_multiplier)
            else:  # Short position
                stop_loss = entry_price + (latest_atr * self.atr_multiplier)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Fallback to fixed percentage stop loss (2%)
            return entry_price * (1 - 0.02 * direction)
    
    def calculate_take_profit(self, entry_price, stop_loss, direction):
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            direction (int): Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Take profit price
        """
        try:
            # Calculate risk (distance from entry to stop loss)
            risk = abs(entry_price - stop_loss)
            
            # Calculate reward based on min profit ratio
            reward = risk * self.min_profit_ratio
            
            # Calculate take profit based on direction
            if direction > 0:  # Long position
                take_profit = entry_price + reward
            else:  # Short position
                take_profit = entry_price - reward
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            # Fallback to fixed percentage take profit (3%)
            return entry_price * (1 + 0.03 * direction)
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss, 
                               market_liquidity=None, risk_score=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            account_balance (float): Account balance
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            market_liquidity (float, optional): Market liquidity score (0-1)
            risk_score (float, optional): Risk score (0-1)
            
        Returns:
            float: Position size in units
        """
        try:
            # Calculate risk amount (max amount to risk per trade)
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Calculate risk per unit (price distance to stop loss)
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                logger.warning("Risk per unit is zero, using default stop loss distance")
                risk_per_unit = entry_price * 0.02  # Default 2% stop loss
            
            # Calculate basic position size
            position_size = risk_amount / risk_per_unit
            
            # Adjust based on liquidity if available
            if market_liquidity is not None:
                # Scale position size based on liquidity (lower liquidity = smaller position)
                liquidity_factor = max(0.1, min(1.0, market_liquidity / self.liquidity_threshold))
                position_size *= liquidity_factor
            
            # Adjust based on risk score if available
            if risk_score is not None:
                # Scale position size based on risk score (higher risk = smaller position)
                risk_factor = max(0.1, min(1.0, (1 - risk_score)))
                position_size *= risk_factor
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            # Fallback to fixed position size (1% of account)
            return account_balance * 0.01 / entry_price
    
    def check_timeframe_conflicts(self, signals_dict):
        """
        Check for conflicts across different timeframes
        
        Args:
            signals_dict (dict): Dictionary with timeframes as keys and signal data as values
            
        Returns:
            dict: Conflict assessment
        """
        try:
            # Count positive and negative signals
            positive_timeframes = []
            negative_timeframes = []
            neutral_timeframes = []
            
            for timeframe, signals in signals_dict.items():
                # Get the combined signal or calculate it if not available
                if 'combined_signal' in signals:
                    signal = signals['combined_signal']
                else:
                    # Simple average of available signals
                    signal_values = [v for v in signals.values() if isinstance(v, (int, float))]
                    signal = sum(signal_values) / len(signal_values) if signal_values else 0
                
                # Classify signal
                if signal > 0.2:
                    positive_timeframes.append(timeframe)
                elif signal < -0.2:
                    negative_timeframes.append(timeframe)
                else:
                    neutral_timeframes.append(timeframe)
            
            # Check for conflicts
            has_conflict = len(positive_timeframes) > 0 and len(negative_timeframes) > 0
            
            # Calculate conflict severity
            conflict_severity = 0
            if has_conflict:
                # Severity based on number of conflicting timeframes
                conflict_severity = min(len(positive_timeframes), len(negative_timeframes)) / len(signals_dict)
            
            # Check if conflicts exceed maximum allowed
            too_many_conflicts = min(len(positive_timeframes), len(negative_timeframes)) > self.max_adverse_timeframes
            
            return {
                'has_conflict': has_conflict,
                'conflict_severity': float(conflict_severity),
                'too_many_conflicts': too_many_conflicts,
                'positive_timeframes': positive_timeframes,
                'negative_timeframes': negative_timeframes,
                'neutral_timeframes': neutral_timeframes
            }
            
        except Exception as e:
            logger.error(f"Error checking timeframe conflicts: {e}")
            return {
                'has_conflict': False,
                'conflict_severity': 0.0,
                'too_many_conflicts': False,
                'positive_timeframes': [],
                'negative_timeframes': [],
                'neutral_timeframes': list(signals_dict.keys())
            }
    
    def analyze_quantum_fourier(self, price_data, n_components=3):
        """
        Analyze price oscillation using Quantum Fourier Transform technique
        
        Args:
            price_data (pd.Series): Price series
            n_components (int): Number of frequency components to extract
            
        Returns:
            dict: Oscillation analysis
        """
        try:
            # Ensure we have enough data
            if len(price_data) < 30:
                logger.warning("Not enough data for Fourier analysis")
                return {'oscillation_score': 0.5, 'trend_direction': 0, 'frequencies': []}
            
            # Get price series
            prices = price_data.values
            n = len(prices)
            
            # Perform Fourier Transform
            price_fft = fft(prices)
            freqs = fftfreq(n)
            
            # Get the power spectrum (absolute values squared)
            power = np.abs(price_fft) ** 2
            
            # Find dominant frequencies (excluding DC component at index 0)
            dominant_indices = np.argsort(power[1:n//2])[-n_components:] + 1
            dominant_freqs = freqs[dominant_indices]
            dominant_power = power[dominant_indices]
            
            # Normalize powers to sum to 1
            norm_power = dominant_power / np.sum(dominant_power) if np.sum(dominant_power) > 0 else dominant_power
            
            # Calculate period for each frequency (in terms of number of data points)
            periods = [int(abs(1/freq)) if freq != 0 else n for freq in dominant_freqs]
            
            # Reconstruct signal using only dominant frequencies
            reconstructed = np.zeros_like(prices, dtype=complex)
            for idx in dominant_indices:
                reconstructed += price_fft[idx] * np.exp(2j * np.pi * freqs[idx] * np.arange(n))
            reconstructed = np.real(reconstructed) / n
            
            # Calculate residuals (original - reconstructed)
            residuals = prices - reconstructed
            
            # Calculate oscillation score based on dominant frequency power
            # High oscillation score means strong cyclic patterns
            oscillation_score = min(1.0, np.sum(norm_power) / 0.8)
            
            # Determine trend direction using slope of reconstructed signal
            recent_window = min(20, n // 4)
            trend = np.polyfit(np.arange(recent_window), reconstructed[-recent_window:], 1)[0]
            trend_direction = 1 if trend > 0 else (-1 if trend < 0 else 0)
            
            # Determine if we're in a high or low phase of the cycle
            cycle_position = np.mean(residuals[-recent_window:])
            phase = 'high' if cycle_position > 0 else 'low'
            
            # Format frequencies for output
            freq_info = []
            for i, (freq, period, power) in enumerate(zip(dominant_freqs, periods, norm_power)):
                freq_info.append({
                    'frequency': float(freq),
                    'period': int(period),
                    'power': float(power),
                    'phase': 'high' if np.real(price_fft[dominant_indices[i]]) > 0 else 'low'
                })
            
            return {
                'oscillation_score': float(oscillation_score),
                'trend_direction': int(trend_direction),
                'cycle_phase': phase,
                'frequencies': freq_info
            }
            
        except Exception as e:
            logger.error(f"Error in Fourier analysis: {e}")
            return {'oscillation_score': 0.5, 'trend_direction': 0, 'frequencies': []}
    
    def filter_signal_by_oscillation(self, signal_value, oscillation_analysis):
        """
        Filter trading signal based on oscillation analysis
        
        Args:
            signal_value (float): Original signal value (-1 to 1)
            oscillation_analysis (dict): Output from analyze_quantum_fourier
            
        Returns:
            float: Filtered signal value
        """
        try:
            oscillation_score = oscillation_analysis.get('oscillation_score', 0.5)
            trend_direction = oscillation_analysis.get('trend_direction', 0)
            cycle_phase = oscillation_analysis.get('cycle_phase', 'neutral')
            
            # If oscillation is weak, trust the original signal
            if oscillation_score < 0.3:
                return signal_value
            
            # If oscillation is strong, adjust signal based on phase and trend
            filtered_signal = signal_value
            
            # Enhance signal if it aligns with trend direction
            if (signal_value > 0 and trend_direction > 0) or (signal_value < 0 and trend_direction < 0):
                filtered_signal = signal_value * 1.2  # Enhance by 20%
            
            # Reduce signal if it contradicts cycle phase
            if (signal_value > 0 and cycle_phase == 'high') or (signal_value < 0 and cycle_phase == 'low'):
                filtered_signal = signal_value * 0.7  # Reduce by 30%
            
            # Enhance signal if it aligns with cycle phase
            if (signal_value > 0 and cycle_phase == 'low') or (signal_value < 0 and cycle_phase == 'high'):
                filtered_signal = signal_value * 1.3  # Enhance by 30%
            
            # Limit to -1 to 1 range
            filtered_signal = max(min(filtered_signal, 1.0), -1.0)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error filtering signal by oscillation: {e}")
            return signal_value
    
    def should_close_in_high_volatility(self, open_positions, current_market_data):
        """
        Determine if positions should be closed during high volatility periods
        
        Args:
            open_positions (list): List of open positions
            current_market_data (pd.DataFrame): Current market data
            
        Returns:
            dict: Positions that should be closed with reasons
        """
        try:
            positions_to_close = {}
            
            # Calculate current volatility (ATR relative to historical)
            atr = self.calculate_atr(current_market_data, self.volatility_lookback)
            
            if atr.empty or len(atr) < self.volatility_lookback * 2:
                logger.warning("Not enough data to assess volatility")
                return positions_to_close
            
            # Calculate volatility ratio (current ATR vs historical average)
            current_atr = atr.iloc[-1]
            historical_atr = atr.iloc[-self.volatility_lookback*2:-self.volatility_lookback].mean()
            volatility_ratio = current_atr / historical_atr if historical_atr > 0 else 1.0
            
            # Check each position
            for position in open_positions:
                position_id = position.get('id', 'unknown')
                asset = position.get('asset', 'unknown')
                direction = position.get('direction', 0)
                entry_price = position.get('entry_price', 0)
                current_price = current_market_data['close'].iloc[-1]
                
                # Check if volatility is extremely high (3x normal)
                if volatility_ratio > 3.0:
                    positions_to_close[position_id] = {
                        'reason': 'extreme_volatility',
                        'volatility_ratio': float(volatility_ratio)
                    }
                    continue
                
                # Check if position is profitable and volatility is rising
                is_profitable = (current_price > entry_price and direction > 0) or \
                               (current_price < entry_price and direction < 0)
                
                volatility_increasing = volatility_ratio > 1.5
                
                if is_profitable and volatility_increasing:
                    # Calculate profit as percentage
                    profit_pct = abs(current_price - entry_price) / entry_price * 100
                    
                    # Close profitable positions in increasing volatility if profit > 3%
                    if profit_pct > 3.0:
                        positions_to_close[position_id] = {
                            'reason': 'lock_profits_in_volatility',
                            'profit_percentage': float(profit_pct),
                            'volatility_ratio': float(volatility_ratio)
                        }
            
            return positions_to_close
            
        except Exception as e:
            logger.error(f"Error checking volatility-based position closure: {e}")
            return {}
    
    def adjust_for_timeframe_conflict(self, position_size, conflict_assessment):
        """
        Adjust position size based on timeframe conflicts
        
        Args:
            position_size (float): Original position size
            conflict_assessment (dict): Output from check_timeframe_conflicts
            
        Returns:
            float: Adjusted position size
        """
        try:
            # If no conflict or severity is low, return original position size
            if not conflict_assessment.get('has_conflict', False) or \
               conflict_assessment.get('conflict_severity', 0) < 0.3:
                return position_size
            
            # If too many conflicts, return zero (don't take the trade)
            if conflict_assessment.get('too_many_conflicts', False):
                logger.info("Too many timeframe conflicts, recommending no position")
                return 0
            
            # Otherwise, reduce position size based on conflict severity
            severity = conflict_assessment.get('conflict_severity', 0)
            reduction_factor = 1 - (severity * 0.8)  # 0.8 means 80% reduction at maximum severity
            
            adjusted_size = position_size * reduction_factor
            
            logger.info(f"Adjusted position size for timeframe conflict: {position_size} -> {adjusted_size}")
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error adjusting for timeframe conflict: {e}")
            return position_size * 0.5  # 50% reduction as fallback
    
    def get_risk_parameters(self, data, entry_price, direction, account_balance, 
                           signals_dict=None, liquidity=None, risk_score=None):
        """
        Get comprehensive risk parameters for a trade
        
        Args:
            data (pd.DataFrame): OHLC data
            entry_price (float): Entry price
            direction (int): Trade direction (1 for long, -1 for short)
            account_balance (float): Account balance
            signals_dict (dict, optional): Dictionary with timeframes as keys and signal data as values
            liquidity (float, optional): Liquidity score (0-1)
            risk_score (float, optional): Risk score (0-1)
            
        Returns:
            dict: Risk parameters
        """
        try:
            # Calculate stop loss
            stop_loss = self.calculate_stop_loss(data, entry_price, direction)
            
            # Calculate take profit
            take_profit = self.calculate_take_profit(entry_price, stop_loss, direction)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                account_balance, entry_price, stop_loss, liquidity, risk_score
            )
            
            # Check for timeframe conflicts if signals are provided
            conflict_assessment = None
            if signals_dict:
                conflict_assessment = self.check_timeframe_conflicts(signals_dict)
                
                # Adjust position size for conflicts
                position_size = self.adjust_for_timeframe_conflict(position_size, conflict_assessment)
            
            # Analyze price oscillation using Fourier analysis
            oscillation_analysis = None
            if 'close' in data.columns and len(data) >= 30:
                oscillation_analysis = self.analyze_quantum_fourier(data['close'])
            
            # Format position size as percentage
            position_pct = (position_size * entry_price / account_balance) * 100
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Calculate expected value
            # Assuming win rate based on signal strength and conflict
            win_rate = 0.5  # Default
            
            if signals_dict and 'consolidated' in signals_dict:
                # Adjust win rate based on signal strength
                signal_strength = abs(signals_dict['consolidated'].get('combined_signal', 0))
                win_rate = 0.3 + (signal_strength * 0.4)  # 0.3 to 0.7 based on signal strength
                
                # Adjust for conflicts
                if conflict_assessment and conflict_assessment.get('has_conflict', False):
                    win_rate *= (1 - conflict_assessment.get('conflict_severity', 0) * 0.5)
            
            expected_value = (win_rate * reward) - ((1 - win_rate) * risk)
            expected_value_pct = (expected_value / entry_price) * 100
            
            # Calculate pyramiding levels (if trending in our direction)
            # Only suggest pyramiding if risk-reward is good and no severe conflicts
            pyramiding_levels = []
            
            if risk_reward_ratio >= 2.0 and (not conflict_assessment or 
                                           not conflict_assessment.get('too_many_conflicts', False)):
                # Calculate 2-3 pyramiding levels
                level_count = 2
                for i in range(1, level_count + 1):
                    # For long positions, pyramid on the way up
                    # For short positions, pyramid on the way down
                    level_price = entry_price * (1 + (0.01 * i * direction))
                    pyramiding_levels.append(float(level_price))
            
            # Calculate confidence interval
            confidence_margin = 0.02  # 2% margin
            if risk_score:
                # Adjust margin based on risk score
                confidence_margin = 0.01 + (risk_score * 0.04)  # 1% to 5%
            
            confidence_interval = [
                float(entry_price * (1 - confidence_margin)),
                float(entry_price * (1 + confidence_margin))
            ]
            
            return {
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'position_size': float(position_size),
                'position_size_pct': f"{position_pct:.2f}%",
                'risk_reward_ratio': float(risk_reward_ratio),
                'expected_value_pct': float(expected_value_pct),
                'win_rate': float(win_rate),
                'pyramiding_levels': pyramiding_levels,
                'confidence_interval': confidence_interval,
                'timeframe_conflicts': conflict_assessment,
                'oscillation_analysis': oscillation_analysis
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk parameters: {e}")
            # Return basic parameters as fallback
            return {
                'entry_price': float(entry_price),
                'stop_loss': float(entry_price * (1 - 0.02 * direction)),
                'take_profit': float(entry_price * (1 + 0.03 * direction)),
                'position_size': float(account_balance * 0.01 / entry_price),
                'position_size_pct': "1.00%",
                'risk_reward_ratio': 1.5,
                'confidence_interval': [
                    float(entry_price * 0.98),
                    float(entry_price * 1.02)
                ]
            }


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    # Create sample data
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.normal(100, 1, 30),
        'high': np.random.normal(101, 1, 30),
        'low': np.random.normal(99, 1, 30),
        'close': np.random.normal(100, 1, 30),
        'volume': np.random.normal(1000, 100, 30)
    })
    
    # Make high/low values consistent
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'close']) + abs(np.random.normal(0, 0.5))
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'close']) - abs(np.random.normal(0, 0.5))
    
    # Create sample signals
    signals_dict = {
        '1h': {'rsi_divergence': 0.5, 'macd_divergence': 0.7, 'combined_signal': 0.6},
        '4h': {'rsi_divergence': 0.3, 'macd_divergence': 0.2, 'combined_signal': 0.25},
        '1d': {'rsi_divergence': -0.2, 'macd_divergence': -0.1, 'combined_signal': -0.15}
    }
    
    # Create risk manager
    risk_manager = RiskManager()
    
    # Test risk parameters
    entry_price = data['close'].iloc[-1]
    direction = 1  # Long
    account_balance = 10000
    
    risk_params = risk_manager.get_risk_parameters(
        data, entry_price, direction, account_balance, signals_dict, 0.6, 0.4
    )
    
    print(json.dumps(risk_params, indent=2)) 