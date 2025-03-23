import logging
import pandas as pd
import numpy as np
import json
import os
import time
import schedule
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
import shap
import traceback
import requests

# Import custom modules
from data_collector import TradingViewDataCollector
from pine_analysis import PineScriptAnalysisEngine
from models.price_prediction import LSTMCNNHybridModel
from models.sentiment_analysis import DualSentimentAnalyzer
from models.ensemble_meta_learner import EnsembleMetaLearner
from risk_management import RiskManager

from config import (
    TRADING_CONFIG, DATA_DIR, MODELS_DIR, LOGS_DIR, BACKTEST_DIR,
    TELEGRAM_CONFIG, WHATSAPP_CONFIG, MONGO_CONFIG
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

class TradingBot:
    """
    Main trading bot that integrates all components
    """
    def __init__(self, assets=None, timeframes=None, backtest_mode=False):
        """
        Initialize the trading bot
        
        Args:
            assets (list, optional): List of assets to trade
            timeframes (list, optional): List of timeframes to analyze
            backtest_mode (bool, optional): Whether to run in backtest mode
        """
        self.assets = assets or TRADING_CONFIG["assets"]
        self.timeframes = timeframes or TRADING_CONFIG["timeframes"]
        self.backtest_mode = backtest_mode
        self.running = False
        self.account_balance = 10000  # Default starting balance
        self.open_positions = {}  # Dictionary to track open positions
        self.signal_queue = queue.Queue()  # Queue for trading signals
        
        # Initialize components
        logger.info("Initializing trading bot components...")
        
        # Data collector
        self.data_collector = TradingViewDataCollector(self.assets, self.timeframes)
        
        # Analysis engine
        self.analysis_engine = PineScriptAnalysisEngine()
        
        # Models
        self.price_models = {}
        self.sentiment_analyzers = {}
        self.ensemble_models = {}
        
        # Risk manager
        self.risk_manager = RiskManager()
        
        # Initialize models for each asset
        for asset in self.assets:
            self.price_models[asset] = LSTMCNNHybridModel(asset, self.timeframes)
            self.sentiment_analyzers[asset] = DualSentimentAnalyzer(asset)
            self.ensemble_models[asset] = EnsembleMetaLearner(asset)
        
        logger.info("Trading bot initialized")
    
    def _fetch_latest_data(self, asset, lookback_days=30):
        """
        Fetch latest data for an asset
        
        Args:
            asset (str): Asset symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            dict: Dictionary with timeframes as keys and dataframes as values
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            data_dict = {}
            for timeframe in self.timeframes:
                # Get data with pivot points
                data = self.data_collector.get_data_with_pivots(
                    asset, timeframe, start_date, end_date
                )
                
                if not data.empty:
                    data_dict[timeframe] = data
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error fetching data for {asset}: {e}")
            return {}
    
    def _fetch_market_news(self, asset, max_items=20):
        """
        Fetch latest news for an asset
        
        Args:
            asset (str): Asset symbol
            max_items (int): Maximum number of news items to fetch
            
        Returns:
            list: List of news items
        """
        try:
            # For demonstration, return sample news
            # In a real implementation, this would connect to a news API
            base_asset = asset.split('/')[0]  # e.g., BTC from BTC/USD
            
            return [
                {
                    'title': f'{base_asset} market shows signs of recovery',
                    'content': f'The {base_asset} market has been showing signs of recovery after recent volatility.',
                    'source': 'sample_source',
                    'published_at': datetime.now().isoformat()
                },
                {
                    'title': f'Analysts remain bullish on {base_asset}',
                    'content': f'Despite market turbulence, analysts maintain a bullish outlook on {base_asset}.',
                    'source': 'sample_source',
                    'published_at': (datetime.now() - timedelta(hours=6)).isoformat()
                }
            ]
            
        except Exception as e:
            logger.error(f"Error fetching news for {asset}: {e}")
            return []
    
    def _analyze_asset(self, asset):
        """
        Perform comprehensive analysis on an asset
        
        Args:
            asset (str): Asset symbol
            
        Returns:
            dict: Analysis results
        """
        try:
            logger.info(f"Analyzing {asset}...")
            
            # Fetch latest data
            data_dict = self._fetch_latest_data(asset)
            
            if not data_dict:
                logger.warning(f"No data available for {asset}")
                return None
            
            # 1. Technical Analysis (Pine Script Engine)
            technical_signals = self.analysis_engine.process_data(data_dict, asset)
            
            # 2. Price Prediction (LSTM+CNN)
            price_model = self.price_models[asset]
            price_predictions = price_model.predict(data_dict)
            
            # 3. Sentiment Analysis
            news_items = self._fetch_market_news(asset)
            sentiment_analyzer = self.sentiment_analyzers[asset]
            sentiment_impact = sentiment_analyzer.get_sentiment_impact_score(news_items)
            
            # 4. Combine all features for ensemble model
            features = {}
            
            # Add price predictions
            if price_predictions:
                for horizon, price in price_predictions.items():
                    features[f"price_prediction_{horizon}"] = price
            
            # Add technical indicators from different timeframes
            if technical_signals and 'per_timeframe' in technical_signals:
                for timeframe, signals in technical_signals['per_timeframe'].items():
                    for signal_type, value in signals.items():
                        if isinstance(value, (int, float)):
                            features[f"{signal_type}_{timeframe}"] = value
            
            # Add sentiment impact
            features["sentiment_impact"] = sentiment_impact
            
            # 5. Ensemble Meta-Learner Prediction
            ensemble_model = self.ensemble_models[asset]
            composite_score = ensemble_model.predict(features)
            
            # 6. Risk Analysis
            # Get current price
            current_price = data_dict[self.timeframes[0]]['close'].iloc[-1]
            
            # Determine direction based on composite score
            direction = 1 if composite_score > 0.5 else (-1 if composite_score < 0.5 else 0)
            
            # Run Monte Carlo risk simulation
            risk_sim = ensemble_model.monte_carlo_risk_simulation(features)
            
            # Get risk parameters
            risk_parameters = self.risk_manager.get_risk_parameters(
                data_dict[self.timeframes[0]],
                current_price,
                direction,
                self.account_balance,
                technical_signals.get('per_timeframe'),
                0.7,  # Default liquidity value
                risk_sim['risk_score'] if risk_sim else 0.5
            )
            
            # 7. Get allocation recommendation
            allocation = ensemble_model.black_litterman_allocation(
                composite_score,
                risk_sim['risk_score'] if risk_sim else 0.5,
                {'sentiment_impact': sentiment_impact, 'market_depth': 0.7}
            )
            
            # 8. Get signal components with SHAP values
            signal_components = ensemble_model.get_signal_components(features)
            
            # 9. Build final analysis result
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "asset": asset,
                "composite_score": float(composite_score),
                "price_prediction": price_predictions,
                "sentiment_impact": float(sentiment_impact),
                "recommended_action": allocation['recommended_action'],
                "risk_parameters": {
                    "max_size": allocation['position_size'],
                    "dynamic_sl": float(risk_parameters['stop_loss']),
                    "take_profit": float(risk_parameters['take_profit']),
                    "pyramiding_levels": risk_parameters.get('pyramiding_levels', []),
                    "confidence_interval": risk_parameters.get('confidence_interval', [])
                },
                "signal_components": signal_components,
                "oscillation_analysis": risk_parameters.get('oscillation_analysis')
            }
            
            logger.info(f"Analysis completed for {asset}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing {asset}: {e}")
            traceback.print_exc()
            return None
    
    def _execute_signal(self, signal):
        """
        Execute a trading signal (place order or close position)
        
        Args:
            signal (dict): Trading signal
            
        Returns:
            dict: Order result
        """
        try:
            # In a real implementation, this would connect to an exchange API
            asset = signal["asset"]
            action = signal["recommended_action"]
            
            # Parse action
            is_entry = "LONG" in action or "SHORT" in action
            direction = 1 if "LONG" in action else (-1 if "SHORT" in action else 0)
            
            # Log the signal
            logger.info(f"Executing signal for {asset}: {action}")
            
            if is_entry:
                # Check if we already have a position
                if asset in self.open_positions:
                    logger.info(f"Already have position for {asset}, skipping entry")
                    return {"status": "skipped", "reason": "position_exists"}
                
                # Check for timeframe conflicts
                if "NEUTRAL" in action:
                    logger.info(f"Neutral signal for {asset}, not taking position")
                    return {"status": "skipped", "reason": "neutral_signal"}
                
                # Calculate position size
                max_size_str = signal["risk_parameters"]["max_size"].replace("%", "")
                position_pct = float(max_size_str) / 100
                position_value = self.account_balance * position_pct
                
                # Get current price
                price = signal.get("price_prediction", {}).get("1h", 0)
                if not price:
                    # Fallback to a dummy price for demonstration
                    price = 50000 if "BTC" in asset else 2000
                
                # Calculate quantity
                quantity = position_value / price
                
                # Place order
                order_id = f"order_{int(time.time())}"
                
                # Record position
                self.open_positions[asset] = {
                    "id": order_id,
                    "asset": asset,
                    "entry_price": price,
                    "quantity": quantity,
                    "direction": direction,
                    "stop_loss": signal["risk_parameters"]["dynamic_sl"],
                    "take_profit": signal["risk_parameters"]["take_profit"],
                    "entry_time": datetime.now().isoformat()
                }
                
                logger.info(f"Opened {direction} position for {asset} at {price}")
                
                return {
                    "status": "success",
                    "order_id": order_id,
                    "action": action,
                    "asset": asset,
                    "price": price,
                    "quantity": quantity,
                    "direction": direction
                }
                
            else:
                # Check if we have a position to close
                if asset not in self.open_positions:
                    logger.info(f"No position for {asset}, skipping close")
                    return {"status": "skipped", "reason": "no_position"}
                
                # Get position details
                position = self.open_positions[asset]
                
                # Close position
                close_price = signal.get("price_prediction", {}).get("1h", 0)
                if not close_price:
                    # Fallback to a dummy price for demonstration
                    close_price = 51000 if "BTC" in asset else 2100
                
                # Calculate profit/loss
                entry_price = position["entry_price"]
                profit_pct = (close_price - entry_price) / entry_price * 100 * position["direction"]
                
                # Remove position
                del self.open_positions[asset]
                
                logger.info(f"Closed position for {asset} at {close_price}, profit: {profit_pct:.2f}%")
                
                return {
                    "status": "success",
                    "action": action,
                    "asset": asset,
                    "entry_price": entry_price,
                    "close_price": close_price,
                    "profit_pct": float(profit_pct),
                    "direction": position["direction"]
                }
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_stop_loss_take_profit(self):
        """
        Check open positions for stop loss or take profit hits
        """
        try:
            positions_to_close = []
            
            for asset, position in self.open_positions.items():
                # In a real implementation, get current price from exchange
                # Here, we'll use a random price change for demonstration
                price_change_pct = np.random.normal(0, 1)  # Random change with 1% std dev
                current_price = position["entry_price"] * (1 + price_change_pct / 100)
                
                # Check stop loss
                if (position["direction"] > 0 and current_price <= position["stop_loss"]) or \
                   (position["direction"] < 0 and current_price >= position["stop_loss"]):
                    logger.info(f"Stop loss hit for {asset} at {current_price}")
                    positions_to_close.append({
                        "asset": asset,
                        "reason": "stop_loss",
                        "current_price": current_price
                    })
                
                # Check take profit
                elif (position["direction"] > 0 and current_price >= position["take_profit"]) or \
                     (position["direction"] < 0 and current_price <= position["take_profit"]):
                    logger.info(f"Take profit hit for {asset} at {current_price}")
                    positions_to_close.append({
                        "asset": asset,
                        "reason": "take_profit",
                        "current_price": current_price
                    })
            
            # Close positions
            for pos_info in positions_to_close:
                asset = pos_info["asset"]
                position = self.open_positions[asset]
                
                # Calculate profit/loss
                entry_price = position["entry_price"]
                current_price = pos_info["current_price"]
                profit_pct = (current_price - entry_price) / entry_price * 100 * position["direction"]
                
                # Remove position
                del self.open_positions[asset]
                
                logger.info(f"Closed position for {asset} at {current_price}, reason: {pos_info['reason']}, profit: {profit_pct:.2f}%")
                
                # Send notification
                self._send_notification(
                    f"Position closed: {asset}\n"
                    f"Reason: {pos_info['reason']}\n"
                    f"Profit: {profit_pct:.2f}%"
                )
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {e}")
    
    def _retrain_models(self):
        """
        Retrain all models
        """
        try:
            logger.info("Retraining models...")
            
            # For each asset
            for asset in self.assets:
                # Fetch historical data
                data_dict = self._fetch_latest_data(asset, lookback_days=90)
                
                if not data_dict:
                    logger.warning(f"No data available for {asset}, skipping retraining")
                    continue
                
                # 1. Retrain price prediction model
                price_model = self.price_models[asset]
                price_model.fit(data_dict)
                
                # 2. For ensemble model, we would need labeled data
                # This would require historical signals and their outcomes
                # For demonstration, we'll skip this step
                
                logger.info(f"Model retraining completed for {asset}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _send_notification(self, message):
        """
        Send notification via Telegram and/or WhatsApp
        
        Args:
            message (str): Message to send
        """
        try:
            # Telegram notification
            if TELEGRAM_CONFIG.get("token") and TELEGRAM_CONFIG.get("chat_id"):
                telegram_url = f"https://api.telegram.org/bot{TELEGRAM_CONFIG['token']}/sendMessage"
                telegram_data = {
                    "chat_id": TELEGRAM_CONFIG["chat_id"],
                    "text": message
                }
                
                requests.post(telegram_url, data=telegram_data)
                logger.info("Telegram notification sent")
            
            # WhatsApp notification
            if WHATSAPP_CONFIG.get("token") and WHATSAPP_CONFIG.get("to_phone"):
                # This is a placeholder - you would need to implement the WhatsApp API
                logger.info("WhatsApp notification would be sent here")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _generate_backtest_report(self):
        """
        Generate a backtest report
        
        Returns:
            dict: Backtest report
        """
        try:
            # In a real implementation, this would analyze historical signals and their outcomes
            # For demonstration, we'll return a placeholder report
            
            return {
                "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "initial_balance": 10000,
                "final_balance": 10500,
                "profit_pct": 5.0,
                "sharpe_ratio": 1.2,
                "max_drawdown": 3.5,
                "win_rate": 0.65,
                "trades": 20
            }
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return None
    
    def _save_signal(self, signal):
        """
        Save signal to database or file
        
        Args:
            signal (dict): Signal to save
        """
        try:
            # Create directory for signals if it doesn't exist
            signals_dir = DATA_DIR / "signals"
            signals_dir.mkdir(exist_ok=True)
            
            # Create file name based on date and asset
            date_str = datetime.now().strftime("%Y-%m-%d")
            asset_name = signal["asset"].replace("/", "_")
            file_path = signals_dir / f"{date_str}_{asset_name}.json"
            
            # Append to file
            with open(file_path, 'a') as f:
                f.write(json.dumps(signal) + "\n")
            
            logger.info(f"Signal saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def run_analysis_cycle(self):
        """
        Run a single analysis cycle for all assets
        """
        try:
            logger.info("Starting analysis cycle...")
            
            for asset in self.assets:
                # Analyze asset
                signal = self._analyze_asset(asset)
                
                if signal:
                    # Save signal
                    self._save_signal(signal)
                    
                    # Add to signal queue
                    self.signal_queue.put(signal)
                    
                    logger.info(f"Analysis completed for {asset}")
                else:
                    logger.warning(f"Analysis failed for {asset}")
            
            logger.info("Analysis cycle completed")
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
    
    def process_signals(self):
        """
        Process all signals in the queue
        """
        try:
            while not self.signal_queue.empty():
                signal = self.signal_queue.get()
                
                # Execute signal
                result = self._execute_signal(signal)
                
                # Log result
                if result["status"] == "success":
                    logger.info(f"Signal executed successfully: {result}")
                    
                    # Send notification
                    action = result.get("action", "unknown")
                    asset = result.get("asset", "unknown")
                    
                    notification = f"Signal executed: {action} {asset}"
                    if "profit_pct" in result:
                        notification += f"\nProfit: {result['profit_pct']:.2f}%"
                    
                    self._send_notification(notification)
                else:
                    logger.warning(f"Signal execution skipped or failed: {result}")
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
    
    def start(self):
        """
        Start the trading bot
        """
        if self.running:
            logger.warning("Trading bot is already running")
            return
        
        self.running = True
        logger.info("Starting trading bot...")
        
        # Start data collection
        if not self.backtest_mode:
            self.data_collector.start_data_collection()
        
        # Schedule tasks
        schedule.every(1).hours.do(self.run_analysis_cycle)
        schedule.every(4).hours.do(self._retrain_models)
        schedule.every(5).minutes.do(self._check_stop_loss_take_profit)
        schedule.every(10).minutes.do(self.process_signals)
        
        # Initial run
        self.run_analysis_cycle()
        self.process_signals()
        
        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping trading bot...")
            self.stop()
    
    def stop(self):
        """
        Stop the trading bot
        """
        if not self.running:
            logger.warning("Trading bot is not running")
            return
        
        self.running = False
        logger.info("Stopping trading bot...")
        
        # Stop data collection
        if not self.backtest_mode:
            self.data_collector.stop_data_collection()
        
        # Generate backtest report
        if self.backtest_mode:
            report = self._generate_backtest_report()
            if report:
                # Save report
                reports_dir = BACKTEST_DIR
                reports_dir.mkdir(exist_ok=True)
                
                report_path = reports_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Backtest report saved to {report_path}")
        
        logger.info("Trading bot stopped")
    
    def run_backtest(self, start_date, end_date):
        """
        Run backtest between specified dates
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Backtest report
        """
        try:
            logger.info(f"Running backtest from {start_date} to {end_date}...")
            
            # Set backtest mode
            self.backtest_mode = True
            
            # Here, you would implement a proper backtesting logic
            # For demonstration, we'll just return a placeholder report
            
            report = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_balance": 10000,
                "final_balance": 10800,
                "profit_pct": 8.0,
                "sharpe_ratio": 1.5,
                "max_drawdown": 4.0,
                "win_rate": 0.68,
                "trades": 25
            }
            
            # Save report
            reports_dir = BACKTEST_DIR
            reports_dir.mkdir(exist_ok=True)
            
            report_path = reports_dir / f"backtest_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Backtest report saved to {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--start', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for backtest (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Create trading bot
    bot = TradingBot(backtest_mode=args.backtest)
    
    if args.backtest and args.start and args.end:
        # Run backtest
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        
        bot.run_backtest(start_date, end_date)
    else:
        # Run trading bot
        try:
            bot.start()
        except KeyboardInterrupt:
            bot.stop() 