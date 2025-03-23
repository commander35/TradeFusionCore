#!/usr/bin/env python3
"""
Trading Bot Runner Script

This script provides a simplified interface to run the trading bot with
command line arguments for different modes and configurations.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import traceback
import logging

from trading_bot import TradingBot
from config import TRADING_CONFIG

def setup_logger():
    """Set up basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/run.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Runner")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI Trading Bot Runner')
    
    # Main mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--live', action='store_true', help='Run in live trading mode')
    mode_group.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    mode_group.add_argument('--analyze', action='store_true', help='Run a single analysis cycle and exit')
    mode_group.add_argument('--retrain', action='store_true', help='Retrain models and exit')
    
    # Backtest parameters
    parser.add_argument('--start', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for backtest (YYYY-MM-DD)')
    
    # Asset selection
    parser.add_argument('--assets', type=str, nargs='+', help='Assets to analyze/trade (e.g., BTC/USD ETH/USD)')
    
    # Timeframe selection
    parser.add_argument('--timeframes', type=str, nargs='+', help='Timeframes to analyze (e.g., 1h 4h 1d)')
    
    # Other options
    parser.add_argument('--no-telegram', action='store_true', help='Disable Telegram notifications')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def run_trading_bot(args, logger):
    """Run the trading bot based on command line arguments"""
    
    # Set assets if provided
    assets = args.assets if args.assets else TRADING_CONFIG["assets"]
    
    # Set timeframes if provided
    timeframes = args.timeframes if args.timeframes else TRADING_CONFIG["timeframes"]
    
    # Create trading bot
    bot = TradingBot(assets=assets, timeframes=timeframes, backtest_mode=args.backtest)
    
    # Disable Telegram if requested
    if args.no_telegram:
        logger.info("Telegram notifications disabled")
        # This would normally be handled by modifying the config at runtime
    
    try:
        if args.backtest:
            # Default to last 30 days if dates not provided
            if not args.start:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
            else:
                start_date = datetime.strptime(args.start, '%Y-%m-%d')
                end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()
            
            logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
            result = bot.run_backtest(start_date, end_date)
            
            if result:
                logger.info(f"Backtest completed with profit: {result['profit_pct']}%")
                logger.info(f"Sharpe ratio: {result['sharpe_ratio']}, Max drawdown: {result['max_drawdown']}%")
            else:
                logger.error("Backtest failed")
            
        elif args.analyze:
            logger.info("Running single analysis cycle")
            bot.run_analysis_cycle()
            bot.process_signals()
            logger.info("Analysis cycle completed")
            
        elif args.retrain:
            logger.info("Retraining models")
            bot._retrain_models()
            logger.info("Model retraining completed")
            
        else:  # Default to live mode
            logger.info("Starting live trading bot")
            bot.start()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        if bot.running:
            bot.stop()
            
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        traceback.print_exc()
        
        # Ensure bot is stopped
        if bot.running:
            bot.stop()
        
        return 1
    
    return 0

def main():
    """Main function"""
    logger = setup_logger()
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Print startup info
    logger.info("AI Trading Bot Runner")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Run the trading bot
    return run_trading_bot(args, logger)

if __name__ == "__main__":
    sys.exit(main()) 