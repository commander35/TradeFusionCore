import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pywt
import logging
import os
from pathlib import Path
import joblib
import time
import json

from config import MODELS_DIR, LSTM_CONFIG, CNN_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/price_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PricePrediction")

class LSTMCNNHybridModel:
    """
    Hybrid LSTM+CNN model for price prediction
    """
    def __init__(self, asset, timeframes=None):
        """
        Initialize the model
        
        Args:
            asset (str): Asset symbol
            timeframes (list, optional): List of timeframes to include in the model
        """
        self.asset = asset
        self.timeframes = timeframes or ["1h", "4h", "1d"]
        self.model = None
        self.scalers = {}
        self.wavelet_transforms = {}
        self.feature_columns = {}
        self.sequence_length = 20  # Number of time steps to look back
        
        # Create directory for this asset
        self.model_dir = MODELS_DIR / self.asset.replace("/", "_") / "price_prediction"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / "hybrid_model.h5"
        self.scaler_path = self.model_dir / "scalers.pkl"
        self.config_path = self.model_dir / "model_config.json"
        
        # Load model if it exists
        self._load_model_if_exists()
    
    def _load_model_if_exists(self):
        """Load the model if it exists"""
        if self.model_path.exists() and self.scaler_path.exists() and self.config_path.exists():
            try:
                logger.info(f"Loading existing model for {self.asset}")
                self.model = load_model(self.model_path)
                self.scalers = joblib.load(self.scaler_path)
                
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.feature_columns = config.get('feature_columns', {})
                    self.sequence_length = config.get('sequence_length', 20)
                
                logger.info(f"Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
    
    def _save_model(self):
        """Save the model and related components"""
        try:
            # Save the keras model
            self.model.save(self.model_path)
            
            # Save the scalers
            joblib.dump(self.scalers, self.scaler_path)
            
            # Save the configuration
            config = {
                'asset': self.asset,
                'timeframes': self.timeframes,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'last_updated': time.time()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Model saved to {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _apply_wavelet_transform(self, data, timeframe):
        """
        Apply wavelet transform to decompose the time series
        
        Args:
            data (pd.DataFrame): Dataframe with time series data
            timeframe (str): Timeframe of the data
            
        Returns:
            np.ndarray: Wavelet transformed features
        """
        try:
            # Check if we have the price data
            if 'close' not in data.columns:
                logger.error(f"No close price data found for {timeframe}")
                return np.array([])
            
            # Get the close price
            close_prices = data['close'].values
            
            # Apply wavelet transform (level 3 decomposition using 'db4' wavelet)
            coeffs = pywt.wavedec(close_prices, 'db4', level=3)
            
            # Store the wavelet transform for this timeframe
            self.wavelet_transforms[timeframe] = coeffs
            
            # Reconstruct approximation and details
            reconstructed = []
            for i, coeff in enumerate(coeffs):
                # Create a list where all elements are zero
                coeff_list = [np.zeros_like(c) for c in coeffs]
                # Replace one element with the actual coefficient
                coeff_list[i] = coeff
                # Reconstruct the signal
                reconstructed.append(pywt.waverec(coeff_list, 'db4')[:len(close_prices)])
            
            # Stack the reconstructed signals
            transformed = np.column_stack(reconstructed)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error applying wavelet transform: {e}")
            return np.array([])
    
    def _prepare_data(self, data_dict, target_timeframe="1h", horizon=24):
        """
        Prepare data for training or prediction
        
        Args:
            data_dict (dict): Dictionary with timeframes as keys and dataframes as values
            target_timeframe (str): Target timeframe for prediction
            horizon (int): Prediction horizon in hours
            
        Returns:
            tuple: (X, y) for training, or X for prediction
        """
        try:
            # Initialize features and scalers for each timeframe
            X_timeframes = {}
            
            for timeframe, data in data_dict.items():
                # Get relevant feature columns for this timeframe
                if timeframe in self.feature_columns:
                    feature_cols = self.feature_columns[timeframe]
                else:
                    # Default feature selection
                    base_features = ['open', 'high', 'low', 'close', 'volume']
                    indicator_features = [col for col in data.columns if any(ind in col for ind in ['rsi', 'macd', 'bollinger'])]
                    pivot_features = [col for col in data.columns if 'pivot' in col or col.startswith(('r', 's')) and '_' in col]
                    
                    feature_cols = base_features + indicator_features + pivot_features
                    self.feature_columns[timeframe] = feature_cols
                
                # Create a copy of the data with selected features
                df = data[feature_cols].copy()
                
                # Apply wavelet transform and add as features
                wavelet_features = self._apply_wavelet_transform(data, timeframe)
                
                if wavelet_features.size > 0:
                    # Add wavelet features to dataframe
                    for i in range(wavelet_features.shape[1]):
                        df[f'wavelet_{i}'] = wavelet_features[:, i]
                    
                    # Add wavelet feature columns
                    wavelet_cols = [f'wavelet_{i}' for i in range(wavelet_features.shape[1])]
                    self.feature_columns[timeframe].extend(wavelet_cols)
                
                # Scale the features
                if timeframe not in self.scalers:
                    self.scalers[timeframe] = MinMaxScaler()
                    scaled_features = self.scalers[timeframe].fit_transform(df)
                else:
                    scaled_features = self.scalers[timeframe].transform(df)
                
                # Create sequences
                sequences = []
                for i in range(len(scaled_features) - self.sequence_length):
                    sequences.append(scaled_features[i:i+self.sequence_length])
                
                if sequences:
                    X_timeframes[timeframe] = np.array(sequences)
                else:
                    logger.warning(f"No sequences created for {timeframe}")
                    X_timeframes[timeframe] = np.array([])
            
            # Ensure we have the target timeframe data
            if target_timeframe not in X_timeframes or X_timeframes[target_timeframe].size == 0:
                logger.error(f"No data available for target timeframe {target_timeframe}")
                return None, None
            
            # Get the corresponding target data
            if target_timeframe in data_dict:
                target_data = data_dict[target_timeframe]
                
                # Create target variable - future price movement
                # For horizon-step-ahead prediction
                target_close = target_data['close'].values
                target_shifts = []
                
                # Create multiple time horizons (e.g., 1h, 4h, 24h ahead for 1h timeframe)
                hours_per_period = {"1h": 1, "4h": 4, "1d": 24}
                periods = horizon // hours_per_period.get(target_timeframe, 1)
                
                for i in range(1, periods + 1):
                    if i <= len(target_close) - self.sequence_length:
                        shift = np.roll(target_close, -i)[self.sequence_length:]
                        target_shifts.append(shift)
                
                if target_shifts:
                    y_multi_horizon = np.column_stack(target_shifts)
                    
                    # Scale the targets
                    if 'target' not in self.scalers:
                        self.scalers['target'] = MinMaxScaler()
                        y_scaled = self.scalers['target'].fit_transform(y_multi_horizon.reshape(-1, 1)).reshape(y_multi_horizon.shape)
                    else:
                        y_scaled = self.scalers['target'].transform(y_multi_horizon.reshape(-1, 1)).reshape(y_multi_horizon.shape)
                    
                    # Remove the last 'periods' rows where we don't have complete targets
                    for timeframe in X_timeframes:
                        if X_timeframes[timeframe].size > 0:
                            X_timeframes[timeframe] = X_timeframes[timeframe][:-periods]
                    
                    y = y_scaled
                else:
                    logger.warning(f"No target shifts created for {target_timeframe}")
                    y = None
            else:
                logger.warning(f"Target timeframe {target_timeframe} not in data_dict")
                y = None
            
            return X_timeframes, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None
    
    def _build_model(self, input_shapes, output_dim):
        """
        Build the hybrid LSTM+CNN model
        
        Args:
            input_shapes (dict): Dictionary with timeframes as keys and input shapes as values
            output_dim (int): Output dimension (number of prediction horizons)
            
        Returns:
            tensorflow.keras.models.Model: The built model
        """
        try:
            # Create input and submodels for each timeframe
            inputs = {}
            timeframe_outputs = {}
            
            for timeframe, shape in input_shapes.items():
                # Create input layer
                inputs[timeframe] = Input(shape=shape[1:], name=f"input_{timeframe}")
                
                # CNN branch
                cnn = Conv1D(filters=CNN_CONFIG["filters"][0], 
                             kernel_size=CNN_CONFIG["kernel_sizes"][0], 
                             activation='relu', 
                             name=f"conv1_{timeframe}")(inputs[timeframe])
                cnn = MaxPooling1D(pool_size=CNN_CONFIG["pool_sizes"][0], name=f"pool1_{timeframe}")(cnn)
                
                cnn = Conv1D(filters=CNN_CONFIG["filters"][1], 
                             kernel_size=CNN_CONFIG["kernel_sizes"][1], 
                             activation='relu', 
                             name=f"conv2_{timeframe}")(cnn)
                cnn = MaxPooling1D(pool_size=CNN_CONFIG["pool_sizes"][1], name=f"pool2_{timeframe}")(cnn)
                
                cnn = Flatten(name=f"flatten_{timeframe}")(cnn)
                
                # LSTM branch
                lstm = LSTM(units=LSTM_CONFIG["units"], 
                           return_sequences=True, 
                           dropout=LSTM_CONFIG["dropout"],
                           recurrent_dropout=LSTM_CONFIG["recurrent_dropout"],
                           name=f"lstm1_{timeframe}")(inputs[timeframe])
                
                lstm = LSTM(units=LSTM_CONFIG["units"] // 2, 
                           return_sequences=False,
                           dropout=LSTM_CONFIG["dropout"],
                           recurrent_dropout=LSTM_CONFIG["recurrent_dropout"],
                           name=f"lstm2_{timeframe}")(lstm)
                
                # Combine CNN and LSTM branches
                combined = Concatenate(name=f"concat_{timeframe}")([cnn, lstm])
                
                # Dense layers
                dense = Dense(64, activation='relu', name=f"dense1_{timeframe}")(combined)
                dense = Dropout(CNN_CONFIG["dropout"], name=f"dropout1_{timeframe}")(dense)
                
                timeframe_outputs[timeframe] = dense
            
            # Combine outputs from all timeframes
            if len(timeframe_outputs) > 1:
                merged = Concatenate(name="merge_timeframes")(list(timeframe_outputs.values()))
            else:
                merged = list(timeframe_outputs.values())[0]
            
            # Final dense layers
            x = Dense(128, activation='relu', name="dense2")(merged)
            x = Dropout(CNN_CONFIG["dropout"], name="dropout2")(x)
            x = Dense(64, activation='relu', name="dense3")(x)
            x = Dropout(CNN_CONFIG["dropout"], name="dropout3")(x)
            
            # Output layer - multi-horizon prediction
            outputs = Dense(output_dim, name="output")(x)
            
            # Create the model
            model = Model(inputs=list(inputs.values()), outputs=outputs)
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=LSTM_CONFIG["learning_rate"]),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"Model built successfully with inputs: {input_shapes}")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return None
    
    def fit(self, data_dict, target_timeframe="1h", horizon=24, epochs=None, batch_size=None):
        """
        Train the model
        
        Args:
            data_dict (dict): Dictionary with timeframes as keys and dataframes as values
            target_timeframe (str): Target timeframe for prediction
            horizon (int): Prediction horizon in hours
            epochs (int, optional): Number of epochs
            batch_size (int, optional): Batch size
            
        Returns:
            dict: Training history
        """
        try:
            # Prepare data
            X_timeframes, y = self._prepare_data(data_dict, target_timeframe, horizon)
            
            if X_timeframes is None or y is None:
                logger.error("Failed to prepare data for training")
                return None
            
            # Check if we have data for all timeframes
            for timeframe in self.timeframes:
                if timeframe not in X_timeframes or X_timeframes[timeframe].size == 0:
                    logger.warning(f"No data for timeframe {timeframe}, skipping training")
                    return None
            
            # Get input shapes
            input_shapes = {tf: X.shape for tf, X in X_timeframes.items()}
            
            # Build model if it doesn't exist
            if self.model is None:
                self.model = self._build_model(input_shapes, y.shape[1])
            
            if self.model is None:
                logger.error("Failed to build model")
                return None
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=LSTM_CONFIG["patience"],
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=self.model_dir / "best_model.h5",
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train the model
            history = self.model.fit(
                list(X_timeframes.values()),
                y,
                epochs=epochs or LSTM_CONFIG["epochs"],
                batch_size=batch_size or LSTM_CONFIG["batch_size"],
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save the model
            self._save_model()
            
            logger.info(f"Model trained successfully")
            return history.history
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def predict(self, data_dict, target_timeframe="1h", horizon=24):
        """
        Make price predictions
        
        Args:
            data_dict (dict): Dictionary with timeframes as keys and dataframes as values
            target_timeframe (str): Target timeframe for prediction
            horizon (int): Prediction horizon in hours
            
        Returns:
            dict: Predictions for different time horizons
        """
        try:
            # Check if model exists
            if self.model is None:
                logger.error("No model available for prediction")
                return None
            
            # Prepare data
            X_timeframes, _ = self._prepare_data(data_dict, target_timeframe, horizon)
            
            if X_timeframes is None:
                logger.error("Failed to prepare data for prediction")
                return None
            
            # Check if we have data for all timeframes
            for timeframe in self.timeframes:
                if timeframe not in X_timeframes or X_timeframes[timeframe].size == 0:
                    logger.warning(f"No data for timeframe {timeframe}, skipping prediction")
                    return None
            
            # Make prediction
            y_pred = self.model.predict(list(X_timeframes.values()))
            
            # Inverse transform the predictions
            if 'target' in self.scalers:
                y_pred_inv = self.scalers['target'].inverse_transform(y_pred)
            else:
                logger.warning("Target scaler not found, returning raw predictions")
                y_pred_inv = y_pred
            
            # Convert to dictionary with time horizons
            hours_per_period = {"1h": 1, "4h": 4, "1d": 24}
            period_length = hours_per_period.get(target_timeframe, 1)
            
            predictions = {}
            for i in range(y_pred_inv.shape[1]):
                time_horizon = f"{(i+1) * period_length}h"
                predictions[time_horizon] = float(y_pred_inv[-1, i])
            
            logger.info(f"Predictions generated: {predictions}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from data_collector import TradingViewDataCollector
    from datetime import datetime, timedelta
    
    # Create data collector
    collector = TradingViewDataCollector(assets=["BTC/USD"], timeframes=["1h", "4h", "1d"])
    
    # Get data for last 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    data_dict = {}
    for timeframe in collector.timeframes:
        data = collector.get_data_with_pivots("BTC/USD", timeframe, start_date, end_date)
        if not data.empty:
            data_dict[timeframe] = data
    
    # Create and train model
    model = LSTMCNNHybridModel("BTC/USD", timeframes=["1h", "4h", "1d"])
    
    # Train the model
    history = model.fit(data_dict, target_timeframe="1h", horizon=24)
    
    # Make predictions
    predictions = model.predict(data_dict, target_timeframe="1h", horizon=24)
    
    print("Predictions:", predictions) 