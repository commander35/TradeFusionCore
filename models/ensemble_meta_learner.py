import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import time
import shap
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import MODELS_DIR, XGBOOST_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ensemble_meta_learner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnsembleMetaLearner")

class EnsembleMetaLearner:
    """
    Ensemble Meta-Learner using XGBoost to combine outputs from other models
    """
    def __init__(self, asset):
        """
        Initialize the meta-learner
        
        Args:
            asset (str): Asset symbol (e.g. "BTC/USD")
        """
        self.asset = asset
        self.model = None
        self.feature_names = None
        self.shap_values = None
        
        # Create directory for this asset
        self.model_dir = MODELS_DIR / self.asset.replace("/", "_") / "ensemble"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for models and configuration
        self.model_path = self.model_dir / "xgboost_model.json"
        self.config_path = self.model_dir / "model_config.json"
        self.shap_path = self.model_dir / "shap_values.pkl"
        
        # Load model if it exists
        self._load_model_if_exists()
    
    def _load_model_if_exists(self):
        """Load the model if it exists"""
        if self.model_path.exists() and self.config_path.exists():
            try:
                logger.info(f"Loading existing ensemble model for {self.asset}")
                
                # Load XGBoost model
                self.model = xgb.Booster()
                self.model.load_model(str(self.model_path))
                
                # Load configuration
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.feature_names = config.get('feature_names', [])
                
                # Load SHAP values if available
                if self.shap_path.exists():
                    self.shap_values = joblib.load(self.shap_path)
                
                logger.info(f"Model loaded successfully with {len(self.feature_names)} features")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
    
    def _save_model(self):
        """Save the model and related components"""
        try:
            # Save the XGBoost model
            self.model.save_model(str(self.model_path))
            
            # Save the configuration
            config = {
                'asset': self.asset,
                'feature_names': self.feature_names,
                'last_updated': time.time()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save SHAP values if available
            if self.shap_values is not None:
                joblib.dump(self.shap_values, self.shap_path)
                
            logger.info(f"Model saved to {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def train(self, features_df, target, test_size=0.2):
        """
        Train the ensemble model
        
        Args:
            features_df (pd.DataFrame): DataFrame with features from all models
            target (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Performance metrics
        """
        try:
            logger.info(f"Training ensemble model for {self.asset} with {features_df.shape[1]} features")
            
            # Store feature names
            self.feature_names = features_df.columns.tolist()
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, target, test_size=test_size, random_state=42
            )
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)
            
            # Set XGBoost parameters
            params = {
                'objective': XGBOOST_CONFIG["objective"],
                'max_depth': XGBOOST_CONFIG["max_depth"],
                'eta': XGBOOST_CONFIG["learning_rate"],
                'subsample': XGBOOST_CONFIG["subsample"],
                'colsample_bytree': XGBOOST_CONFIG["colsample_bytree"],
                'eval_metric': XGBOOST_CONFIG["eval_metric"]
            }
            
            # Train model
            watchlist = [(dtrain, 'train'), (dtest, 'eval')]
            self.model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=XGBOOST_CONFIG["n_estimators"],
                evals=watchlist,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Calculate SHAP values for feature importance
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_test)
            
            # Save model and SHAP values
            self._save_model()
            
            # Evaluate model
            y_pred = self.model.predict(dtest)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred_binary)),
                'precision': float(precision_score(y_test, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_binary, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred_binary, zero_division=0))
            }
            
            logger.info(f"Model trained successfully. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return None
    
    def predict(self, features):
        """
        Make predictions using the ensemble model
        
        Args:
            features (pd.DataFrame or dict): Features from all models
            
        Returns:
            float: Prediction score (0-1)
        """
        try:
            if self.model is None:
                logger.error("No model available for prediction")
                return 0.5  # Return neutral score
            
            # Convert dict to DataFrame if necessary
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            # Ensure all feature names are present
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                
                # Add missing features with default values
                for feature in missing_features:
                    features[feature] = 0
            
            # Reorder columns to match model's expectations
            features = features[self.feature_names]
            
            # Create DMatrix for prediction
            dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
            
            # Make prediction
            prediction = self.model.predict(dmatrix)
            
            logger.info(f"Prediction made: {prediction[0]}")
            return float(prediction[0])
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5  # Return neutral score
    
    def get_feature_importance(self, plot=False):
        """
        Get feature importance based on SHAP values
        
        Args:
            plot (bool): Whether to plot the feature importance
            
        Returns:
            pd.DataFrame: Feature importance
        """
        try:
            if self.model is None or self.shap_values is None:
                logger.error("No model or SHAP values available")
                return None
            
            # Calculate feature importance from SHAP values
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.shap_values).mean(axis=0)
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            if plot:
                plt.figure(figsize=(10, 6))
                plt.barh(feature_importance['feature'], feature_importance['importance'])
                plt.xlabel('Mean |SHAP value|')
                plt.title('Feature Importance')
                plt.tight_layout()
                
                # Save plot
                plot_path = self.model_dir / "feature_importance.png"
                plt.savefig(plot_path)
                plt.close()
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
    
    def get_signal_components(self, features):
        """
        Get the components of a signal based on SHAP values
        
        Args:
            features (pd.DataFrame or dict): Features used for prediction
            
        Returns:
            dict: Signal components with their contributions
        """
        try:
            if self.model is None or self.shap_values is None:
                logger.error("No model or SHAP values available")
                return {}
            
            # Convert dict to DataFrame if necessary
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            # Ensure all feature names are present
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                
                # Add missing features with default values
                for feature in missing_features:
                    features[feature] = 0
            
            # Reorder columns to match model's expectations
            features = features[self.feature_names]
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values for this prediction
            shap_values = explainer.shap_values(features)
            
            # Get component contributions
            components = {}
            for i, feature in enumerate(self.feature_names):
                components[feature] = float(shap_values[0, i])
            
            # Sort by absolute contribution
            sorted_components = {k: v for k, v in sorted(
                components.items(), 
                key=lambda item: abs(item[1]), 
                reverse=True
            )}
            
            # Return top N components
            top_components = dict(list(sorted_components.items())[:10])
            
            return top_components
            
        except Exception as e:
            logger.error(f"Error getting signal components: {e}")
            return {}
    
    def monte_carlo_risk_simulation(self, features, n_simulations=1000):
        """
        Run Monte Carlo simulation to calculate risk score
        
        Args:
            features (pd.DataFrame or dict): Features from all models
            n_simulations (int): Number of simulations to run
            
        Returns:
            dict: Risk simulation results
        """
        try:
            if self.model is None:
                logger.error("No model available for risk simulation")
                return None
            
            # Convert dict to DataFrame if necessary
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            # Ensure all feature names are present and reorder
            for feature in self.feature_names:
                if feature not in features.columns:
                    features[feature] = 0
            
            features = features[self.feature_names]
            
            # Base prediction
            base_prediction = self.predict(features)
            
            # Run simulations by adding random noise to features
            simulated_predictions = []
            
            for _ in range(n_simulations):
                # Add random noise to features (5-15% variance)
                noise_factors = np.random.normal(1, 0.1, features.shape[1])
                noisy_features = features.copy()
                
                # Apply noise to each feature
                for i, feature in enumerate(self.feature_names):
                    noisy_features[feature] = features[feature] * noise_factors[i]
                
                # Get prediction with noisy features
                pred = self.predict(noisy_features)
                simulated_predictions.append(pred)
            
            # Calculate statistics
            predictions = np.array(simulated_predictions)
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            
            # Calculate Value at Risk (VaR) at different confidence levels
            var_95 = np.percentile(predictions, 5)
            var_99 = np.percentile(predictions, 1)
            
            # Calculate confidence interval
            ci_lower = np.percentile(predictions, 2.5)
            ci_upper = np.percentile(predictions, 97.5)
            
            # Calculate risk score (0-1, higher means more risk)
            # Based on coefficient of variation
            risk_score = min(1.0, (std_prediction / (mean_prediction + 1e-10)) * 2)
            
            results = {
                'base_prediction': float(base_prediction),
                'mean_prediction': float(mean_prediction),
                'std_prediction': float(std_prediction),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'confidence_interval': [float(ci_lower), float(ci_upper)],
                'risk_score': float(risk_score)
            }
            
            logger.info(f"Monte Carlo simulation completed with risk score: {risk_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}")
            return None
    
    def black_litterman_allocation(self, prediction_score, risk_score, market_view=None):
        """
        Calculate suggested position allocation using Black-Litterman model
        
        Args:
            prediction_score (float): Prediction score (0-1)
            risk_score (float): Risk score (0-1)
            market_view (dict, optional): Additional market views
            
        Returns:
            dict: Suggested position allocation
        """
        try:
            # Convert prediction score from 0-1 to -1 to 1 scale
            market_return = (prediction_score - 0.5) * 2
            
            # Default maximum position size
            max_position = 0.5  # 50% of available capital
            
            # Adjust max position based on risk score
            # Higher risk -> smaller position
            risk_adjusted_max = max_position * (1 - risk_score * 0.8)
            
            # Calculate confidence level from risk score
            # Higher risk -> lower confidence
            confidence = 1 - risk_score
            
            # Calculate position size based on expected return and confidence
            # Kelly criterion inspired allocation
            position_size = market_return * confidence * risk_adjusted_max
            
            # Ensure position is between -max_position and max_position
            position_size = max(min(position_size, risk_adjusted_max), -risk_adjusted_max)
            
            # Adjust position size based on market views if available
            if market_view:
                sentiment_impact = market_view.get('sentiment_impact', 0.5)
                market_depth = market_view.get('market_depth', 0.5)
                
                # Convert sentiment from 0-1 to -0.2 to 0.2 scale
                sentiment_adj = (sentiment_impact - 0.5) * 0.4
                
                # Convert market depth from 0-1 to 0 to 0.2 scale
                # High market depth (liquidity) allows larger positions
                depth_adj = (market_depth - 0.5) * 0.4
                
                # Apply adjustments
                position_size += sentiment_adj
                position_size *= (1 + depth_adj)
            
            # Calculate absolute position size as percentage
            abs_position = abs(position_size) * 100
            
            # Determine action based on position size
            if position_size > 0.1:
                action = "LONG"
            elif position_size < -0.1:
                action = "SHORT"
            else:
                action = "NEUTRAL"
            
            # Add risk qualifier
            if abs_position < 1.0:
                action_with_risk = f"{action}_MINIMAL"
            elif risk_score > 0.7:
                action_with_risk = f"{action}_HIGH_RISK"
            elif risk_score > 0.4:
                action_with_risk = f"{action}_MODERATE_RISK"
            else:
                action_with_risk = f"{action}_LIMITED_RISK"
            
            return {
                'recommended_action': action_with_risk,
                'position_size': f"{abs_position:.1f}%",
                'position_direction': 1 if position_size > 0 else (-1 if position_size < 0 else 0),
                'confidence': float(confidence),
                'risk_score': float(risk_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating allocation: {e}")
            return {
                'recommended_action': "NEUTRAL",
                'position_size': "0.0%",
                'position_direction': 0,
                'confidence': 0.5,
                'risk_score': 0.5
            }


# Example usage
if __name__ == "__main__":
    # Create sample features
    features = {
        'price_prediction_1h': 0.65,
        'price_prediction_4h': 0.58,
        'price_prediction_24h': 0.62,
        'rsi_1h': 58,
        'rsi_4h': 62,
        'rsi_1d': 57,
        'macd_1h': 0.5,
        'macd_4h': 0.8,
        'macd_1d': 0.3,
        'bollinger_squeeze_1h': 0,
        'bollinger_squeeze_4h': 1,
        'bollinger_squeeze_1d': 0,
        'pivot_breakout_1h': 0.3,
        'pivot_breakout_4h': 0.7,
        'pivot_breakout_1d': 0.2,
        'sentiment_impact': 0.65,
        'market_depth': 0.75
    }
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Create sample target (1 for bullish signal, 0 for bearish)
    target = pd.Series([1])
    
    # Create ensemble
    ensemble = EnsembleMetaLearner("BTC/USD")
    
    # Train model
    metrics = ensemble.train(features_df, target)
    print("Training metrics:", metrics)
    
    # Make prediction
    prediction = ensemble.predict(features)
    print(f"Prediction: {prediction}")
    
    # Get feature importance
    importance = ensemble.get_feature_importance(plot=True)
    print("Feature importance:", importance)
    
    # Run Monte Carlo simulation
    risk_sim = ensemble.monte_carlo_risk_simulation(features)
    print("Risk simulation:", risk_sim)
    
    # Calculate allocation
    allocation = ensemble.black_litterman_allocation(
        prediction, 
        risk_sim['risk_score'],
        {'sentiment_impact': features['sentiment_impact'], 'market_depth': features['market_depth']}
    )
    print("Allocation:", allocation) 