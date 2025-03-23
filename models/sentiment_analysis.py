import logging
import pandas as pd
import numpy as np
import os
import torch
import re
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import requests
import unicodedata

from config import MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SentimentAnalysis")

class DualSentimentAnalyzer:
    """
    Sentiment analyzer using both FinBERT and VADER
    """
    def __init__(self, asset):
        """
        Initialize the sentiment analyzer
        
        Args:
            asset (str): Asset symbol (e.g. "BTC/USD")
        """
        self.asset = asset
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.ensemble_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directory for this asset
        self.model_dir = MODELS_DIR / self.asset.replace("/", "_") / "sentiment"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for models
        self.ensemble_path = self.model_dir / "ensemble_model.pkl"
        self.config_path = self.model_dir / "config.json"
        
        # Assets for keywords and patterns
        self.asset_keywords = self._generate_asset_keywords()
        
        # Load models
        self._load_finbert()
        self._load_ensemble_if_exists()
    
    def _generate_asset_keywords(self):
        """
        Generate keywords related to the asset for filtering
        
        Returns:
            dict: Dictionary of keywords
        """
        asset_base = self.asset.split('/')[0]  # e.g., BTC from BTC/USD
        
        # Dictionary of asset names and keywords
        asset_dict = {
            "BTC": ["bitcoin", "btc", "satoshi", "crypto", "xbt", "#bitcoin", "#btc"],
            "ETH": ["ethereum", "eth", "vitalik", "buterin", "#ethereum", "#eth", "ether"],
            "BNB": ["binance", "bnb", "#bnbchain", "#binance", "#bnb"],
            "XRP": ["ripple", "xrp", "#xrp", "#ripple"],
            "SOL": ["solana", "sol", "#solana", "#sol"],
            "ADA": ["cardano", "ada", "#cardano", "#ada"],
            "DOT": ["polkadot", "dot", "#polkadot", "#dot"],
            "DOGE": ["dogecoin", "doge", "#dogecoin", "#doge"],
        }
        
        # Get keywords for the requested asset
        return asset_dict.get(asset_base, [asset_base.lower()])
    
    def _load_finbert(self):
        """Load the FinBERT model"""
        try:
            # Load FinBERT model and tokenizer
            logger.info("Loading FinBERT model...")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            
            # Move model to the appropriate device
            self.finbert_model.to(self.device)
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def _load_ensemble_if_exists(self):
        """Load the ensemble model if it exists"""
        if self.ensemble_path.exists():
            try:
                logger.info(f"Loading ensemble model for {self.asset}")
                self.ensemble_model = joblib.load(self.ensemble_path)
                logger.info("Ensemble model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading ensemble model: {e}")
                return False
        return False
    
    def _save_ensemble(self):
        """Save the ensemble model"""
        try:
            # Save the ensemble model
            joblib.dump(self.ensemble_model, self.ensemble_path)
            
            # Save configuration
            config = {
                'asset': self.asset,
                'last_updated': time.time()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Ensemble model saved to {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            return False
    
    def _preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove non-alphanumeric characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        return text
    
    def _is_relevant(self, text):
        """
        Check if text is relevant to the asset
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if relevant, False otherwise
        """
        # Check if any of the asset keywords are in the text
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.asset_keywords)
    
    def analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment scores
        """
        try:
            # Preprocess text
            clean_text = self._preprocess_text(text)
            
            if not clean_text:
                return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
            
            # Get sentiment scores
            sentiment = self.vader_analyzer.polarity_scores(clean_text)
            
            return sentiment
        
        except Exception as e:
            logger.error(f"Error analyzing text with VADER: {e}")
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
    
    def analyze_with_finbert(self, text):
        """
        Analyze sentiment using FinBERT
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment scores
        """
        try:
            # Preprocess text
            clean_text = self._preprocess_text(text)
            
            if not clean_text:
                return {'positive': 0, 'negative': 0, 'neutral': 1, 'sentiment': 'neutral', 'score': 0}
            
            # Tokenize text
            inputs = self.finbert_tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to the appropriate device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to numpy array
            scores = predictions.cpu().numpy()[0]
            
            # Get sentiment labels (positive, negative, neutral)
            sentiment_dict = {
                0: 'positive',
                1: 'negative',
                2: 'neutral'
            }
            
            # Get class with highest probability
            sentiment_class = scores.argmax()
            sentiment = sentiment_dict[sentiment_class]
            
            # Map to a score between -1 and 1
            score = 0
            if sentiment == 'positive':
                score = scores[0]
            elif sentiment == 'negative':
                score = -scores[1]
            
            return {
                'positive': float(scores[0]),
                'negative': float(scores[1]),
                'neutral': float(scores[2]),
                'sentiment': sentiment,
                'score': float(score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text with FinBERT: {e}")
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'sentiment': 'neutral', 'score': 0}
    
    def train_ensemble(self, texts, labels):
        """
        Train an ensemble model on labeled data
        
        Args:
            texts (list): List of texts
            labels (list): List of labels (1 for positive, 0 for neutral, -1 for negative)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Preprocess texts
            clean_texts = [self._preprocess_text(text) for text in texts]
            
            # Get features from VADER and FinBERT
            features = []
            
            for text in clean_texts:
                # VADER features
                vader_scores = self.analyze_with_vader(text)
                
                # FinBERT features
                finbert_scores = self.analyze_with_finbert(text)
                
                # Combine features
                feature_vector = [
                    vader_scores['compound'],
                    vader_scores['pos'],
                    vader_scores['neg'],
                    vader_scores['neu'],
                    finbert_scores['positive'],
                    finbert_scores['negative'],
                    finbert_scores['neutral'],
                    finbert_scores['score']
                ]
                
                features.append(feature_vector)
            
            # Convert to numpy array
            X = np.array(features)
            y = np.array(labels)
            
            # Train logistic regression models with different hyperparameters
            lr1 = LogisticRegression(C=0.1, max_iter=1000)
            lr2 = LogisticRegression(C=1.0, max_iter=1000)
            lr3 = LogisticRegression(C=10.0, max_iter=1000)
            
            # Create voting classifier
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('lr1', lr1),
                    ('lr2', lr2),
                    ('lr3', lr3)
                ],
                voting='soft'
            )
            
            # Train ensemble model
            self.ensemble_model.fit(X, y)
            
            # Save the model
            self._save_ensemble()
            
            logger.info(f"Ensemble model trained on {len(texts)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return False
    
    def predict_ensemble(self, text):
        """
        Predict sentiment using the ensemble model
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment prediction
        """
        try:
            if self.ensemble_model is None:
                logger.warning("Ensemble model not loaded, falling back to rule-based combination")
                return self.analyze_sentiment(text)
            
            # Preprocess text
            clean_text = self._preprocess_text(text)
            
            # Get features
            vader_scores = self.analyze_with_vader(clean_text)
            finbert_scores = self.analyze_with_finbert(clean_text)
            
            # Combine features
            feature_vector = np.array([
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neg'],
                vader_scores['neu'],
                finbert_scores['positive'],
                finbert_scores['negative'],
                finbert_scores['neutral'],
                finbert_scores['score']
            ]).reshape(1, -1)
            
            # Get prediction and probabilities
            pred = self.ensemble_model.predict(feature_vector)[0]
            probs = self.ensemble_model.predict_proba(feature_vector)[0]
            
            # Get sentiment score
            if pred == 1:  # Positive
                score = probs[2]  # Probability of positive class
            elif pred == -1:  # Negative
                score = -probs[0]  # Negative probability of negative class
            else:  # Neutral
                score = 0
            
            # Map prediction to sentiment label
            sentiment_map = {
                1: 'positive',
                0: 'neutral',
                -1: 'negative'
            }
            
            sentiment = sentiment_map[pred]
            
            return {
                'sentiment': sentiment,
                'score': float(score),
                'probabilities': {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2])
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting with ensemble model: {e}")
            return self.analyze_sentiment(text)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using both VADER and FinBERT
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Combined sentiment analysis
        """
        try:
            # Check if text is relevant to the asset
            if not self._is_relevant(text):
                return {
                    'sentiment': 'neutral',
                    'score': 0,
                    'relevant': False,
                    'vader': {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0},
                    'finbert': {'positive': 0, 'negative': 0, 'neutral': 1, 'sentiment': 'neutral', 'score': 0}
                }
            
            # Get VADER and FinBERT scores
            vader_scores = self.analyze_with_vader(text)
            finbert_scores = self.analyze_with_finbert(text)
            
            # Rule-based combination of scores
            # Convert VADER compound score (-1 to 1) and FinBERT score (-1 to 1)
            # to weighted average
            vader_weight = 0.4
            finbert_weight = 0.6
            
            combined_score = (vader_scores['compound'] * vader_weight) + (finbert_scores['score'] * finbert_weight)
            
            # Determine overall sentiment
            if combined_score >= 0.15:
                sentiment = 'positive'
            elif combined_score <= -0.15:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': float(combined_score),
                'relevant': True,
                'vader': vader_scores,
                'finbert': finbert_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0,
                'relevant': False,
                'error': str(e)
            }
    
    def analyze_news_batch(self, news_items):
        """
        Analyze sentiment for a batch of news items
        
        Args:
            news_items (list): List of dictionaries containing news items (must include 'title' and/or 'content' keys)
            
        Returns:
            list: List of sentiment analysis results for each news item
        """
        try:
            results = []
            
            for item in news_items:
                # Combine title and content if available
                text = ""
                if 'title' in item and item['title']:
                    text += item['title'] + " "
                
                if 'content' in item and item['content']:
                    text += item['content']
                
                # Skip if no text
                if not text:
                    continue
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(text)
                
                # Add metadata
                result = {
                    **sentiment,
                    'text': text[:200] + "..." if len(text) > 200 else text,
                    'source': item.get('source', 'unknown'),
                    'published_at': item.get('published_at', None)
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing news batch: {e}")
            return []
    
    def get_sentiment_impact_score(self, news_items):
        """
        Calculate overall sentiment impact score from news items
        
        Args:
            news_items (list): List of dictionaries containing news items
            
        Returns:
            float: Sentiment impact score between 0 and 1
        """
        try:
            # Analyze sentiment for all news items
            sentiment_results = self.analyze_news_batch(news_items)
            
            if not sentiment_results:
                return 0.5  # Neutral if no results
            
            # Filter for relevant results
            relevant_results = [r for r in sentiment_results if r.get('relevant', False)]
            
            if not relevant_results:
                return 0.5  # Neutral if no relevant results
            
            # Calculate weighted average of sentiment scores
            total_score = 0
            total_weight = 0
            
            for result in relevant_results:
                # Calculate weight based on source reliability and recency
                source_weight = 1.0
                recency_weight = 1.0
                
                # Adjust source weight based on source
                if 'source' in result:
                    if result['source'] in ['bloomberg', 'reuters', 'wsj', 'financial_times']:
                        source_weight = 1.5
                    elif result['source'] in ['twitter', 'reddit']:
                        source_weight = 0.7
                
                # Adjust recency weight if published_at is available
                if 'published_at' in result and result['published_at']:
                    try:
                        published_time = datetime.fromisoformat(result['published_at'].replace('Z', '+00:00'))
                        now = datetime.now().astimezone()
                        
                        # Calculate hours since publication
                        hours_ago = (now - published_time).total_seconds() / 3600
                        
                        # Exponential decay with half-life of 24 hours
                        recency_weight = 2 ** (-hours_ago / 24)
                    except Exception:
                        pass
                
                weight = source_weight * recency_weight
                total_score += result['score'] * weight
                total_weight += weight
            
            # Calculate weighted average
            average_score = total_score / total_weight if total_weight > 0 else 0
            
            # Convert to 0-1 scale
            impact_score = (average_score + 1) / 2
            
            return round(impact_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment impact score: {e}")
            return 0.5


# Example usage
if __name__ == "__main__":
    # Create sentiment analyzer
    analyzer = DualSentimentAnalyzer("BTC/USD")
    
    # Example news items
    news_items = [
        {
            'title': 'Bitcoin surges to new all-time high as institutional adoption grows',
            'content': 'Bitcoin reached a new all-time high today as more institutional investors are entering the crypto space.',
            'source': 'bloomberg',
            'published_at': '2023-03-01T12:00:00Z'
        },
        {
            'title': 'Cryptocurrency market faces regulatory challenges',
            'content': 'The SEC has announced new regulations that could impact cryptocurrency exchanges, causing uncertainty in the market.',
            'source': 'reuters',
            'published_at': '2023-03-02T10:30:00Z'
        }
    ]
    
    # Analyze sentiment
    sentiment_results = analyzer.analyze_news_batch(news_items)
    
    # Calculate overall impact
    impact_score = analyzer.get_sentiment_impact_score(news_items)
    
    print(f"Sentiment Impact Score: {impact_score}")
    print("Sentiment Results:")
    for result in sentiment_results:
        print(f"- {result['text']}: {result['sentiment']} ({result['score']})") 