"""
Prediction Service
==================

Generates real-time predictions using trained ML models.
Called every minute after new data is collected.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import (
    SessionLocal, TradingProfile, ProfileModel, ProfilePrediction,
    MarketData, ModelStatus
)

logger = logging.getLogger(__name__)

# Model storage paths
MODELS_DIR = Path(__file__).parent.parent / "models" / "trained"
SCALERS_DIR = Path(__file__).parent.parent / "models" / "scalers"


class PredictionService:
    """Service for generating predictions from trained models"""

    def __init__(self):
        self.models_cache = {}  # Cache loaded models {model_id: (model, scaler)}
        self.indicators_cache = {}  # Cache calculated indicators {symbol: indicators_dict}

    async def generate_predictions_for_profile(
        self,
        profile: TradingProfile,
        db: Session
    ) -> Dict:
        """
        Generate predictions for a profile using all deployed models

        Args:
            profile: TradingProfile instance
            db: Database session

        Returns:
            dict with prediction results
        """
        try:
            symbol = profile.symbol
            logger.info(f"Generating predictions for {symbol}")

            # Get all deployed models for this profile
            deployed_models = db.query(ProfileModel).filter(
                ProfileModel.profile_id == profile.id,
                ProfileModel.is_deployed == True,
                ProfileModel.status == ModelStatus.DEPLOYED
            ).all()

            if not deployed_models:
                logger.debug(f"{symbol}: No deployed models found")
                return {
                    'success': False,
                    'reason': 'no_deployed_models',
                    'symbol': symbol
                }

            # Fetch latest market data (need enough for indicator calculation)
            latest_data = self._fetch_latest_market_data(db, symbol, lookback=200)

            if latest_data is None or len(latest_data) < 50:
                logger.warning(f"{symbol}: Insufficient market data ({len(latest_data) if latest_data is not None else 0} rows)")
                return {
                    'success': False,
                    'reason': 'insufficient_data',
                    'symbol': symbol
                }

            # Calculate all technical indicators
            indicators = self._calculate_indicators(latest_data)

            if indicators is None:
                logger.error(f"{symbol}: Failed to calculate indicators")
                return {
                    'success': False,
                    'reason': 'indicator_calculation_failed',
                    'symbol': symbol
                }

            # Generate predictions from each model
            predictions_generated = 0
            for model in deployed_models:
                try:
                    prediction_result = self._generate_single_prediction(
                        model=model,
                        indicators=indicators,
                        latest_price=float(latest_data.iloc[-1]['close_price']),
                        db=db,
                        profile=profile
                    )

                    if prediction_result['success']:
                        predictions_generated += 1
                        logger.info(f"✅ {symbol}/{model.model_name}: Prediction saved - "
                                  f"Signal={prediction_result['signal']}, "
                                  f"Confidence={prediction_result['confidence']:.2f}")

                except Exception as e:
                    logger.error(f"Error generating prediction for {model.model_name}: {e}")
                    continue

            return {
                'success': True,
                'symbol': symbol,
                'predictions_generated': predictions_generated,
                'total_models': len(deployed_models)
            }

        except Exception as e:
            logger.error(f"Error in generate_predictions_for_profile: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'symbol': profile.symbol
            }

    def _fetch_latest_market_data(
        self,
        db: Session,
        symbol: str,
        lookback: int = 200
    ) -> Optional[pd.DataFrame]:
        """Fetch latest market data for indicator calculation"""
        try:
            # Get latest N records
            records = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(
                desc(MarketData.timestamp)
            ).limit(lookback).all()

            if not records:
                return None

            # Convert to DataFrame (reverse to chronological order)
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'open_price': r.open_price,
                'high_price': r.high_price,
                'low_price': r.low_price,
                'close_price': r.close_price,
                'volume': r.volume,
                'number_of_trades': r.number_of_trades or 0,
                'quote_asset_volume': r.quote_asset_volume or 0
            } for r in reversed(records)])

            return df

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate all technical indicators for the latest data point

        Returns:
            dict with indicator values for the most recent timestamp
        """
        try:
            from crypto_ml_trading.features.technical_indicators import EnhancedTechnicalIndicators

            # Initialize indicator calculator
            calculator = EnhancedTechnicalIndicators()

            # Calculate all indicators
            df_with_indicators = calculator.calculate_all_indicators(df)

            if df_with_indicators is None or df_with_indicators.empty:
                return None

            # Get the latest row (most recent data)
            latest = df_with_indicators.iloc[-1]

            # Extract all indicator values
            indicators = {}
            for col in df_with_indicators.columns:
                if col not in ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                    value = latest[col]
                    # Handle NaN/Inf values
                    if pd.isna(value) or np.isinf(value):
                        indicators[col] = 0.0
                    else:
                        indicators[col] = float(value)

            # Add OHLCV data
            indicators['open'] = float(latest['open_price'])
            indicators['high'] = float(latest['high_price'])
            indicators['low'] = float(latest['low_price'])
            indicators['close'] = float(latest['close_price'])
            indicators['volume'] = float(latest['volume'])

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return None

    def _generate_single_prediction(
        self,
        model: ProfileModel,
        indicators: Dict,
        latest_price: float,
        db: Session,
        profile: TradingProfile
    ) -> Dict:
        """Generate prediction from a single model"""
        try:
            # Load model and scaler
            model_obj, scaler = self._load_model_and_scaler(model)

            if model_obj is None:
                return {
                    'success': False,
                    'reason': 'model_load_failed'
                }

            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(
                indicators=indicators,
                features_used=model.features
            )

            if feature_vector is None:
                return {
                    'success': False,
                    'reason': 'feature_preparation_failed'
                }

            # Scale features
            if scaler is not None:
                feature_vector_scaled = scaler.transform([feature_vector])
            else:
                feature_vector_scaled = [feature_vector]

            # Generate prediction
            # Different models have different output formats
            if hasattr(model_obj, 'predict_proba'):
                # Classification model (BUY/SELL/HOLD)
                proba = model_obj.predict_proba(feature_vector_scaled)[0]
                predicted_class = np.argmax(proba)
                confidence = float(proba[predicted_class])

                # Map class to signal
                signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
                signal = signal_map.get(predicted_class, 'HOLD')

                # Estimate price movement based on signal
                if signal == 'BUY':
                    price_prediction = latest_price * (1 + 0.01)  # +1% movement
                elif signal == 'SELL':
                    price_prediction = latest_price * (1 - 0.01)  # -1% movement
                else:
                    price_prediction = latest_price

            else:
                # Regression model (price prediction)
                price_prediction = float(model_obj.predict(feature_vector_scaled)[0])

                # Determine signal from price movement
                price_change = (price_prediction - latest_price) / latest_price

                if price_change > 0.005:  # > 0.5% increase
                    signal = 'BUY'
                    confidence = min(abs(price_change) * 100, 1.0)
                elif price_change < -0.005:  # > 0.5% decrease
                    signal = 'SELL'
                    confidence = min(abs(price_change) * 100, 1.0)
                else:
                    signal = 'HOLD'
                    confidence = 0.5

            # Determine direction
            if price_prediction > latest_price:
                direction = 'up'
            elif price_prediction < latest_price:
                direction = 'down'
            else:
                direction = 'neutral'

            # Calculate take profit and stop loss
            if signal == 'BUY':
                take_profit = latest_price * 1.02  # 2% profit
                stop_loss = latest_price * 0.98  # 2% loss
            elif signal == 'SELL':
                take_profit = latest_price * 0.98
                stop_loss = latest_price * 1.02
            else:
                take_profit = None
                stop_loss = None

            # Save prediction to database
            prediction = ProfilePrediction(
                profile_id=profile.id,
                model_id=model.id,
                timestamp=datetime.utcnow(),
                prediction_horizon='1h',  # Default 1-hour prediction
                price_prediction=price_prediction,
                direction_prediction=direction,
                confidence=confidence,
                signal=signal,
                signal_strength=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features=indicators  # Store all indicator values
            )

            db.add(prediction)
            db.commit()

            return {
                'success': True,
                'model_name': model.model_name,
                'signal': signal,
                'confidence': confidence,
                'price_prediction': price_prediction,
                'direction': direction
            }

        except Exception as e:
            logger.error(f"Error generating prediction for {model.model_name}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _load_model_and_scaler(
        self,
        model: ProfileModel
    ) -> Tuple[Optional[object], Optional[object]]:
        """Load model and scaler from disk (with caching)"""
        try:
            # Check cache
            if model.id in self.models_cache:
                return self.models_cache[model.id]

            # Load model from disk
            if not model.model_path or not os.path.exists(model.model_path):
                logger.warning(f"Model file not found: {model.model_path}")
                return None, None

            with open(model.model_path, 'rb') as f:
                model_obj = pickle.load(f)

            # Load scaler if exists
            scaler = None
            if model.scaler_path and os.path.exists(model.scaler_path):
                with open(model.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

            # Cache for future use
            self.models_cache[model.id] = (model_obj, scaler)

            return model_obj, scaler

        except Exception as e:
            logger.error(f"Error loading model {model.model_name}: {e}")
            return None, None

    def _prepare_feature_vector(
        self,
        indicators: Dict,
        features_used: Optional[List] = None
    ) -> Optional[np.ndarray]:
        """Prepare feature vector for model input"""
        try:
            if features_used is None:
                # Use all indicators
                feature_vector = list(indicators.values())
            else:
                # Use only specified features (in correct order)
                feature_vector = []
                for feature_name in features_used:
                    if feature_name in indicators:
                        feature_vector.append(indicators[feature_name])
                    else:
                        # Missing feature - use 0
                        feature_vector.append(0.0)

            return np.array(feature_vector)

        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None

    def clear_cache(self):
        """Clear model and indicator caches"""
        self.models_cache.clear()
        self.indicators_cache.clear()
        logger.info("Prediction service caches cleared")


# Singleton instance
prediction_service = PredictionService()
