"""
Accuracy Tracking Service
=========================

Tracks prediction accuracy by comparing predicted vs actual prices.
Runs hourly to update performance metrics for all models.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from database.models import (
    SessionLocal, TradingProfile, ProfileModel, ProfilePrediction,
    MarketData, ModelPerformanceMetrics
)

logger = logging.getLogger(__name__)


class AccuracyTracker:
    """Service for tracking model prediction accuracy"""

    def __init__(self):
        pass

    async def track_accuracy_all_models(self) -> Dict:
        """
        Track accuracy for all models that have predictions
        Called every hour by scheduler
        """
        db = SessionLocal()
        try:
            logger.info("Starting hourly accuracy tracking...")

            # Get all active profiles
            profiles = db.query(TradingProfile).filter(
                TradingProfile.is_active == True
            ).all()

            total_tracked = 0
            total_updated = 0

            for profile in profiles:
                try:
                    result = await self._track_accuracy_for_profile(profile, db)
                    total_tracked += result.get('predictions_tracked', 0)
                    total_updated += result.get('metrics_updated', 0)
                except Exception as e:
                    logger.error(f"Error tracking accuracy for {profile.symbol}: {e}")
                    continue

            logger.info(f"✅ Accuracy tracking complete - Tracked {total_tracked} predictions, "
                       f"Updated {total_updated} metrics")

            return {
                'success': True,
                'predictions_tracked': total_tracked,
                'metrics_updated': total_updated
            }

        except Exception as e:
            logger.error(f"Error in track_accuracy_all_models: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            db.close()

    async def _track_accuracy_for_profile(
        self,
        profile: TradingProfile,
        db: Session
    ) -> Dict:
        """Track accuracy for all models of a profile"""
        try:
            symbol = profile.symbol

            # Get all models with predictions
            models = db.query(ProfileModel).filter(
                ProfileModel.profile_id == profile.id
            ).all()

            predictions_tracked = 0
            metrics_updated = 0

            for model in models:
                # Update actual prices for predictions made 1+ hours ago
                updated_count = self._update_prediction_actuals(db, model, symbol)
                predictions_tracked += updated_count

                # Calculate performance metrics for different time windows
                for period in ['1h', '24h', '7d', '30d']:
                    metrics = self._calculate_performance_metrics(
                        db=db,
                        profile=profile,
                        model=model,
                        period=period
                    )

                    if metrics:
                        self._save_performance_metrics(db, profile, model, period, metrics)
                        metrics_updated += 1

            return {
                'predictions_tracked': predictions_tracked,
                'metrics_updated': metrics_updated
            }

        except Exception as e:
            logger.error(f"Error tracking accuracy for profile {profile.symbol}: {e}")
            return {
                'predictions_tracked': 0,
                'metrics_updated': 0
            }

    def _update_prediction_actuals(
        self,
        db: Session,
        model: ProfileModel,
        symbol: str
    ) -> int:
        """
        Update predictions with actual prices
        Only update predictions that are 1+ hours old and don't have actual_price set
        """
        try:
            # Get predictions without actual prices (older than 1 hour)
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)

            predictions = db.query(ProfilePrediction).filter(
                ProfilePrediction.model_id == model.id,
                ProfilePrediction.actual_price == None,
                ProfilePrediction.timestamp < one_hour_ago
            ).all()

            updated_count = 0

            for prediction in predictions:
                # Get actual price from market data at prediction_horizon time
                # For example, if prediction was at 10:00 AM with horizon='1h',
                # get actual price from 11:00 AM
                target_time = prediction.timestamp + self._parse_horizon(
                    prediction.prediction_horizon
                )

                # Find closest market data point
                actual_data = db.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp >= target_time
                ).order_by(MarketData.timestamp).first()

                if actual_data:
                    prediction.actual_price = actual_data.close_price

                    # Calculate prediction error
                    if prediction.price_prediction:
                        prediction.prediction_error = abs(
                            prediction.price_prediction - prediction.actual_price
                        )

                    # Calculate actual high/low for the period
                    period_data = db.query(MarketData).filter(
                        MarketData.symbol == symbol,
                        MarketData.timestamp >= prediction.timestamp,
                        MarketData.timestamp <= target_time
                    ).all()

                    if period_data:
                        prediction.actual_high = max(d.high_price for d in period_data)
                        prediction.actual_low = min(d.low_price for d in period_data)

                    updated_count += 1

            if updated_count > 0:
                db.commit()
                logger.info(f"{symbol}/{model.model_name}: Updated {updated_count} predictions with actual prices")

            return updated_count

        except Exception as e:
            logger.error(f"Error updating prediction actuals: {e}")
            db.rollback()
            return 0

    def _calculate_performance_metrics(
        self,
        db: Session,
        profile: TradingProfile,
        model: ProfileModel,
        period: str
    ) -> Optional[Dict]:
        """Calculate performance metrics for a time window"""
        try:
            # Get time range
            window_start, window_end = self._get_time_window(period)

            # Get all predictions with actual prices in this window
            predictions = db.query(ProfilePrediction).filter(
                ProfilePrediction.profile_id == profile.id,
                ProfilePrediction.model_id == model.id,
                ProfilePrediction.timestamp >= window_start,
                ProfilePrediction.timestamp <= window_end,
                ProfilePrediction.actual_price != None
            ).all()

            if not predictions or len(predictions) < 5:
                # Not enough predictions to calculate meaningful metrics
                return None

            # Extract data
            predicted_prices = [p.price_prediction for p in predictions if p.price_prediction]
            actual_prices = [p.actual_price for p in predictions if p.actual_price]
            errors = [p.prediction_error for p in predictions if p.prediction_error]

            # Calculate price prediction metrics
            mae = np.mean(errors) if errors else None
            rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else None

            # MAPE (Mean Absolute Percentage Error)
            mape = None
            if predicted_prices and actual_prices:
                percentage_errors = [
                    abs((pred - actual) / actual) * 100
                    for pred, actual in zip(predicted_prices, actual_prices)
                    if actual != 0
                ]
                mape = np.mean(percentage_errors) if percentage_errors else None

            # R-squared
            r2_score = None
            if len(predicted_prices) > 1 and len(actual_prices) > 1:
                ss_res = sum((a - p)**2 for a, p in zip(actual_prices, predicted_prices))
                ss_tot = sum((a - np.mean(actual_prices))**2 for a in actual_prices)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else None

            # Signal accuracy
            signals = [p.signal for p in predictions if p.signal]
            total_signals = len(signals)

            correct_signals = 0
            buy_signals = 0
            sell_signals = 0
            correct_buys = 0
            correct_sells = 0

            for pred in predictions:
                if not pred.signal or not pred.actual_price:
                    continue

                # Get initial price (from features)
                initial_price = pred.features.get('close') if pred.features else None
                if not initial_price:
                    continue

                actual_movement = pred.actual_price - initial_price

                # Check if signal was correct
                if pred.signal == 'BUY':
                    buy_signals += 1
                    if actual_movement > 0:
                        correct_signals += 1
                        correct_buys += 1
                elif pred.signal == 'SELL':
                    sell_signals += 1
                    if actual_movement < 0:
                        correct_signals += 1
                        correct_sells += 1
                elif pred.signal == 'HOLD':
                    # HOLD is correct if movement is small
                    if abs(actual_movement) < initial_price * 0.005:
                        correct_signals += 1

            signal_accuracy = (correct_signals / total_signals * 100) if total_signals > 0 else None
            buy_accuracy = (correct_buys / buy_signals * 100) if buy_signals > 0 else None
            sell_accuracy = (correct_sells / sell_signals * 100) if sell_signals > 0 else None

            # Direction accuracy
            up_predictions = 0
            down_predictions = 0
            correct_directions = 0

            for pred in predictions:
                if not pred.direction_prediction or not pred.actual_price:
                    continue

                initial_price = pred.features.get('close') if pred.features else None
                if not initial_price:
                    continue

                actual_direction = 'up' if pred.actual_price > initial_price else 'down'

                if pred.direction_prediction == 'up':
                    up_predictions += 1
                elif pred.direction_prediction == 'down':
                    down_predictions += 1

                if pred.direction_prediction == actual_direction:
                    correct_directions += 1

            direction_accuracy = (correct_directions / len(predictions) * 100) if predictions else None

            # Calculate confidence metrics
            confidences = [p.confidence for p in predictions if p.confidence]
            avg_confidence = np.mean(confidences) if confidences else None

            # Error distribution
            error_stats = None
            if errors:
                error_stats = {
                    'min': float(np.min(errors)),
                    'max': float(np.max(errors)),
                    'std': float(np.std(errors)),
                    'percentiles': {
                        '25': float(np.percentile(errors, 25)),
                        '50': float(np.percentile(errors, 50)),
                        '75': float(np.percentile(errors, 75)),
                        '95': float(np.percentile(errors, 95))
                    }
                }

            return {
                'total_predictions': len(predictions),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2_score,
                'signal_accuracy': signal_accuracy,
                'buy_signal_accuracy': buy_accuracy,
                'sell_signal_accuracy': sell_accuracy,
                'total_signals': total_signals,
                'correct_signals': correct_signals,
                'direction_accuracy': direction_accuracy,
                'up_predictions': up_predictions,
                'down_predictions': down_predictions,
                'correct_directions': correct_directions,
                'avg_confidence': avg_confidence,
                'error_stats': error_stats,
                'window_start': window_start,
                'window_end': window_end
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return None

    def _save_performance_metrics(
        self,
        db: Session,
        profile: TradingProfile,
        model: ProfileModel,
        period: str,
        metrics: Dict
    ):
        """Save performance metrics to database"""
        try:
            # Create new metrics record
            perf_metrics = ModelPerformanceMetrics(
                profile_id=profile.id,
                model_id=model.id,
                timestamp=datetime.utcnow(),
                evaluation_period=period,
                window_start=metrics['window_start'],
                window_end=metrics['window_end'],
                total_predictions=metrics['total_predictions'],
                mae=metrics.get('mae'),
                rmse=metrics.get('rmse'),
                mape=metrics.get('mape'),
                r2_score=metrics.get('r2_score'),
                signal_accuracy=metrics.get('signal_accuracy'),
                buy_signal_accuracy=metrics.get('buy_signal_accuracy'),
                sell_signal_accuracy=metrics.get('sell_signal_accuracy'),
                total_signals=metrics.get('total_signals', 0),
                correct_signals=metrics.get('correct_signals', 0),
                direction_accuracy=metrics.get('direction_accuracy'),
                up_predictions=metrics.get('up_predictions', 0),
                down_predictions=metrics.get('down_predictions', 0),
                correct_directions=metrics.get('correct_directions', 0),
                avg_confidence=metrics.get('avg_confidence'),
                error_stats=metrics.get('error_stats'),
                model_version=model.model_version
            )

            db.add(perf_metrics)
            db.commit()

            logger.debug(f"{profile.symbol}/{model.model_name}: Saved {period} performance metrics")

        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            db.rollback()

    def _parse_horizon(self, horizon: str) -> timedelta:
        """Parse prediction horizon string to timedelta"""
        try:
            if horizon.endswith('h'):
                hours = int(horizon[:-1])
                return timedelta(hours=hours)
            elif horizon.endswith('d'):
                days = int(horizon[:-1])
                return timedelta(days=days)
            else:
                # Default to 1 hour
                return timedelta(hours=1)
        except:
            return timedelta(hours=1)

    def _get_time_window(self, period: str) -> tuple:
        """Get start and end time for a period"""
        end_time = datetime.utcnow()

        if period == '1h':
            start_time = end_time - timedelta(hours=1)
        elif period == '24h':
            start_time = end_time - timedelta(days=1)
        elif period == '7d':
            start_time = end_time - timedelta(days=7)
        elif period == '30d':
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)

        return start_time, end_time


# Singleton instance
accuracy_tracker = AccuracyTracker()
