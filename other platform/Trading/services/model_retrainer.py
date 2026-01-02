"""
Model Retraining Service
========================

Handles monthly model retraining with smart deployment:
- Trains new version of each model with latest data
- Compares performance with current deployed model
- Deploys new model only if performance improves or stays same
- Keeps complete version history for rollback
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import (
    SessionLocal, TradingProfile, ProfileModel, ModelPerformanceMetrics,
    ModelStatus
)

logger = logging.getLogger(__name__)


class ModelRetrainer:
    """Service for monthly model retraining"""

    def __init__(self):
        pass

    async def retrain_all_models(self) -> Dict:
        """
        Retrain all models for all active profiles
        Called on 1st of month at 2 AM by scheduler
        """
        db = SessionLocal()
        try:
            logger.info("=" * 80)
            logger.info("Starting monthly model retraining...")
            logger.info("=" * 80)

            # Get all active profiles
            profiles = db.query(TradingProfile).filter(
                TradingProfile.is_active == True,
                TradingProfile.has_data == True
            ).all()

            total_retrained = 0
            total_deployed = 0
            total_kept_old = 0

            for profile in profiles:
                try:
                    result = await self._retrain_profile_models(profile, db)
                    total_retrained += result.get('models_retrained', 0)
                    total_deployed += result.get('models_deployed', 0)
                    total_kept_old += result.get('kept_old_version', 0)
                except Exception as e:
                    logger.error(f"Error retraining models for {profile.symbol}: {e}")
                    continue

            logger.info("=" * 80)
            logger.info(f"✅ Monthly retraining complete!")
            logger.info(f"  - Models retrained: {total_retrained}")
            logger.info(f"  - New models deployed: {total_deployed}")
            logger.info(f"  - Kept old versions: {total_kept_old}")
            logger.info("=" * 80)

            return {
                'success': True,
                'models_retrained': total_retrained,
                'models_deployed': total_deployed,
                'kept_old_version': total_kept_old
            }

        except Exception as e:
            logger.error(f"Error in retrain_all_models: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            db.close()

    async def _retrain_profile_models(
        self,
        profile: TradingProfile,
        db: Session
    ) -> Dict:
        """Retrain all models for a single profile"""
        try:
            symbol = profile.symbol
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Retraining models for {symbol}")
            logger.info(f"{'=' * 60}")

            # Get all deployed models for this profile
            deployed_models = db.query(ProfileModel).filter(
                ProfileModel.profile_id == profile.id,
                ProfileModel.is_deployed == True
            ).all()

            if not deployed_models:
                logger.info(f"{symbol}: No deployed models to retrain")
                return {
                    'models_retrained': 0,
                    'models_deployed': 0,
                    'kept_old_version': 0
                }

            models_retrained = 0
            models_deployed = 0
            kept_old = 0

            for old_model in deployed_models:
                try:
                    logger.info(f"\nRetraining {old_model.model_name}...")

                    # Train new version
                    new_model_result = await self._train_new_model_version(
                        profile=profile,
                        old_model=old_model,
                        db=db
                    )

                    if not new_model_result['success']:
                        logger.warning(f"  Failed to train new version: {new_model_result.get('error')}")
                        continue

                    models_retrained += 1
                    new_model = new_model_result['model']

                    # Compare performance
                    should_deploy = self._should_deploy_new_model(
                        old_model=old_model,
                        new_model=new_model,
                        db=db
                    )

                    if should_deploy:
                        # Deploy new model
                        self._deploy_new_model(old_model, new_model, db)
                        models_deployed += 1
                        logger.info(f"  ✅ New model deployed! Performance improved.")
                    else:
                        # Keep old model
                        new_model.is_deployed = False
                        new_model.status = ModelStatus.TRAINED
                        db.commit()
                        kept_old += 1
                        logger.info(f"  ℹ️  Kept old model. New model performance did not improve.")

                except Exception as e:
                    logger.error(f"  Error retraining {old_model.model_name}: {e}")
                    continue

            return {
                'models_retrained': models_retrained,
                'models_deployed': models_deployed,
                'kept_old_version': kept_old
            }

        except Exception as e:
            logger.error(f"Error retraining profile models: {e}")
            return {
                'models_retrained': 0,
                'models_deployed': 0,
                'kept_old_version': 0
            }

    async def _train_new_model_version(
        self,
        profile: TradingProfile,
        old_model: ProfileModel,
        db: Session
    ) -> Dict:
        """
        Train a new version of the model
        This will call the ML pipeline training endpoint
        """
        try:
            # Import here to avoid circular dependency
            from api.routers.ml_pipeline import train_model_for_profile

            # Train new model using ML pipeline
            result = await train_model_for_profile(
                profile_id=profile.id,
                model_type=old_model.model_type,
                db=db
            )

            if not result['success']:
                return {
                    'success': False,
                    'error': result.get('error', 'Training failed')
                }

            # Get the newly created model
            new_model = db.query(ProfileModel).filter(
                ProfileModel.id == result['model_id']
            ).first()

            if not new_model:
                return {
                    'success': False,
                    'error': 'Model not found after training'
                }

            return {
                'success': True,
                'model': new_model
            }

        except Exception as e:
            logger.error(f"Error training new model version: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _should_deploy_new_model(
        self,
        old_model: ProfileModel,
        new_model: ProfileModel,
        db: Session
    ) -> bool:
        """
        Determine if new model should be deployed based on performance comparison

        Deploy if new model >= old model on at least 2 out of 3 key metrics:
        1. Signal accuracy
        2. Sharpe ratio
        3. Win rate
        """
        try:
            # Get latest performance metrics for both models (30-day window)
            old_metrics = db.query(ModelPerformanceMetrics).filter(
                ModelPerformanceMetrics.model_id == old_model.id,
                ModelPerformanceMetrics.evaluation_period == '30d'
            ).order_by(
                desc(ModelPerformanceMetrics.timestamp)
            ).first()

            new_metrics = db.query(ModelPerformanceMetrics).filter(
                ModelPerformanceMetrics.model_id == new_model.id,
                ModelPerformanceMetrics.evaluation_period == '30d'
            ).order_by(
                desc(ModelPerformanceMetrics.timestamp)
            ).first()

            # If no metrics for old model, deploy new
            if not old_metrics:
                logger.info("  No metrics for old model, deploying new model")
                return True

            # If no metrics for new model, don't deploy
            if not new_metrics:
                logger.warning("  No metrics for new model yet, keeping old model")
                return False

            # Compare 3 key metrics
            improvements = 0
            total_comparisons = 0

            # Metric 1: Signal accuracy
            if old_metrics.signal_accuracy and new_metrics.signal_accuracy:
                total_comparisons += 1
                if new_metrics.signal_accuracy >= old_metrics.signal_accuracy:
                    improvements += 1
                    logger.info(f"  Signal Accuracy: {new_metrics.signal_accuracy:.2f}% >= {old_metrics.signal_accuracy:.2f}% ✓")
                else:
                    logger.info(f"  Signal Accuracy: {new_metrics.signal_accuracy:.2f}% < {old_metrics.signal_accuracy:.2f}% ✗")

            # Metric 2: Sharpe ratio (if available)
            if old_metrics.sharpe_ratio and new_metrics.sharpe_ratio:
                total_comparisons += 1
                if new_metrics.sharpe_ratio >= old_metrics.sharpe_ratio:
                    improvements += 1
                    logger.info(f"  Sharpe Ratio: {new_metrics.sharpe_ratio:.3f} >= {old_metrics.sharpe_ratio:.3f} ✓")
                else:
                    logger.info(f"  Sharpe Ratio: {new_metrics.sharpe_ratio:.3f} < {old_metrics.sharpe_ratio:.3f} ✗")

            # Metric 3: Win rate (if available)
            if old_metrics.win_rate and new_metrics.win_rate:
                total_comparisons += 1
                if new_metrics.win_rate >= old_metrics.win_rate:
                    improvements += 1
                    logger.info(f"  Win Rate: {new_metrics.win_rate:.2f}% >= {old_metrics.win_rate:.2f}% ✓")
                else:
                    logger.info(f"  Win Rate: {new_metrics.win_rate:.2f}% < {old_metrics.win_rate:.2f}% ✗")

            # Deploy if improved on at least 2/3 metrics
            if total_comparisons == 0:
                logger.warning("  No comparable metrics, keeping old model")
                return False

            threshold = max(2, int(total_comparisons * 0.67))  # At least 2/3
            should_deploy = improvements >= threshold

            logger.info(f"  Performance comparison: {improvements}/{total_comparisons} metrics improved")
            logger.info(f"  Decision: {'DEPLOY NEW MODEL' if should_deploy else 'KEEP OLD MODEL'}")

            return should_deploy

        except Exception as e:
            logger.error(f"Error comparing model performance: {e}")
            # On error, keep old model (conservative approach)
            return False

    def _deploy_new_model(
        self,
        old_model: ProfileModel,
        new_model: ProfileModel,
        db: Session
    ):
        """Deploy new model and archive old model"""
        try:
            # Undeploy old model
            old_model.is_deployed = False
            old_model.status = ModelStatus.TRAINED  # Keep as trained for rollback

            # Deploy new model
            new_model.is_deployed = True
            new_model.is_primary = True
            new_model.status = ModelStatus.DEPLOYED
            new_model.deployed_at = datetime.utcnow()

            db.commit()

            logger.info(f"  Model deployment complete:")
            logger.info(f"    Old version: {old_model.model_version} (archived)")
            logger.info(f"    New version: {new_model.model_version} (deployed)")

        except Exception as e:
            logger.error(f"Error deploying new model: {e}")
            db.rollback()
            raise


# Singleton instance
model_retrainer = ModelRetrainer()
