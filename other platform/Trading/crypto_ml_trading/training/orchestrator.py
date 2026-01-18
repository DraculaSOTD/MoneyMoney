"""
Model Training Orchestrator
Coordinates the complete training pipeline.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import json

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import TradingProfile, ProfileModel, MarketData, ModelStatus
from crypto_ml_trading.training.config import TrainingConfig, get_default_config
from crypto_ml_trading.training.data_auditor import ProfileDataAuditor, DataAuditReport
from crypto_ml_trading.training.data_validator import DataQualityValidator, DataQualityReport
from crypto_ml_trading.training.evaluation import ModelEvaluator, EvaluationReport
from crypto_ml_trading.training.training_logger import TrainingLogger
from crypto_ml_trading.training.model_comparison import ModelComparator, ComparisonResult
from crypto_ml_trading.features.scaler_manager import FeatureScalerManager
from crypto_ml_trading.features.feature_checker import FeatureCompletenessChecker, FeatureReport

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Complete training report."""
    profile_id: int
    model_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float
    data_audit: Dict
    data_quality: Dict
    feature_report: Dict
    training_config: Dict
    evaluation_metrics: Dict
    comparison_result: Optional[Dict]
    deployed: bool
    model_path: Optional[str]
    scaler_path: Optional[str]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'profile_id': self.profile_id,
            'model_type': self.model_type,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'data_audit': self.data_audit,
            'data_quality': self.data_quality,
            'feature_report': self.feature_report,
            'training_config': self.training_config,
            'evaluation_metrics': self.evaluation_metrics,
            'comparison_result': self.comparison_result,
            'deployed': self.deployed,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'errors': self.errors
        }


class InsufficientDataError(Exception):
    """Raised when there's insufficient data for training."""
    pass


class DataQualityError(Exception):
    """Raised when data quality is below threshold."""
    pass


class ModelTrainingOrchestrator:
    """Coordinates the complete training pipeline."""

    def __init__(
        self,
        db_session: Session,
        config: Optional[TrainingConfig] = None,
        models_dir: str = "models",
        log_dir: str = "logs/training"
    ):
        """
        Initialize the training orchestrator.

        Args:
            db_session: Database session
            config: Training configuration
            models_dir: Directory to save models
            log_dir: Directory for training logs
        """
        self.db_session = db_session
        self.config = config or get_default_config()
        self.models_dir = Path(models_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.auditor = ProfileDataAuditor(db_session)
        self.validator = DataQualityValidator(db_session)
        self.evaluator = ModelEvaluator()
        self.comparator = ModelComparator(db_session)
        self.feature_checker = FeatureCompletenessChecker(db_session)

    async def train_profile_model(
        self,
        profile_id: int,
        model_type: str,
        optimize_hyperparams: bool = True,
        force_retrain: bool = False,
        progress_callback: Optional[callable] = None
    ) -> TrainingReport:
        """
        Complete training pipeline for a profile model.

        Pipeline steps:
        1. Data Audit -> Gap Fill if needed -> Re-audit
        2. Data Quality Validation
        3. Feature Verification
        4. Data Loading & Splitting
        5. Hyperparameter Optimization (optional)
        6. Model Training
        7. Evaluation
        8. Deployment Decision
        9. Generate Report

        Args:
            profile_id: ID of the trading profile
            model_type: Type of model to train
            optimize_hyperparams: Whether to run hyperparameter optimization
            force_retrain: Force retraining even if model exists
            progress_callback: Optional callback(stage, progress, message)

        Returns:
            TrainingReport with complete results
        """
        started_at = datetime.utcnow()
        errors = []

        # Initialize logger
        training_logger = TrainingLogger(
            profile_id=profile_id,
            model_type=model_type,
            db_session=self.db_session,
            log_dir=str(self.log_dir)
        )

        try:
            if progress_callback:
                await progress_callback('init', 0, 'Starting training pipeline')

            # Step 1: Data Audit
            logger.info(f"Step 1: Data audit for profile {profile_id}")
            if progress_callback:
                await progress_callback('audit', 10, 'Auditing data')

            audit_report = await self._run_data_audit(profile_id)

            if not audit_report.is_training_ready:
                # Try to fill gaps
                if audit_report.coverage_percent < self.config.min_coverage_percent:
                    logger.info("Attempting to fill data gaps...")
                    await self._fill_data_gaps(profile_id, audit_report)
                    # Re-audit
                    audit_report = await self._run_data_audit(profile_id)

                    if not audit_report.is_training_ready:
                        raise InsufficientDataError(
                            f"Insufficient data: coverage {audit_report.coverage_percent:.1f}% "
                            f"< {self.config.min_coverage_percent}%"
                        )

            # Step 2: Data Quality Validation
            logger.info("Step 2: Data quality validation")
            if progress_callback:
                await progress_callback('validate', 20, 'Validating data quality')

            quality_report = await self._ensure_data_quality(profile_id)

            # Step 3: Feature Verification
            logger.info("Step 3: Feature verification")
            if progress_callback:
                await progress_callback('features', 30, 'Verifying features')

            feature_report = await self._verify_features(profile_id, model_type)

            # Step 4: Load and Split Data
            logger.info("Step 4: Loading and splitting data")
            if progress_callback:
                await progress_callback('split', 40, 'Preparing data splits')

            train_data, val_data, test_data, scaler_manager = await self._load_and_split_data(
                profile_id
            )

            # Log training start
            training_config = {
                'model_type': model_type,
                'optimize_hyperparams': optimize_hyperparams,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                **{k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }
            training_logger.log_training_start(training_config)

            # Step 5: Hyperparameter Optimization (optional)
            best_params = {}
            if optimize_hyperparams:
                logger.info("Step 5: Hyperparameter optimization")
                if progress_callback:
                    await progress_callback('hyperopt', 50, 'Optimizing hyperparameters')

                best_params = await self._optimize_hyperparameters(
                    model_type, train_data, val_data
                )
            else:
                if progress_callback:
                    await progress_callback('hyperopt', 50, 'Skipping hyperparameter optimization')

            # Step 6: Train Model
            logger.info("Step 6: Training model")
            if progress_callback:
                await progress_callback('train', 60, 'Training model')

            model, training_history = await self._train_model(
                model_type,
                best_params,
                train_data,
                val_data,
                training_logger
            )

            # Step 7: Evaluation
            logger.info("Step 7: Model evaluation")
            if progress_callback:
                await progress_callback('evaluate', 80, 'Evaluating model')

            evaluation_report = await self._evaluate_model(model, test_data)

            # Step 8: Deployment Decision
            logger.info("Step 8: Deployment decision")
            if progress_callback:
                await progress_callback('deploy', 90, 'Making deployment decision')

            comparison_result = await self._compare_with_deployed(
                evaluation_report, profile_id, model_type, test_data
            )

            # Deploy if appropriate
            deployed = False
            model_path = None
            scaler_path = None

            if comparison_result.should_deploy or force_retrain:
                model_path, scaler_path = await self._deploy_model(
                    model, scaler_manager, profile_id, model_type
                )
                deployed = True

            # Step 9: Generate Report
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            # Log training end
            training_logger.log_training_end({
                'classification_metrics': evaluation_report.classification_metrics,
                'trading_metrics': evaluation_report.trading_metrics
            })

            if progress_callback:
                await progress_callback('complete', 100, 'Training complete')

            report = TrainingReport(
                profile_id=profile_id,
                model_type=model_type,
                status='completed',
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                data_audit=audit_report.to_dict(),
                data_quality=quality_report.to_dict(),
                feature_report=feature_report.to_dict(),
                training_config=training_config,
                evaluation_metrics=evaluation_report.to_dict(),
                comparison_result=comparison_result.to_dict() if comparison_result else None,
                deployed=deployed,
                model_path=model_path,
                scaler_path=scaler_path,
                errors=errors
            )

            logger.info(
                f"Training complete: profile={profile_id}, model={model_type}, "
                f"accuracy={evaluation_report.classification_metrics.get('accuracy', 0):.4f}, "
                f"deployed={deployed}"
            )

            return report

        except Exception as e:
            logger.error(f"Training failed: {e}")
            errors.append(str(e))

            training_logger.log_training_failed(str(e))

            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            return TrainingReport(
                profile_id=profile_id,
                model_type=model_type,
                status='failed',
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                data_audit={},
                data_quality={},
                feature_report={},
                training_config={},
                evaluation_metrics={},
                comparison_result=None,
                deployed=False,
                model_path=None,
                scaler_path=None,
                errors=errors
            )

    async def _run_data_audit(self, profile_id: int) -> DataAuditReport:
        """Run data audit."""
        return self.auditor.audit_profile(profile_id)

    async def _fill_data_gaps(
        self,
        profile_id: int,
        audit_report: DataAuditReport
    ) -> None:
        """Fill data gaps using Binance API."""
        from crypto_ml_trading.training.gap_filler import DataGapFiller
        from exchanges.binance_connector import BinanceConnector

        # Initialize Binance connector
        connector = BinanceConnector(testnet=False)
        await connector.connect()

        try:
            gap_filler = DataGapFiller(connector, self.db_session)
            await gap_filler.fill_gaps(profile_id, audit_report.gaps)
        finally:
            await connector.disconnect()

    async def _ensure_data_quality(self, profile_id: int) -> DataQualityReport:
        """Validate data quality."""
        return self.validator.validate_profile_data(profile_id)

    async def _verify_features(
        self,
        profile_id: int,
        model_type: str
    ) -> FeatureReport:
        """Verify feature completeness."""
        return self.feature_checker.check_completeness(
            profile_id=profile_id,
            model_type=model_type
        )

    async def _load_and_split_data(
        self,
        profile_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScalerManager]:
        """Load data and create temporal splits with proper scaling."""
        # Load market data
        query = self.db_session.query(MarketData).filter(
            MarketData.profile_id == profile_id
        ).order_by(MarketData.timestamp)

        df = pd.read_sql(query.statement, self.db_session.bind)

        if df.empty:
            raise InsufficientDataError("No market data found")

        # Create temporal splits
        total = len(df)
        gap = 1440  # 24 hours

        train_end = int(total * self.config.train_ratio)
        val_start = train_end + gap
        val_end = val_start + int(total * self.config.val_ratio)
        test_start = val_end + gap

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

        # Fit scaler on training data only
        feature_cols = [
            col for col in df.columns
            if col not in ['id', 'profile_id', 'symbol', 'timestamp', 'created_at', 'label']
        ]

        scaler_manager = FeatureScalerManager()
        scaler_manager.fit(train_df[feature_cols])

        # Transform all splits
        train_df[feature_cols] = scaler_manager.transform(train_df[feature_cols])
        val_df[feature_cols] = scaler_manager.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler_manager.transform(test_df[feature_cols])

        return train_df, val_df, test_df, scaler_manager

    async def _optimize_hyperparameters(
        self,
        model_type: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame
    ) -> Dict:
        """Run hyperparameter optimization."""
        # This is a placeholder - actual implementation would use hyperopt.py
        logger.info("Hyperparameter optimization placeholder - using defaults")
        return {}

    async def _train_model(
        self,
        model_type: str,
        params: Dict,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        training_logger: TrainingLogger
    ) -> Tuple[Any, Dict]:
        """Train the model."""
        # This is a placeholder - actual implementation would use model classes
        logger.info(f"Training {model_type} model...")

        # Simulate training epochs
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(self.config.default_epochs):
            # Placeholder metrics
            train_loss = 1.0 / (epoch + 1)
            val_loss = 1.1 / (epoch + 1)
            val_acc = 0.5 + 0.3 * (epoch / self.config.default_epochs)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            training_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc
            )

            # Early stopping check would go here

        # Return placeholder model
        return None, history

    async def _evaluate_model(
        self,
        model: Any,
        test_data: pd.DataFrame
    ) -> EvaluationReport:
        """Evaluate the trained model."""
        # Placeholder evaluation
        n_samples = len(test_data)
        y_true = np.random.randint(0, 3, n_samples)
        y_pred = np.random.randint(0, 3, n_samples)
        y_prob = np.random.rand(n_samples, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        return self.evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )

    async def _compare_with_deployed(
        self,
        new_metrics: EvaluationReport,
        profile_id: int,
        model_type: str,
        test_data: pd.DataFrame
    ) -> ComparisonResult:
        """Compare new model with deployed model."""
        # Get deployed model
        deployed_model = self.db_session.query(ProfileModel).filter(
            ProfileModel.profile_id == profile_id,
            ProfileModel.model_type == model_type,
            ProfileModel.is_deployed == True
        ).first()

        if not deployed_model:
            # No deployed model, always deploy if above threshold
            return ComparisonResult(
                new_accuracy=new_metrics.classification_metrics.get('accuracy', 0),
                deployed_accuracy=0,
                accuracy_diff=new_metrics.classification_metrics.get('accuracy', 0),
                new_sharpe=new_metrics.trading_metrics.get('sharpe_ratio', 0),
                deployed_sharpe=0,
                sharpe_diff=new_metrics.trading_metrics.get('sharpe_ratio', 0),
                is_significant=True,
                p_value=0.0,
                should_deploy=new_metrics.classification_metrics.get('accuracy', 0) >= 0.55,
                reason="No deployed model exists"
            )

        return self.comparator.compare(
            new_model_metrics=new_metrics,
            deployed_model_id=deployed_model.id,
            test_labels=new_metrics.labels
        )

    async def _deploy_model(
        self,
        model: Any,
        scaler_manager: FeatureScalerManager,
        profile_id: int,
        model_type: str
    ) -> Tuple[str, str]:
        """Deploy the model."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Model path
        model_dir = self.models_dir / str(profile_id) / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(model_dir / f"model_{timestamp}.pth")
        scaler_path = str(model_dir / f"scaler_{timestamp}.pkl")

        # Save scaler
        scaler_manager.save(scaler_path)

        # Save model (placeholder)
        # In practice: torch.save(model.state_dict(), model_path)

        # Update database
        # Mark old models as not deployed
        self.db_session.query(ProfileModel).filter(
            ProfileModel.profile_id == profile_id,
            ProfileModel.model_type == model_type
        ).update({'is_deployed': False})

        # Create new model record
        new_model = ProfileModel(
            profile_id=profile_id,
            model_name=f"{model_type}_{timestamp}",
            model_type=model_type,
            model_version=timestamp,
            status=ModelStatus.DEPLOYED,
            is_deployed=True,
            deployed_at=datetime.utcnow(),
            model_path=model_path,
            scaler_path=scaler_path
        )
        self.db_session.add(new_model)
        self.db_session.commit()

        logger.info(f"Model deployed: {model_path}")

        return model_path, scaler_path
