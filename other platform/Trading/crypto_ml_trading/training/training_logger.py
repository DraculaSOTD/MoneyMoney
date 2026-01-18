"""
Training Logger
Dual logging to files and database for model training.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import Session

from database.models import TradingProfile, ModelTrainingHistory


@dataclass
class EpochLog:
    """Log entry for a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    metrics: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TrainingLogger:
    """Dual logging to files and database."""

    def __init__(
        self,
        profile_id: int,
        model_type: str,
        db_session: Session,
        log_dir: str = "logs/training",
        run_id: Optional[str] = None
    ):
        self.profile_id = profile_id
        self.model_type = model_type
        self.db_session = db_session
        self.log_dir = Path(log_dir)

        # Generate run ID
        self.run_id = run_id or f"{model_type}_{profile_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize file logger
        self.file_logger = self._setup_file_logger()

        # Training state
        self.training_history_id: Optional[int] = None
        self.epoch_logs: List[Dict] = []
        self.start_time: Optional[datetime] = None
        self.config: Optional[Dict] = None
        self.alerts: List[Dict] = []

    def _setup_file_logger(self) -> logging.Logger:
        """Set up file-based logger."""
        logger = logging.getLogger(f"training.{self.run_id}")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        logger.handlers = []

        # File handler
        log_file = self.log_dir / f"{self.run_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Also add stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def log_training_start(self, config: Dict) -> int:
        """
        Log training start to both file and database.

        Args:
            config: Training configuration

        Returns:
            Training history ID
        """
        self.start_time = datetime.utcnow()
        self.config = config

        # Log to file
        self.file_logger.info(f"Training started for profile {self.profile_id}, model {self.model_type}")
        self.file_logger.info(f"Run ID: {self.run_id}")
        self.file_logger.debug(f"Configuration: {json.dumps(config, indent=2, default=str)}")

        # Create database entry
        training_history = ModelTrainingHistory(
            profile_id=self.profile_id,
            run_id=self.run_id,
            started_at=self.start_time,
            status='running',
            parameters=config,
            training_logs=[]
        )

        self.db_session.add(training_history)
        self.db_session.commit()

        self.training_history_id = training_history.id
        self.file_logger.info(f"Database record created: ID {self.training_history_id}")

        return self.training_history_id

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Optional[Dict] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Log a training epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
        """
        epoch_log = EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            metrics=metrics or {}
        )

        self.epoch_logs.append(asdict(epoch_log))

        # Log to file
        msg = f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        if val_accuracy is not None:
            msg += f", val_acc={val_accuracy:.4f}"
        if learning_rate is not None:
            msg += f", lr={learning_rate:.6f}"

        self.file_logger.info(msg)

        # Update database periodically (every 10 epochs)
        if epoch % 10 == 0:
            self._update_db_training_logs()

    def log_training_end(self, report: Dict) -> None:
        """
        Log training completion.

        Args:
            report: Final evaluation report
        """
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0

        # Log to file
        self.file_logger.info(f"Training completed in {duration:.1f} seconds")
        self.file_logger.info(f"Final metrics: {json.dumps(report.get('classification_metrics', {}), indent=2)}")

        # Update database
        if self.training_history_id:
            history = self.db_session.query(ModelTrainingHistory).filter(
                ModelTrainingHistory.id == self.training_history_id
            ).first()

            if history:
                history.completed_at = end_time
                history.duration = duration
                history.status = 'completed'
                history.training_logs = self.epoch_logs

                # Extract metrics
                class_metrics = report.get('classification_metrics', {})
                history.final_train_loss = self.epoch_logs[-1].get('train_loss') if self.epoch_logs else None
                history.final_val_loss = self.epoch_logs[-1].get('val_loss') if self.epoch_logs else None
                history.val_accuracy = class_metrics.get('accuracy')
                history.test_accuracy = class_metrics.get('test_accuracy')

                # Trading metrics
                trading_metrics = report.get('trading_metrics', {})
                history.backtest_sharpe = trading_metrics.get('sharpe_ratio')
                history.backtest_returns = trading_metrics.get('total_return')
                history.backtest_max_drawdown = trading_metrics.get('max_drawdown')
                history.backtest_win_rate = trading_metrics.get('win_rate')

                # Find best epoch
                if self.epoch_logs:
                    best_epoch = min(self.epoch_logs, key=lambda x: x.get('val_loss', float('inf')))
                    history.best_epoch = best_epoch.get('epoch')
                    history.best_val_loss = best_epoch.get('val_loss')

                self.db_session.commit()

        # Save final report to file
        report_file = self.log_dir / f"{self.run_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'profile_id': self.profile_id,
                'model_type': self.model_type,
                'duration_seconds': duration,
                'epoch_count': len(self.epoch_logs),
                'config': self.config,
                'final_report': report,
                'alerts': self.alerts
            }, f, indent=2, default=str)

        self.file_logger.info(f"Report saved to {report_file}")

    def log_training_failed(self, error: str, error_details: Optional[Dict] = None) -> None:
        """
        Log training failure.

        Args:
            error: Error message
            error_details: Additional error details
        """
        end_time = datetime.utcnow()

        # Log to file
        self.file_logger.error(f"Training failed: {error}")
        if error_details:
            self.file_logger.error(f"Error details: {json.dumps(error_details, default=str)}")

        # Update database
        if self.training_history_id:
            history = self.db_session.query(ModelTrainingHistory).filter(
                ModelTrainingHistory.id == self.training_history_id
            ).first()

            if history:
                history.completed_at = end_time
                history.status = 'failed'
                history.error_logs = error
                history.training_logs = self.epoch_logs
                self.db_session.commit()

    def log_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = 'info'
    ) -> None:
        """
        Log an alert.

        Args:
            alert_type: Type of alert (e.g., 'early_stopping', 'nan_loss', 'low_accuracy')
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
        """
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.alerts.append(alert)

        # Log to file
        log_func = getattr(self.file_logger, severity, self.file_logger.info)
        log_func(f"ALERT [{alert_type}]: {message}")

    def log_checkpoint(self, checkpoint_path: str, metrics: Dict) -> None:
        """
        Log model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            metrics: Metrics at checkpoint
        """
        self.file_logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.file_logger.debug(f"Checkpoint metrics: {json.dumps(metrics, default=str)}")

    def _update_db_training_logs(self) -> None:
        """Update training logs in database."""
        if not self.training_history_id:
            return

        try:
            history = self.db_session.query(ModelTrainingHistory).filter(
                ModelTrainingHistory.id == self.training_history_id
            ).first()

            if history:
                history.training_logs = self.epoch_logs
                history.epochs_trained = len(self.epoch_logs)
                self.db_session.commit()
        except Exception as e:
            self.file_logger.warning(f"Failed to update database logs: {e}")

    def save_training_report(self, report: Dict) -> int:
        """
        Save a training report to database.

        Args:
            report: Training report dictionary

        Returns:
            Report ID
        """
        if self.training_history_id:
            history = self.db_session.query(ModelTrainingHistory).filter(
                ModelTrainingHistory.id == self.training_history_id
            ).first()

            if history:
                # Store full report in artifacts_path as JSON
                report_path = self.log_dir / f"{self.run_id}_full_report.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                history.artifacts_path = str(report_path)
                self.db_session.commit()

                return self.training_history_id

        return 0

    def get_training_history(
        self,
        profile_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get training history for a profile.

        Args:
            profile_id: Profile ID
            limit: Maximum number of records

        Returns:
            List of training history records
        """
        histories = self.db_session.query(ModelTrainingHistory).filter(
            ModelTrainingHistory.profile_id == profile_id
        ).order_by(
            ModelTrainingHistory.started_at.desc()
        ).limit(limit).all()

        return [
            {
                'id': h.id,
                'run_id': h.run_id,
                'started_at': h.started_at.isoformat() if h.started_at else None,
                'completed_at': h.completed_at.isoformat() if h.completed_at else None,
                'status': h.status,
                'duration': h.duration,
                'epochs_trained': h.epochs_trained,
                'val_accuracy': h.val_accuracy,
                'test_accuracy': h.test_accuracy,
                'backtest_sharpe': h.backtest_sharpe
            }
            for h in histories
        ]


def create_training_logger(
    profile_id: int,
    model_type: str,
    db_session: Session,
    log_dir: str = "logs/training"
) -> TrainingLogger:
    """
    Factory function to create a training logger.

    Args:
        profile_id: Profile ID
        model_type: Model type
        db_session: Database session
        log_dir: Log directory

    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(
        profile_id=profile_id,
        model_type=model_type,
        db_session=db_session,
        log_dir=log_dir
    )
