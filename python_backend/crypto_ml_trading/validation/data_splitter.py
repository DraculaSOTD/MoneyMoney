"""
Time Series Aware Data Splitting for Financial Data.

Implements various splitting strategies that prevent data leakage in time series:
- Sequential train/test splits
- Purged splits with gap periods
- Embargo periods to prevent leakage
- Walk-forward splitting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Iterator, Optional, Union
from datetime import datetime, timedelta
from sklearn.model_selection import KFold


class TimeSeriesDataSplitter:
    """
    Time series aware data splitter with leakage prevention.
    
    Features:
    - Sequential splitting (no future data in training)
    - Purging and embargo to prevent leakage
    - Support for multiple validation sets
    - Gap periods between train/test
    """
    
    def __init__(self,
                 test_size: float = 0.2,
                 validation_size: float = 0.1,
                 gap_size: int = 0,  # Number of periods to skip between train/test
                 purge_days: int = 2):  # Days to purge around test set
        """
        Initialize time series splitter.
        
        Args:
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            gap_size: Number of periods to skip between train/test
            purge_days: Number of days to purge around splits
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.gap_size = gap_size
        self.purge_days = purge_days
        
    def split(self, X: np.ndarray, y: np.ndarray,
              timestamps: Optional[pd.Series] = None) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/validation/test sets.
        
        Args:
            X: Feature matrix
            y: Target values
            timestamps: Optional timestamps for purging
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        
        # Calculate split indices
        test_size = int(n_samples * self.test_size)
        val_size = int(n_samples * self.validation_size)
        train_size = n_samples - test_size - val_size - (2 * self.gap_size)
        
        if train_size <= 0:
            raise ValueError("Not enough data for splitting with given parameters")
        
        # Create indices
        train_end = train_size
        val_start = train_end + self.gap_size
        val_end = val_start + val_size
        test_start = val_end + self.gap_size
        
        # Apply purging if timestamps provided
        if timestamps is not None:
            train_mask, val_mask, test_mask = self._apply_purging(
                timestamps, train_end, val_start, val_end, test_start
            )
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
        else:
            # Simple sequential split
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            X_test = X[test_start:]
            y_test = y[test_start:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray,
                        timestamps: Optional[pd.Series] = None) -> Tuple[np.ndarray, ...]:
        """
        Simple train/test split for time series.
        
        Args:
            X: Feature matrix
            y: Target values
            timestamps: Optional timestamps for purging
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        train_size = n_samples - test_size - self.gap_size
        
        if train_size <= 0:
            raise ValueError("Not enough data for splitting")
        
        train_end = train_size
        test_start = train_end + self.gap_size
        
        if timestamps is not None:
            # Apply purging
            train_mask = np.arange(n_samples) < train_end
            test_mask = np.arange(n_samples) >= test_start
            
            # Purge around split point
            if self.purge_days > 0:
                split_time = timestamps.iloc[train_end]
                purge_start = split_time - timedelta(days=self.purge_days)
                purge_end = split_time + timedelta(days=self.purge_days)
                
                train_mask &= timestamps < purge_start
                test_mask &= timestamps > purge_end
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
        else:
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:]
            y_test = y[test_start:]
        
        return X_train, X_test, y_train, y_test
    
    def _apply_purging(self, timestamps: pd.Series,
                      train_end: int, val_start: int,
                      val_end: int, test_start: int) -> Tuple[np.ndarray, ...]:
        """Apply purging around split points."""
        n_samples = len(timestamps)
        
        # Create initial masks
        train_mask = np.arange(n_samples) < train_end
        val_mask = (np.arange(n_samples) >= val_start) & (np.arange(n_samples) < val_end)
        test_mask = np.arange(n_samples) >= test_start
        
        if self.purge_days > 0:
            # Purge around validation set
            val_start_time = timestamps.iloc[val_start] if val_start < n_samples else timestamps.iloc[-1]
            val_end_time = timestamps.iloc[val_end-1] if val_end <= n_samples else timestamps.iloc[-1]
            
            train_mask &= timestamps < (val_start_time - timedelta(days=self.purge_days))
            test_mask &= timestamps > (val_end_time + timedelta(days=self.purge_days))
        
        return train_mask, val_mask, test_mask


class PurgedKFold:
    """
    K-Fold cross-validation with purging for time series.
    
    Prevents data leakage by:
    - Purging training samples that are too close to test samples
    - Maintaining temporal order
    - Adding embargo periods
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = 2,
                 embargo_days: int = 1):
        """
        Initialize purged K-fold.
        
        Args:
            n_splits: Number of folds
            purge_days: Days to purge before test set
            embargo_days: Days to embargo after test set
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              timestamps: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for purged K-fold splits.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            timestamps: Timestamps for each sample
            
        Yields:
            Train and test indices for each fold
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if timestamps is None:
            # Simple K-fold without purging
            kf = KFold(n_splits=self.n_splits, shuffle=False)
            for train_idx, test_idx in kf.split(X):
                yield train_idx, test_idx
        else:
            # Purged K-fold
            fold_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                # Define test fold
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
                
                test_indices = indices[test_start:test_end]
                
                # Get test period timestamps
                test_start_time = timestamps.iloc[test_start]
                test_end_time = timestamps.iloc[test_end - 1]
                
                # Define purge period
                purge_start = test_start_time - timedelta(days=self.purge_days)
                embargo_end = test_end_time + timedelta(days=self.embargo_days)
                
                # Create train indices with purging
                train_mask = (
                    ((timestamps < purge_start) | (timestamps > embargo_end)) &
                    ((indices < test_start) | (indices >= test_end))
                )
                
                train_indices = indices[train_mask]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices


class WalkForwardSplitter:
    """
    Walk-forward analysis splitter for time series.
    
    Creates overlapping train/test windows that move forward in time.
    """
    
    def __init__(self,
                 train_window: int,
                 test_window: int,
                 step_size: int,
                 min_train_size: Optional[int] = None):
        """
        Initialize walk-forward splitter.
        
        Args:
            train_window: Size of training window
            test_window: Size of test window
            step_size: Step size for moving windows
            min_train_size: Minimum training size required
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size or train_window // 2
        
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            
        Yields:
            Train and test indices for each window
        """
        n_samples = X.shape[0]
        
        # Start position ensures we have enough training data
        start_pos = max(self.min_train_size, self.train_window)
        
        while start_pos + self.test_window <= n_samples:
            # Define training window
            train_start = max(0, start_pos - self.train_window)
            train_end = start_pos
            
            # Define test window
            test_start = start_pos
            test_end = min(start_pos + self.test_window, n_samples)
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            # Move forward
            start_pos += self.step_size


class BlockingTimeSeriesSplitter:
    """
    Blocked time series splitter for reducing variance in CV.
    
    Groups consecutive samples into blocks before splitting.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 block_size: int = 100):
        """
        Initialize blocking splitter.
        
        Args:
            n_splits: Number of folds
            block_size: Size of each block
        """
        self.n_splits = n_splits
        self.block_size = block_size
        
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate blocked splits.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            
        Yields:
            Train and test indices for each fold
        """
        n_samples = X.shape[0]
        n_blocks = n_samples // self.block_size
        
        if n_blocks < self.n_splits:
            raise ValueError(f"Not enough blocks ({n_blocks}) for {self.n_splits} splits")
        
        # Assign blocks to folds
        blocks_per_fold = n_blocks // self.n_splits
        
        for i in range(self.n_splits):
            # Define test blocks
            test_block_start = i * blocks_per_fold
            test_block_end = (i + 1) * blocks_per_fold if i < self.n_splits - 1 else n_blocks
            
            # Convert blocks to indices
            test_start = test_block_start * self.block_size
            test_end = min(test_block_end * self.block_size, n_samples)
            
            test_indices = np.arange(test_start, test_end)
            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n_samples)
            ])
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices