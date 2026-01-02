"""
Advanced Cross-Validation Strategies for Financial Time Series.

Implements sophisticated cross-validation techniques that respect
the temporal nature of financial data and prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List, Dict, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    gap_days: int


class PurgedTimeSeriesCrossValidator:
    """
    Advanced time series cross-validator with purging and embargo.
    
    Features:
    - Purging: Remove training samples near test samples
    - Embargo: Add gap after test samples
    - Combinatorial purging for overlapping labels
    - Support for event-based sampling
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 purge_days: int = 2,
                 embargo_days: int = 1,
                 max_train_size: Optional[int] = None,
                 test_size: Optional[int] = None):
        """
        Initialize purged cross-validator.
        
        Args:
            n_splits: Number of CV folds
            purge_days: Days to purge before test set
            embargo_days: Days to embargo after test set
            max_train_size: Maximum training set size
            test_size: Fixed test set size
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.max_train_size = max_train_size
        self.test_size = test_size
        
    def split(self,
              X: np.ndarray,
              y: Optional[np.ndarray] = None,
              timestamps: Optional[pd.Series] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """
        Generate cross-validation folds with purging.
        
        Args:
            X: Feature matrix
            y: Target values
            timestamps: Timestamp for each sample
            groups: Group labels for grouped CV
            
        Yields:
            CVFold objects containing train/validation indices
        """
        n_samples = X.shape[0]
        
        if timestamps is None:
            timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='T')
            warnings.warn("No timestamps provided, using synthetic timestamps")
        
        # Determine test fold size
        if self.test_size is None:
            test_fold_size = n_samples // self.n_splits
        else:
            test_fold_size = self.test_size
        
        for fold_id in range(self.n_splits):
            # Calculate test fold boundaries
            test_start_idx = fold_id * test_fold_size
            test_end_idx = min((fold_id + 1) * test_fold_size, n_samples)
            
            if test_end_idx > n_samples:
                break
            
            # Get test period timestamps
            test_start_time = timestamps.iloc[test_start_idx]
            test_end_time = timestamps.iloc[test_end_idx - 1]
            
            # Apply purging and embargo
            purge_start_time = test_start_time - timedelta(days=self.purge_days)
            embargo_end_time = test_end_time + timedelta(days=self.embargo_days)
            
            # Create masks
            before_test_mask = timestamps < purge_start_time
            after_test_mask = timestamps > embargo_end_time
            test_mask = (timestamps >= test_start_time) & (timestamps <= test_end_time)
            
            # Combine masks for training data
            train_mask = before_test_mask | after_test_mask
            
            # Apply max training size if specified
            if self.max_train_size is not None:
                train_indices = np.where(train_mask)[0]
                if len(train_indices) > self.max_train_size:
                    # Keep most recent training data
                    before_indices = train_indices[train_indices < test_start_idx]
                    after_indices = train_indices[train_indices > test_end_idx]
                    
                    # Prioritize recent data before test
                    if len(before_indices) >= self.max_train_size:
                        train_indices = before_indices[-self.max_train_size:]
                    else:
                        remaining = self.max_train_size - len(before_indices)
                        train_indices = np.concatenate([
                            before_indices,
                            after_indices[:remaining]
                        ])
                    
                    train_mask = np.zeros(n_samples, dtype=bool)
                    train_mask[train_indices] = True
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(test_mask)[0]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield CVFold(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    train_start=timestamps[train_mask].min(),
                    train_end=timestamps[train_mask].max(),
                    val_start=test_start_time,
                    val_end=test_end_time,
                    gap_days=self.purge_days
                )
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


class WalkForwardCrossValidator:
    """
    Walk-forward cross-validation for time series.
    
    Features:
    - Expanding or sliding window training
    - Configurable step sizes
    - Support for anchored and unanchored walks
    - Multiple test periods
    """
    
    def __init__(self,
                 n_splits: int = 10,
                 train_window: Optional[int] = None,
                 test_window: int = 100,
                 step_size: Optional[int] = None,
                 expanding: bool = False,
                 anchored: bool = True):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of walk-forward steps
            train_window: Training window size (None for expanding)
            test_window: Test window size
            step_size: Step size (defaults to test_window)
            expanding: Use expanding window instead of sliding
            anchored: Anchor training start at beginning
        """
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size or test_window
        self.expanding = expanding
        self.anchored = anchored
        
    def split(self,
              X: np.ndarray,
              y: Optional[np.ndarray] = None,
              timestamps: Optional[pd.Series] = None,
              min_train_size: int = 100) -> Iterator[CVFold]:
        """
        Generate walk-forward validation folds.
        
        Args:
            X: Feature matrix
            y: Target values
            timestamps: Timestamps
            min_train_size: Minimum training size
            
        Yields:
            CVFold objects
        """
        n_samples = X.shape[0]
        
        if timestamps is None:
            timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='T')
        
        # Calculate starting position
        if self.expanding or self.train_window is None:
            initial_train_size = min_train_size
        else:
            initial_train_size = self.train_window
        
        current_pos = initial_train_size
        fold_id = 0
        
        while current_pos + self.test_window <= n_samples and fold_id < self.n_splits:
            # Define test window
            test_start = current_pos
            test_end = min(current_pos + self.test_window, n_samples)
            
            # Define training window
            if self.expanding:
                train_start = 0 if self.anchored else max(0, test_start - initial_train_size)
            else:
                if self.train_window is None:
                    train_start = 0
                else:
                    train_start = max(0, test_start - self.train_window)
            
            train_end = test_start
            
            # Create indices
            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(test_start, test_end)
            
            if len(train_indices) >= min_train_size and len(val_indices) > 0:
                yield CVFold(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    train_start=timestamps.iloc[train_start],
                    train_end=timestamps.iloc[train_end - 1],
                    val_start=timestamps.iloc[test_start],
                    val_end=timestamps.iloc[test_end - 1],
                    gap_days=0
                )
                fold_id += 1
            
            # Move forward
            current_pos += self.step_size
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


class BlockingTimeSeriesCrossValidator:
    """
    Blocked time series cross-validation.
    
    Groups consecutive observations into blocks to reduce variance
    and respect time series structure.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 block_size: Optional[int] = None,
                 block_duration: Optional[str] = None):
        """
        Initialize blocking cross-validator.
        
        Args:
            n_splits: Number of CV folds
            block_size: Fixed block size in samples
            block_duration: Block duration ('1D', '1W', etc.)
        """
        self.n_splits = n_splits
        self.block_size = block_size
        self.block_duration = block_duration
        
        if block_size is None and block_duration is None:
            raise ValueError("Either block_size or block_duration must be specified")
        
    def split(self,
              X: np.ndarray,
              y: Optional[np.ndarray] = None,
              timestamps: Optional[pd.Series] = None) -> Iterator[CVFold]:
        """
        Generate blocked cross-validation folds.
        
        Args:
            X: Feature matrix
            y: Target values
            timestamps: Timestamps for blocking
            
        Yields:
            CVFold objects
        """
        n_samples = X.shape[0]
        
        if self.block_duration and timestamps is None:
            raise ValueError("Timestamps required for duration-based blocking")
        
        # Create blocks
        if self.block_duration and timestamps is not None:
            blocks = self._create_time_blocks(timestamps)
        else:
            blocks = self._create_fixed_blocks(n_samples)
        
        n_blocks = len(blocks)
        if n_blocks < self.n_splits:
            raise ValueError(f"Not enough blocks ({n_blocks}) for {self.n_splits} splits")
        
        # Assign blocks to folds
        blocks_per_fold = n_blocks // self.n_splits
        
        for fold_id in range(self.n_splits):
            # Select test blocks
            test_block_start = fold_id * blocks_per_fold
            test_block_end = (fold_id + 1) * blocks_per_fold if fold_id < self.n_splits - 1 else n_blocks
            
            # Get indices for test blocks
            val_indices = []
            for block_idx in range(test_block_start, test_block_end):
                val_indices.extend(blocks[block_idx])
            
            # Get indices for training blocks
            train_indices = []
            for block_idx in range(n_blocks):
                if block_idx < test_block_start or block_idx >= test_block_end:
                    train_indices.extend(blocks[block_idx])
            
            val_indices = np.array(val_indices)
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                # Get timestamps if available
                if timestamps is not None:
                    train_start = timestamps.iloc[train_indices].min()
                    train_end = timestamps.iloc[train_indices].max()
                    val_start = timestamps.iloc[val_indices].min()
                    val_end = timestamps.iloc[val_indices].max()
                else:
                    train_start = train_end = val_start = val_end = datetime.now()
                
                yield CVFold(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    gap_days=0
                )
    
    def _create_fixed_blocks(self, n_samples: int) -> List[List[int]]:
        """Create fixed-size blocks."""
        if self.block_size is None:
            # Auto-determine block size
            self.block_size = max(1, n_samples // (self.n_splits * 5))
        
        blocks = []
        for i in range(0, n_samples, self.block_size):
            block = list(range(i, min(i + self.block_size, n_samples)))
            if block:
                blocks.append(block)
        
        return blocks
    
    def _create_time_blocks(self, timestamps: pd.Series) -> List[List[int]]:
        """Create time-based blocks."""
        # Group by time period
        df = pd.DataFrame({'idx': range(len(timestamps)), 'timestamp': timestamps})
        df['block'] = df['timestamp'].dt.to_period(self.block_duration)
        
        blocks = []
        for _, group in df.groupby('block'):
            blocks.append(group['idx'].tolist())
        
        return blocks
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


class CombinatorialPurgedCrossValidator:
    """
    Combinatorial purged cross-validation for overlapping labels.
    
    Handles cases where prediction labels overlap in time,
    requiring special purging logic.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 n_test_groups: int = 2,
                 purge_pct: float = 0.01):
        """
        Initialize combinatorial purged CV.
        
        Args:
            n_splits: Number of paths (combinations)
            n_test_groups: Number of groups in test set
            purge_pct: Percentage of observations to purge
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_pct = purge_pct
        
    def split(self,
              X: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None,
              pred_times: Optional[pd.Series] = None,
              eval_times: Optional[pd.Series] = None) -> Iterator[CVFold]:
        """
        Generate combinatorial purged folds.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels
            pred_times: Prediction times
            eval_times: Evaluation times (when labels are revealed)
            
        Yields:
            CVFold objects
        """
        if groups is None:
            # Create synthetic groups
            n_samples = X.shape[0]
            n_groups = max(self.n_splits, n_samples // 100)
            groups = np.repeat(range(n_groups), n_samples // n_groups + 1)[:n_samples]
        
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_test_groups:
            raise ValueError(f"Not enough groups ({n_groups}) for {self.n_test_groups} test groups")
        
        # Generate combinations
        from itertools import combinations
        test_combinations = list(combinations(unique_groups, self.n_test_groups))
        
        # Sample n_splits combinations
        if len(test_combinations) > self.n_splits:
            selected_combinations = np.random.choice(
                len(test_combinations), 
                self.n_splits, 
                replace=False
            )
            test_combinations = [test_combinations[i] for i in selected_combinations]
        
        for fold_id, test_groups in enumerate(test_combinations[:self.n_splits]):
            # Create test mask
            test_mask = np.isin(groups, test_groups)
            val_indices = np.where(test_mask)[0]
            
            # Apply purging
            if pred_times is not None and eval_times is not None:
                train_indices = self._purge_overlapping_labels(
                    test_mask, pred_times, eval_times
                )
            else:
                # Simple purging based on proximity
                train_mask = ~test_mask
                if self.purge_pct > 0:
                    n_purge = int(len(val_indices) * self.purge_pct)
                    for test_idx in val_indices:
                        # Purge nearby samples
                        purge_start = max(0, test_idx - n_purge)
                        purge_end = min(len(train_mask), test_idx + n_purge)
                        train_mask[purge_start:purge_end] = False
                
                train_indices = np.where(train_mask)[0]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield CVFold(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    train_start=datetime.now(),  # Placeholder
                    train_end=datetime.now(),
                    val_start=datetime.now(),
                    val_end=datetime.now(),
                    gap_days=0
                )
    
    def _purge_overlapping_labels(self,
                                  test_mask: np.ndarray,
                                  pred_times: pd.Series,
                                  eval_times: pd.Series) -> np.ndarray:
        """Purge training samples with overlapping labels."""
        train_mask = ~test_mask
        test_indices = np.where(test_mask)[0]
        
        for test_idx in test_indices:
            test_eval_time = eval_times.iloc[test_idx]
            
            # Find training samples whose evaluation overlaps
            overlap_mask = (
                (pred_times < test_eval_time) & 
                (eval_times > pred_times.iloc[test_idx])
            )
            
            train_mask &= ~overlap_mask
        
        return np.where(train_mask)[0]
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits