"""
HDF5 PyTorch Datasets for Memory-Efficient Large-Scale Training.

These datasets read data on-demand from HDF5 files, enabling training on
millions of samples with minimal memory usage (<500 MB).

Key Features:
- Lazy loading: Only reads requested sequences (not whole file)
- Chunked HDF5: Efficient sequential and random access
- Compatible with PyTorch DataLoader (num_workers, prefetch, etc.)
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HDF5SequenceDataset(Dataset):
    """
    PyTorch Dataset that reads sequences on-demand from HDF5 file.

    Memory efficient: Only loads the exact sequences needed for each batch,
    not the entire dataset. Enables training on 4.4M+ samples with <500 MB RAM.

    Usage:
        dataset = HDF5SequenceDataset('cache/preprocessed/BTCUSDT/features.h5')
        loader = DataLoader(dataset, batch_size=64, num_workers=4)

        for x, y in loader:
            # x: (batch_size, sequence_length, n_features)
            # y: (batch_size,) - labels
            model(x)

    Args:
        h5_path: Path to HDF5 file with 'features' and 'labels' datasets
        sequence_length: Number of time steps per sequence
        transform: Optional transform to apply to features
    """

    def __init__(
        self,
        h5_path: str,
        sequence_length: int = 60,
        transform: Optional[callable] = None
    ):
        self.h5_path = Path(h5_path)
        self.sequence_length = sequence_length
        self.transform = transform

        # Validate file exists
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # Open file to get metadata (will be reopened per-worker)
        with h5py.File(self.h5_path, 'r') as f:
            if 'features' not in f or 'labels' not in f:
                raise ValueError(f"HDF5 file must contain 'features' and 'labels' datasets")

            self.n_samples = len(f['features'])
            self.n_features = f['features'].shape[1]

        # Number of valid sequences
        self._length = max(0, self.n_samples - self.sequence_length)

        # File handle (opened lazily in worker processes)
        self._h5_file = None

        logger.info(
            f"HDF5SequenceDataset initialized: {self.n_samples:,} samples, "
            f"{self.n_features} features, {self._length:,} sequences"
        )

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its label.

        Args:
            idx: Sequence index

        Returns:
            Tuple of (features, label):
                features: Tensor of shape (sequence_length, n_features)
                label: Tensor scalar (class index)
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        # Lazy open file (important for multi-worker DataLoader)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        # Read sequence
        start_idx = idx
        end_idx = idx + self.sequence_length

        # Read features for sequence
        features = self._h5_file['features'][start_idx:end_idx]

        # Read label (at end of sequence)
        label = self._h5_file['labels'][end_idx - 1]

        # Convert to tensors
        x = torch.from_numpy(features.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)

        # Apply transform if provided
        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def get_batch(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get multiple sequences at once (more efficient than multiple __getitem__).

        Args:
            indices: List of sequence indices

        Returns:
            Tuple of batched (features, labels)
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        batch_x = []
        batch_y = []

        for idx in indices:
            start_idx = idx
            end_idx = idx + self.sequence_length

            features = self._h5_file['features'][start_idx:end_idx]
            label = self._h5_file['labels'][end_idx - 1]

            batch_x.append(features)
            batch_y.append(label)

        x = torch.from_numpy(np.stack(batch_x).astype(np.float32))
        y = torch.tensor(batch_y, dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.

        Returns:
            Tensor of shape (n_classes,) with inverse frequency weights
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        # Read all labels (still memory efficient - just integers)
        labels = self._h5_file['labels'][self.sequence_length - 1:]

        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        # Inverse frequency weighting
        weights = total / (len(unique) * counts)
        weights = weights / weights.sum()  # Normalize

        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of shape (n_sequences,) with per-sample weights
        """
        class_weights = self.get_class_weights()

        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        # Read all labels
        labels = self._h5_file['labels'][self.sequence_length - 1:]

        # Map labels to weights
        sample_weights = class_weights[labels]

        return sample_weights

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict with n_samples, n_features, class_distribution, etc.
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        labels = self._h5_file['labels'][:]

        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))

        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_sequences': self._length,
            'sequence_length': self.sequence_length,
            'class_distribution': class_dist,
            'file_path': str(self.h5_path),
        }

    def close(self):
        """Close HDF5 file handle."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class HDF5FlatDataset(Dataset):
    """
    PyTorch Dataset for tree-based models (no sequences needed).

    For XGBoost, LightGBM, Random Forest - these models don't need
    sequential data, just individual feature vectors.

    Usage:
        dataset = HDF5FlatDataset('cache/preprocessed/BTCUSDT/features.h5')
        X, y = dataset.get_numpy_arrays()  # For sklearn-style API
    """

    def __init__(self, h5_path: str, transform: Optional[callable] = None):
        self.h5_path = Path(h5_path)
        self.transform = transform

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = len(f['features'])
            self.n_features = f['features'].shape[1]

        self._h5_file = None

        logger.info(f"HDF5FlatDataset: {self.n_samples:,} samples, {self.n_features} features")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        features = self._h5_file['features'][idx]
        label = self._h5_file['labels'][idx]

        x = torch.from_numpy(features.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def get_numpy_arrays(
        self,
        start: int = 0,
        end: Optional[int] = None,
        chunk_size: int = 100_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data as numpy arrays (for sklearn-style models).

        Args:
            start: Start index
            end: End index (None = end of dataset)
            chunk_size: Chunk size for reading (to manage memory)

        Returns:
            Tuple of (X, y) numpy arrays
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        if end is None:
            end = self.n_samples

        # For smaller datasets, read directly
        if end - start <= chunk_size:
            X = self._h5_file['features'][start:end]
            y = self._h5_file['labels'][start:end]
            return X, y

        # For larger datasets, read in chunks
        X_chunks = []
        y_chunks = []

        for offset in range(start, end, chunk_size):
            chunk_end = min(offset + chunk_size, end)
            X_chunks.append(self._h5_file['features'][offset:chunk_end])
            y_chunks.append(self._h5_file['labels'][offset:chunk_end])

        return np.concatenate(X_chunks), np.concatenate(y_chunks)

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


class HDF5DataModule:
    """
    High-level data module that creates train/val/test splits from HDF5 cache.

    Handles:
    - Time-series aware splits (no future leakage)
    - Train/validation/test DataLoaders
    - Class weights for imbalanced data
    """

    def __init__(
        self,
        h5_path: str,
        sequence_length: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        """
        Initialize data module.

        Args:
            h5_path: Path to HDF5 cache file
            sequence_length: Sequence length for time series
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
        """
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Get total samples
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = len(f['features'])

        # Calculate split indices (time-series splits, no shuffling)
        n_sequences = self.n_samples - sequence_length
        self.train_end = int(n_sequences * train_ratio)
        self.val_end = int(n_sequences * (train_ratio + val_ratio))

        logger.info(
            f"Data splits: train={self.train_end:,}, "
            f"val={self.val_end - self.train_end:,}, "
            f"test={n_sequences - self.val_end:,}"
        )

    def train_dataset(self) -> 'HDF5SubsetDataset':
        """Get training dataset."""
        return HDF5SubsetDataset(
            self.h5_path,
            start_idx=0,
            end_idx=self.train_end,
            sequence_length=self.sequence_length
        )

    def val_dataset(self) -> 'HDF5SubsetDataset':
        """Get validation dataset."""
        return HDF5SubsetDataset(
            self.h5_path,
            start_idx=self.train_end,
            end_idx=self.val_end,
            sequence_length=self.sequence_length
        )

    def test_dataset(self) -> 'HDF5SubsetDataset':
        """Get test dataset."""
        return HDF5SubsetDataset(
            self.h5_path,
            start_idx=self.val_end,
            end_idx=self.n_samples - self.sequence_length,
            sequence_length=self.sequence_length
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training DataLoader."""
        return torch.utils.data.DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation DataLoader."""
        return torch.utils.data.DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get test DataLoader."""
        return torch.utils.data.DataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class HDF5SubsetDataset(Dataset):
    """
    Subset of HDF5 dataset for train/val/test splits.

    Reads from a contiguous range within the HDF5 file.
    """

    def __init__(
        self,
        h5_path: str,
        start_idx: int,
        end_idx: int,
        sequence_length: int = 60
    ):
        self.h5_path = h5_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.sequence_length = sequence_length
        self._length = end_idx - start_idx

        self._h5_file = None

        # Get feature count
        with h5py.File(h5_path, 'r') as f:
            self.n_features = f['features'].shape[1]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        # Map to global index
        global_idx = self.start_idx + idx

        # Read sequence
        features = self._h5_file['features'][global_idx:global_idx + self.sequence_length]
        label = self._h5_file['labels'][global_idx + self.sequence_length - 1]

        x = torch.from_numpy(features.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()
