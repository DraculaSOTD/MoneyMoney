"""
Comprehensive Data Preprocessing Example.

This example demonstrates how to use all data preprocessing capabilities
in the crypto ML trading system, from loading raw data to creating
ML-ready features with GPU acceleration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.enhanced_data_loader import UniversalDataLoader
from data.preprocessing import DataPreprocessor, PreprocessingConfig
from data.data_validator import DataValidator
from feature_engineering.feature_pipeline import FeaturePipeline
from feature_engineering.gpu_indicators import GPUTechnicalIndicators
from features.enhanced_features import create_enhanced_features
from data_feeds.realtime_pipeline import RealTimeDataPipeline
from data_feeds.alternative_data import AlternativeDataAggregator
from utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(level='INFO')
logger = get_logger(__name__)


class ComprehensiveDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline demonstrating all capabilities.
    """
    
    def __init__(self, enable_gpu: bool = True):
        """Initialize preprocessing pipeline with all components."""
        self.enable_gpu = enable_gpu
        
        # Initialize components
        self.data_loader = UniversalDataLoader()
        self.preprocessor = DataPreprocessor()
        self.validator = DataValidator()
        self.feature_pipeline = FeaturePipeline()
        
        if enable_gpu:
            self.gpu_indicators = GPUTechnicalIndicators(use_gpu=True)
            logger.info("GPU acceleration enabled for preprocessing")
        else:
            self.gpu_indicators = None
            logger.info("Using CPU for preprocessing")
        
        # Configure preprocessing
        self.preprocessing_config = PreprocessingConfig(
            handle_missing='smart',
            handle_infinite='clip',
            remove_duplicates=True,
            stationarity_method='pct_change',
            outlier_method='iqr',
            outlier_threshold=3.0,
            scaling_method='feature_specific',
            add_time_features=True
        )
    
    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from file and perform initial validation.
        
        Args:
            filepath: Path to data file (CSV format)
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        
        # Load data with auto-detection
        data = self.data_loader.load_data(filepath)
        logger.info(f"Loaded {len(data)} rows with columns: {list(data.columns)}")
        
        # Initial validation
        validation_report = self.validator.validate_data(data)
        logger.info(f"Data quality score: {validation_report.quality_score:.2f}")
        
        if validation_report.has_critical_issues():
            logger.warning("Critical data issues found:")
            for issue in validation_report.issues:
                if issue['severity'] == 'critical':
                    logger.warning(f"  - {issue['issue']}: {issue['details']}")
        
        return data
    
    def preprocess_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing to historical data.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Preprocessed data ready for ML
        """
        logger.info("Starting comprehensive preprocessing")
        
        # Step 1: Basic preprocessing
        logger.info("Step 1: Basic preprocessing (missing values, outliers)")
        preprocessed_data, preprocessing_report = self.preprocessor.preprocess(
            data, self.preprocessing_config
        )
        
        logger.info(f"  - Handled {preprocessing_report['missing_values_handled']} missing values")
        logger.info(f"  - Removed {preprocessing_report['outliers_handled']} outliers")
        logger.info(f"  - Applied stationarity correction to {len(preprocessing_report['stationarity_columns'])} columns")
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering")
        
        # Technical indicators
        if self.enable_gpu and self.gpu_indicators:
            logger.info("  - Calculating GPU-accelerated technical indicators")
            ohlcv_data = {
                'open': preprocessed_data['open'].values,
                'high': preprocessed_data['high'].values,
                'low': preprocessed_data['low'].values,
                'close': preprocessed_data['close'].values,
                'volume': preprocessed_data['volume'].values
            }
            
            indicators = self.gpu_indicators.calculate_all_indicators(ohlcv_data)
            
            # Add indicators to dataframe
            for name, values in indicators.items():
                if len(values) == len(preprocessed_data):
                    preprocessed_data[name] = values
                else:
                    # Handle indicators with different lengths
                    pad_length = len(preprocessed_data) - len(values)
                    padded_values = np.concatenate([np.full(pad_length, np.nan), values])
                    preprocessed_data[name] = padded_values
        else:
            logger.info("  - Calculating CPU-based technical indicators")
            preprocessed_data = self.feature_pipeline.create_features(preprocessed_data)
        
        # Enhanced features
        logger.info("  - Creating enhanced ML features")
        enhanced_features = create_enhanced_features(preprocessed_data)
        preprocessed_data = pd.concat([preprocessed_data, enhanced_features], axis=1)
        
        # Step 3: Feature-specific scaling
        logger.info("Step 3: Feature-specific scaling")
        
        # Group features by type
        price_features = [col for col in preprocessed_data.columns if any(
            x in col.lower() for x in ['price', 'open', 'high', 'low', 'close']
        )]
        volume_features = [col for col in preprocessed_data.columns if 'volume' in col.lower()]
        bounded_features = [col for col in preprocessed_data.columns if any(
            x in col.lower() for x in ['rsi', 'stoch', 'percent', 'ratio']
        )]
        
        # Apply different scaling to different feature groups
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        
        scalers = {}
        
        # Standard scaling for price features
        if price_features:
            scaler = StandardScaler()
            preprocessed_data[price_features] = scaler.fit_transform(
                preprocessed_data[price_features].fillna(0)
            )
            scalers['price'] = scaler
        
        # Robust scaling for volume features
        if volume_features:
            scaler = RobustScaler()
            preprocessed_data[volume_features] = scaler.fit_transform(
                preprocessed_data[volume_features].fillna(0)
            )
            scalers['volume'] = scaler
        
        # MinMax scaling for bounded features
        if bounded_features:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            preprocessed_data[bounded_features] = scaler.fit_transform(
                preprocessed_data[bounded_features].fillna(0)
            )
            scalers['bounded'] = scaler
        
        # Step 4: Handle remaining NaN values
        logger.info("Step 4: Final NaN handling")
        nan_counts = preprocessed_data.isna().sum()
        if nan_counts.sum() > 0:
            logger.info(f"  - Found {nan_counts.sum()} remaining NaN values")
            preprocessed_data = preprocessed_data.fillna(method='ffill').fillna(0)
        
        # Step 5: Final validation
        logger.info("Step 5: Final validation")
        final_validation = self.validator.check_ml_readiness(preprocessed_data.values)
        logger.info(f"  - ML readiness: {'✓ Ready' if final_validation['ready'] else '✗ Not ready'}")
        
        if not final_validation['ready']:
            for issue in final_validation['issues']:
                logger.warning(f"  - {issue}")
        
        # Add metadata
        preprocessed_data.attrs['preprocessing_config'] = self.preprocessing_config.__dict__
        preprocessed_data.attrs['scalers'] = scalers
        preprocessed_data.attrs['feature_count'] = len(preprocessed_data.columns)
        
        logger.info(f"Preprocessing complete. Final shape: {preprocessed_data.shape}")
        
        return preprocessed_data
    
    def create_ml_datasets(self, preprocessed_data: pd.DataFrame,
                          target_column: str = 'close',
                          sequence_length: int = 60,
                          prediction_horizon: int = 1,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1) -> Dict:
        """
        Create ML-ready datasets with sequences.
        
        Args:
            preprocessed_data: Preprocessed DataFrame
            target_column: Column to predict
            sequence_length: Length of input sequences
            prediction_horizon: How far ahead to predict
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            
        Returns:
            Dictionary with train/val/test datasets
        """
        logger.info("Creating ML datasets")
        
        # Create target (future returns)
        if target_column == 'close':
            # Predict future returns
            preprocessed_data['target'] = preprocessed_data['close'].pct_change(
                prediction_horizon
            ).shift(-prediction_horizon)
        else:
            preprocessed_data['target'] = preprocessed_data[target_column].shift(-prediction_horizon)
        
        # Remove rows with NaN targets
        preprocessed_data = preprocessed_data.dropna(subset=['target'])
        
        # Feature columns (exclude target and metadata)
        feature_columns = [col for col in preprocessed_data.columns 
                         if col not in ['target', 'timestamp', 'symbol']]
        
        # Create sequences
        logger.info(f"  - Creating sequences of length {sequence_length}")
        
        X, y = [], []
        
        for i in range(sequence_length, len(preprocessed_data) - prediction_horizon):
            X.append(preprocessed_data[feature_columns].iloc[i-sequence_length:i].values)
            y.append(preprocessed_data['target'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"  - Created {len(X)} sequences")
        logger.info(f"  - Input shape: {X.shape}")
        logger.info(f"  - Target shape: {y.shape}")
        
        # Split data
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Time-based splitting
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Create classification targets (buy/hold/sell)
        threshold = 0.002  # 0.2% threshold
        
        y_train_class = np.where(y_train > threshold, 2,  # Buy
                                np.where(y_train < -threshold, 0,  # Sell
                                        1))  # Hold
        
        y_val_class = np.where(y_val > threshold, 2,
                              np.where(y_val < -threshold, 0, 1))
        
        y_test_class = np.where(y_test > threshold, 2,
                               np.where(y_test < -threshold, 0, 1))
        
        # Log class distribution
        logger.info("  - Class distribution:")
        unique, counts = np.unique(y_train_class, return_counts=True)
        class_dist = dict(zip(['Sell', 'Hold', 'Buy'], counts))
        logger.info(f"    Train: {class_dist}")
        
        datasets = {
            'X_train': X_train,
            'y_train_reg': y_train,
            'y_train_class': y_train_class,
            'X_val': X_val,
            'y_val_reg': y_val,
            'y_val_class': y_val_class,
            'X_test': X_test,
            'y_test_reg': y_test,
            'y_test_class': y_test_class,
            'feature_names': feature_columns,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon
        }
        
        return datasets
    
    async def process_realtime_data(self, symbols: List[str], exchanges: List[str],
                                  duration_minutes: int = 5):
        """
        Demonstrate real-time data processing pipeline.
        
        Args:
            symbols: List of symbols to track
            exchanges: List of exchanges
            duration_minutes: How long to run the demo
        """
        logger.info("Starting real-time data processing demo")
        
        # Initialize real-time pipeline
        pipeline = RealTimeDataPipeline(
            symbols=symbols,
            exchanges=exchanges,
            enable_gpu=self.enable_gpu
        )
        
        # Initialize alternative data aggregator
        alt_data = AlternativeDataAggregator()
        
        # Data collection
        collected_data = []
        
        def data_callback(processed_data):
            """Callback for processed data."""
            logger.info(f"Received data for {processed_data.symbol}: "
                       f"Price: ${processed_data.market_data.last:.2f}, "
                       f"Features: {len(processed_data.features)}")
            collected_data.append({
                'timestamp': processed_data.timestamp,
                'symbol': processed_data.symbol,
                'price': processed_data.market_data.last,
                'features': processed_data.features,
                'sentiment': processed_data.sentiment_score
            })
        
        # Register callback
        pipeline.register_data_callback(data_callback)
        
        # Start pipeline
        pipeline.start()
        
        # Run for specified duration
        import time
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                # Fetch alternative data periodically
                if len(collected_data) % 10 == 0:
                    logger.info("Fetching alternative data...")
                    alt_data_result = await alt_data.fetch_all_data(symbols, lookback_hours=1)
                    logger.info(f"  - Fetched {len(alt_data_result.get('news', []))} news items")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Real-time demo interrupted")
        finally:
            pipeline.stop()
        
        logger.info(f"Collected {len(collected_data)} data points")
        
        return collected_data
    
    def save_preprocessed_data(self, data: pd.DataFrame, output_path: str):
        """Save preprocessed data with metadata."""
        # Save as parquet for efficiency
        data.to_parquet(output_path, index=True)
        
        # Save metadata
        import json
        metadata = {
            'preprocessing_config': data.attrs.get('preprocessing_config', {}),
            'feature_count': data.attrs.get('feature_count', len(data.columns)),
            'shape': list(data.shape),
            'columns': list(data.columns)
        }
        
        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved preprocessed data to {output_path}")
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Run comprehensive data preprocessing demonstration."""
    # Initialize preprocessor
    preprocessor = ComprehensiveDataPreprocessor(enable_gpu=True)
    
    # Example 1: Process historical data
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE 1: Historical Data Preprocessing")
    logger.info("="*50)
    
    # Load sample data (you would use your actual data file)
    # For demo, we'll create synthetic data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1h')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.lognormal(10, 1, len(dates))
    })
    
    # Add some missing values and outliers for demonstration
    sample_data.loc[100:110, 'volume'] = np.nan
    sample_data.loc[500, 'close'] = sample_data['close'].mean() * 10  # Outlier
    
    # Save to temp file
    temp_file = 'temp_sample_data.csv'
    sample_data.to_csv(temp_file, index=False)
    
    # Load and validate
    data = preprocessor.load_and_validate_data(temp_file)
    
    # Preprocess
    preprocessed_data = preprocessor.preprocess_historical_data(data)
    
    # Create ML datasets
    datasets = preprocessor.create_ml_datasets(
        preprocessed_data,
        target_column='close',
        sequence_length=60,
        prediction_horizon=1
    )
    
    logger.info("\nDataset summary:")
    logger.info(f"  - Training samples: {len(datasets['X_train'])}")
    logger.info(f"  - Validation samples: {len(datasets['X_val'])}")
    logger.info(f"  - Test samples: {len(datasets['X_test'])}")
    logger.info(f"  - Features per sample: {datasets['X_train'].shape[2]}")
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(preprocessed_data, 'preprocessed_data.parquet')
    
    # Clean up temp file
    import os
    os.remove(temp_file)
    
    # Example 2: Real-time data processing (commented out as it requires live connections)
    """
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE 2: Real-time Data Processing")
    logger.info("="*50)
    
    # Process real-time data
    symbols = ['BTC/USDT', 'ETH/USDT']
    exchanges = ['binance']
    
    # Run real-time processing
    import asyncio
    real_time_data = asyncio.run(
        preprocessor.process_realtime_data(symbols, exchanges, duration_minutes=1)
    )
    """
    
    logger.info("\n" + "="*50)
    logger.info("Comprehensive data preprocessing complete!")
    logger.info("="*50)


if __name__ == "__main__":
    main()