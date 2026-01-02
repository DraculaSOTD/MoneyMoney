import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from statsmodels.tsa.stattools import adfuller


class DataValidator:
    """
    Comprehensive data validation for ML readiness.
    Includes checks for NaN, infinity, stationarity, and data quality.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_report = {}
        
    def validate(self, df: pd.DataFrame, 
                target_col: Optional[str] = None,
                feature_cols: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            target_col: Target column for ML
            feature_cols: Feature columns to validate
            
        Returns:
            Validation report dictionary
        """
        self.validation_report = {
            'data_shape': df.shape,
            'timestamp': self._validate_timestamp(df),
            'missing_values': self._check_missing_values(df),
            'infinite_values': self._check_infinite_values(df),
            'data_types': self._check_data_types(df),
            'duplicates': self._check_duplicates(df),
            'stationarity': self._check_stationarity(df, feature_cols),
            'outliers': self._check_outliers(df, feature_cols),
            'data_quality': self._calculate_data_quality_score(df)
        }
        
        if target_col and target_col in df.columns:
            self.validation_report['target_distribution'] = self._check_target_distribution(df[target_col])
        
        self.validation_report['is_ml_ready'] = self._is_ml_ready()
        
        return self.validation_report
    
    def _validate_timestamp(self, df: pd.DataFrame) -> Dict:
        """Validate timestamp column if present."""
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            ts = df[timestamp_col]
            return {
                'has_timestamp': True,
                'column': timestamp_col,
                'is_datetime': pd.api.types.is_datetime64_any_dtype(ts),
                'is_sorted': ts.is_monotonic_increasing,
                'has_gaps': self._check_time_gaps(ts) if pd.api.types.is_datetime64_any_dtype(ts) else None,
                'date_range': (ts.min(), ts.max()) if pd.api.types.is_datetime64_any_dtype(ts) else None
            }
        
        return {'has_timestamp': False}
    
    def _check_time_gaps(self, ts: pd.Series) -> Dict:
        """Check for gaps in time series."""
        if not pd.api.types.is_datetime64_any_dtype(ts):
            return {}
        
        # Calculate time differences
        diffs = ts.diff().dropna()
        
        # Find mode (most common interval)
        mode_interval = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs.median()
        
        # Find gaps (intervals larger than mode)
        gaps = diffs[diffs > mode_interval * 1.5]
        
        return {
            'expected_interval': mode_interval,
            'num_gaps': len(gaps),
            'gap_locations': gaps.index.tolist() if len(gaps) > 0 else [],
            'max_gap': gaps.max() if len(gaps) > 0 else pd.Timedelta(0)
        }
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        
        return {
            'has_missing': df.isnull().any().any(),
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_pct_by_column': missing_pct[missing_pct > 0].to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
    
    def _check_infinite_values(self, df: pd.DataFrame) -> Dict:
        """Check for infinite values in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        return {
            'has_infinite': len(inf_counts) > 0,
            'total_infinite': sum(inf_counts.values()),
            'infinite_by_column': inf_counts,
            'columns_with_infinite': list(inf_counts.keys())
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """Check data types and identify non-numeric columns."""
        dtypes = df.dtypes.to_dict()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'all_types': {k: str(v) for k, v in dtypes.items()},
            'numeric_columns': numeric_cols,
            'object_columns': object_cols,
            'datetime_columns': datetime_cols,
            'num_numeric': len(numeric_cols),
            'num_object': len(object_cols),
            'num_datetime': len(datetime_cols)
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows."""
        duplicates = df.duplicated()
        
        # Check for duplicate timestamps if present
        timestamp_duplicates = {}
        for col in ['timestamp', 'date', 'datetime']:
            if col in df.columns:
                dup_count = df[col].duplicated().sum()
                if dup_count > 0:
                    timestamp_duplicates[col] = dup_count
        
        return {
            'has_duplicates': duplicates.any(),
            'num_duplicates': duplicates.sum(),
            'duplicate_indices': df[duplicates].index.tolist(),
            'timestamp_duplicates': timestamp_duplicates
        }
    
    def _check_stationarity(self, df: pd.DataFrame, 
                          feature_cols: Optional[List[str]] = None) -> Dict:
        """Check stationarity of time series features using ADF test."""
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Focus on price and return columns
            feature_cols = [col for col in numeric_cols if any(
                keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low', 'return', 'pct']
            )]
        
        stationarity_results = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            # Skip if too many NaN values
            if df[col].isna().sum() > len(df) * 0.1:
                continue
            
            try:
                # Perform ADF test
                series = df[col].dropna()
                if len(series) < 20:  # Need minimum observations
                    continue
                
                result = adfuller(series, autolag='AIC')
                
                stationarity_results[col] = {
                    'adf_statistic': result[0],
                    'p_value': result[1],
                    'is_stationary': result[1] < 0.05,
                    'critical_values': result[4]
                }
            except Exception as e:
                stationarity_results[col] = {'error': str(e)}
        
        # Summary
        stationary_cols = [col for col, res in stationarity_results.items() 
                          if res.get('is_stationary', False)]
        non_stationary_cols = [col for col, res in stationarity_results.items() 
                              if not res.get('is_stationary', True) and 'error' not in res]
        
        return {
            'results': stationarity_results,
            'stationary_columns': stationary_cols,
            'non_stationary_columns': non_stationary_cols,
            'num_stationary': len(stationary_cols),
            'num_non_stationary': len(non_stationary_cols)
        }
    
    def _check_outliers(self, df: pd.DataFrame, 
                       feature_cols: Optional[List[str]] = None) -> Dict:
        """Check for outliers using IQR and z-score methods."""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_results = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = (z_scores > 3).sum()
            
            outlier_results[col] = {
                'iqr_outliers': int(iqr_outliers),
                'iqr_outlier_pct': (iqr_outliers / len(series)) * 100,
                'z_score_outliers': int(z_outliers),
                'z_score_outlier_pct': (z_outliers / len(series)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        # Summary
        total_iqr_outliers = sum(res['iqr_outliers'] for res in outlier_results.values())
        total_z_outliers = sum(res['z_score_outliers'] for res in outlier_results.values())
        
        return {
            'results': outlier_results,
            'total_iqr_outliers': total_iqr_outliers,
            'total_z_score_outliers': total_z_outliers,
            'columns_with_outliers': [col for col, res in outlier_results.items() 
                                     if res['iqr_outliers'] > 0]
        }
    
    def _check_target_distribution(self, target: pd.Series) -> Dict:
        """Check target variable distribution for classification."""
        if target.dtype in ['object', 'category'] or target.nunique() < 10:
            # Classification target
            value_counts = target.value_counts()
            value_pcts = (value_counts / len(target)) * 100
            
            # Check for imbalance
            max_pct = value_pcts.max()
            min_pct = value_pcts.min()
            imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
            
            return {
                'type': 'classification',
                'num_classes': target.nunique(),
                'value_counts': value_counts.to_dict(),
                'value_percentages': value_pcts.to_dict(),
                'is_balanced': imbalance_ratio < 3,
                'imbalance_ratio': imbalance_ratio
            }
        else:
            # Regression target
            return {
                'type': 'regression',
                'mean': target.mean(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max(),
                'skewness': stats.skew(target),
                'kurtosis': stats.kurtosis(target)
            }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Deduct for missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        score -= min(missing_pct * 2, 30)  # Max 30 point deduction
        
        # Deduct for infinite values
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            inf_pct = (np.isinf(numeric_df).sum().sum() / 
                      (numeric_df.shape[0] * numeric_df.shape[1])) * 100
            score -= min(inf_pct * 5, 20)  # Max 20 point deduction
        
        # Deduct for duplicates
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        score -= min(dup_pct * 2, 20)  # Max 20 point deduction
        
        return max(score, 0.0)
    
    def _is_ml_ready(self) -> bool:
        """Determine if data is ready for ML based on validation results."""
        # Check critical issues
        if self.validation_report['missing_values']['total_missing'] > len(self.validation_report['data_shape']) * 0.1:
            return False
        
        if self.validation_report['infinite_values']['has_infinite']:
            return False
        
        if self.validation_report['data_quality'] < 70:
            return False
        
        return True
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on validation results."""
        recommendations = []
        
        # Missing values
        if self.validation_report['missing_values']['has_missing']:
            recommendations.append(
                f"Handle missing values in columns: {self.validation_report['missing_values']['columns_with_missing']}"
            )
        
        # Infinite values
        if self.validation_report['infinite_values']['has_infinite']:
            recommendations.append(
                f"Remove or cap infinite values in: {self.validation_report['infinite_values']['columns_with_infinite']}"
            )
        
        # Stationarity
        non_stationary = self.validation_report['stationarity']['non_stationary_columns']
        if non_stationary:
            recommendations.append(
                f"Apply differencing or percentage changes to non-stationary columns: {non_stationary}"
            )
        
        # Outliers
        outlier_cols = self.validation_report['outliers']['columns_with_outliers']
        if outlier_cols:
            recommendations.append(
                f"Consider handling outliers in: {outlier_cols[:5]}{'...' if len(outlier_cols) > 5 else ''}"
            )
        
        # Duplicates
        if self.validation_report['duplicates']['has_duplicates']:
            recommendations.append(
                f"Remove {self.validation_report['duplicates']['num_duplicates']} duplicate rows"
            )
        
        return recommendations
    
    def print_report(self):
        """Print a formatted validation report."""
        print("="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        print(f"\nData Shape: {self.validation_report['data_shape']}")
        print(f"Data Quality Score: {self.validation_report['data_quality']:.1f}/100")
        print(f"ML Ready: {'✓' if self.validation_report['is_ml_ready'] else '✗'}")
        
        print("\n--- Missing Values ---")
        missing = self.validation_report['missing_values']
        print(f"Has Missing: {missing['has_missing']}")
        if missing['has_missing']:
            print(f"Total Missing: {missing['total_missing']}")
            print(f"Columns with Missing: {len(missing['columns_with_missing'])}")
        
        print("\n--- Data Types ---")
        types = self.validation_report['data_types']
        print(f"Numeric Columns: {types['num_numeric']}")
        print(f"Object Columns: {types['num_object']}")
        print(f"Datetime Columns: {types['num_datetime']}")
        
        print("\n--- Stationarity ---")
        stat = self.validation_report['stationarity']
        print(f"Stationary Columns: {stat['num_stationary']}")
        print(f"Non-stationary Columns: {stat['num_non_stationary']}")
        
        print("\n--- Recommendations ---")
        for i, rec in enumerate(self.get_recommendations(), 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)