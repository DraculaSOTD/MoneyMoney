import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict
from enum import Enum


class LabelingMethod(Enum):
    """Different methods for creating trading labels."""
    PRICE_DIRECTION = "price_direction"  # Simple next price up/down
    FIXED_THRESHOLD = "fixed_threshold"  # Fixed % threshold
    TRIPLE_BARRIER = "triple_barrier"  # Stop loss, take profit, time limit
    DYNAMIC_THRESHOLD = "dynamic_threshold"  # Volatility-adjusted threshold
    RETURN_BINS = "return_bins"  # Quantile-based bins


class DecisionLabeler:
    """
    Create decision labels for machine learning models.
    Supports various labeling strategies for supervised learning.
    """
    
    def __init__(self, method: Union[str, LabelingMethod] = LabelingMethod.PRICE_DIRECTION):
        """
        Initialize the decision labeler.
        
        Args:
            method: Labeling method to use
        """
        self.method = LabelingMethod(method) if isinstance(method, str) else method
        
    def create_labels(self, df: pd.DataFrame, 
                     lookforward: int = 1,
                     threshold: float = 0.001,
                     stop_loss: float = 0.02,
                     take_profit: float = 0.02,
                     time_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Create trading labels based on future price movements.
        
        Args:
            df: DataFrame with OHLCV data
            lookforward: Number of periods to look ahead
            threshold: Threshold for buy/sell decisions (%)
            stop_loss: Stop loss threshold for triple barrier
            take_profit: Take profit threshold for triple barrier
            time_limit: Time limit for triple barrier
            
        Returns:
            DataFrame with added label columns
        """
        result = df.copy()
        
        if self.method == LabelingMethod.PRICE_DIRECTION:
            result = self._label_price_direction(result, lookforward)
        elif self.method == LabelingMethod.FIXED_THRESHOLD:
            result = self._label_fixed_threshold(result, lookforward, threshold)
        elif self.method == LabelingMethod.TRIPLE_BARRIER:
            result = self._label_triple_barrier(result, stop_loss, take_profit, time_limit)
        elif self.method == LabelingMethod.DYNAMIC_THRESHOLD:
            result = self._label_dynamic_threshold(result, lookforward)
        elif self.method == LabelingMethod.RETURN_BINS:
            result = self._label_return_bins(result, lookforward)
            
        return result
    
    def _label_price_direction(self, df: pd.DataFrame, lookforward: int) -> pd.DataFrame:
        """
        Simple labeling based on whether price goes up or down.
        Similar to the ML project's approach.
        """
        # Calculate future price
        df['future_price'] = df['close'].shift(-lookforward)
        
        # Create labels
        df['decision'] = 'hold'  # Default
        
        # Buy if future price is higher
        mask_buy = df['future_price'] > df['close']
        df.loc[mask_buy, 'decision'] = 'buy'
        
        # Sell if future price is lower
        mask_sell = df['future_price'] < df['close']
        df.loc[mask_sell, 'decision'] = 'sell'
        
        # Create numeric labels for ML
        decision_map = {'buy': 0, 'sell': 1, 'hold': 2}
        df['label'] = df['decision'].map(decision_map)
        
        # Calculate future return for analysis
        df['future_return'] = (df['future_price'] - df['close']) / df['close']
        
        # Clean up
        df = df.drop(columns=['future_price'])
        
        return df
    
    def _label_fixed_threshold(self, df: pd.DataFrame, lookforward: int, 
                              threshold: float) -> pd.DataFrame:
        """
        Label based on fixed return threshold.
        """
        # Calculate future return
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
        
        # Create labels based on threshold
        df['decision'] = 'hold'
        df.loc[df['future_return'] > threshold, 'decision'] = 'buy'
        df.loc[df['future_return'] < -threshold, 'decision'] = 'sell'
        
        # Numeric labels
        decision_map = {'buy': 0, 'sell': 1, 'hold': 2}
        df['label'] = df['decision'].map(decision_map)
        
        return df
    
    def _label_triple_barrier(self, df: pd.DataFrame, stop_loss: float, 
                             take_profit: float, time_limit: Optional[int]) -> pd.DataFrame:
        """
        Triple barrier method: stop loss, take profit, or time limit.
        """
        df['decision'] = 'hold'
        df['label'] = 2
        df['exit_return'] = 0.0
        df['exit_bars'] = 0
        
        if time_limit is None:
            time_limit = len(df)
        
        for i in range(len(df) - 1):
            entry_price = df.iloc[i]['close']
            
            for j in range(1, min(time_limit + 1, len(df) - i)):
                current_price = df.iloc[i + j]['close']
                ret = (current_price - entry_price) / entry_price
                
                # Check barriers
                if ret >= take_profit:
                    df.loc[df.index[i], 'decision'] = 'buy'
                    df.loc[df.index[i], 'label'] = 0
                    df.loc[df.index[i], 'exit_return'] = ret
                    df.loc[df.index[i], 'exit_bars'] = j
                    break
                elif ret <= -stop_loss:
                    df.loc[df.index[i], 'decision'] = 'sell'
                    df.loc[df.index[i], 'label'] = 1
                    df.loc[df.index[i], 'exit_return'] = ret
                    df.loc[df.index[i], 'exit_bars'] = j
                    break
                elif j == time_limit:
                    # Time limit reached
                    if ret > 0:
                        df.loc[df.index[i], 'decision'] = 'buy'
                        df.loc[df.index[i], 'label'] = 0
                    elif ret < 0:
                        df.loc[df.index[i], 'decision'] = 'sell'
                        df.loc[df.index[i], 'label'] = 1
                    df.loc[df.index[i], 'exit_return'] = ret
                    df.loc[df.index[i], 'exit_bars'] = j
                    break
        
        return df
    
    def _label_dynamic_threshold(self, df: pd.DataFrame, lookforward: int) -> pd.DataFrame:
        """
        Dynamic threshold based on volatility (ATR).
        """
        # Calculate future return
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
        
        # Use ATR for dynamic threshold if available
        if 'ATR_14' in df.columns:
            # Dynamic threshold as percentage of price
            df['threshold'] = df['ATR_14'] / df['close'] * 0.5  # Half ATR as threshold
        else:
            # Fallback to rolling std
            df['threshold'] = df['close'].pct_change().rolling(20).std() * 0.5
        
        # Create labels
        df['decision'] = 'hold'
        df.loc[df['future_return'] > df['threshold'], 'decision'] = 'buy'
        df.loc[df['future_return'] < -df['threshold'], 'decision'] = 'sell'
        
        # Numeric labels
        decision_map = {'buy': 0, 'sell': 1, 'hold': 2}
        df['label'] = df['decision'].map(decision_map)
        
        return df
    
    def _label_return_bins(self, df: pd.DataFrame, lookforward: int, 
                          n_bins: int = 3) -> pd.DataFrame:
        """
        Label based on return quantiles.
        """
        # Calculate future return
        df['future_return'] = df['close'].shift(-lookforward) / df['close'] - 1
        
        # Create bins based on quantiles
        # Filter out NaN values for quantile calculation
        valid_returns = df['future_return'].dropna()
        
        if n_bins == 3:
            # Tertiles for buy/hold/sell
            q1 = valid_returns.quantile(0.33)
            q2 = valid_returns.quantile(0.67)
            
            df['decision'] = 'hold'
            df.loc[df['future_return'] <= q1, 'decision'] = 'sell'
            df.loc[df['future_return'] >= q2, 'decision'] = 'buy'
            
            decision_map = {'buy': 0, 'sell': 1, 'hold': 2}
        else:
            # Multiple bins for more granular classification
            df['decision'] = pd.qcut(df['future_return'], n_bins, 
                                    labels=range(n_bins), duplicates='drop')
            df['label'] = df['decision'].astype(int)
            return df
        
        df['label'] = df['decision'].map(decision_map)
        
        return df
    
    def create_ml_labels(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Create labels based on configuration dict.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration dict with labeling parameters
            
        Returns:
            DataFrame with labels
        """
        method = config.get('method', 'price_direction')
        self.method = LabelingMethod(method)
        
        return self.create_labels(
            df,
            lookforward=config.get('lookforward', 1),
            threshold=config.get('threshold', 0.001),
            stop_loss=config.get('stop_loss', 0.02),
            take_profit=config.get('take_profit', 0.02),
            time_limit=config.get('time_limit', None)
        )
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of labels for analysis."""
        if 'label' not in df.columns:
            return {}
        
        # Count labels
        label_counts = df['label'].value_counts().to_dict()
        total = len(df['label'].dropna())
        
        # Calculate percentages
        label_pcts = {k: v/total*100 for k, v in label_counts.items()}
        
        # Map to decision names if possible
        if 'decision' in df.columns:
            decision_counts = df['decision'].value_counts().to_dict()
            decision_pcts = {k: v/total*100 for k, v in decision_counts.items()}
        else:
            decision_counts = {}
            decision_pcts = {}
        
        return {
            'label_counts': label_counts,
            'label_percentages': label_pcts,
            'decision_counts': decision_counts,
            'decision_percentages': decision_pcts,
            'total_samples': total,
            'null_samples': df['label'].isna().sum()
        }
    
    def validate_labels(self, df: pd.DataFrame) -> Dict:
        """Validate label quality and balance."""
        if 'label' not in df.columns:
            return {'error': 'No labels found'}
        
        # Get distribution
        dist = self.get_label_distribution(df)
        
        # Check balance
        if len(dist['label_counts']) > 0:
            max_pct = max(dist['label_percentages'].values())
            min_pct = min(dist['label_percentages'].values())
            imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        else:
            imbalance_ratio = float('inf')
        
        # Check null labels
        null_pct = dist['null_samples'] / len(df) * 100
        
        # Recommendations
        recommendations = []
        if imbalance_ratio > 3:
            recommendations.append("High class imbalance detected. Consider rebalancing.")
        if null_pct > 5:
            recommendations.append(f"High null label percentage ({null_pct:.1f}%). Check data completeness.")
        
        return {
            'distribution': dist,
            'imbalance_ratio': imbalance_ratio,
            'null_percentage': null_pct,
            'is_balanced': imbalance_ratio < 3,
            'recommendations': recommendations
        }