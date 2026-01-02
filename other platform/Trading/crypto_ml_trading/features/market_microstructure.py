import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings


class MarketMicrostructureFeatures:
    """
    Advanced market microstructure features for cryptocurrency trading.
    
    These features capture:
    - Order flow dynamics
    - Liquidity patterns  
    - Price impact
    - Market depth
    - Trading intensity
    - Bid-ask spread dynamics
    
    Specifically designed for high-frequency crypto data.
    """
    
    @staticmethod
    def calculate_order_flow_imbalance(bid_volume: pd.Series, 
                                     ask_volume: pd.Series,
                                     window: int = 10) -> pd.Series:
        """
        Calculate order flow imbalance (OFI).
        
        OFI measures the imbalance between buying and selling pressure.
        
        Args:
            bid_volume: Bid volume series
            ask_volume: Ask volume series
            window: Rolling window size
            
        Returns:
            Order flow imbalance series
        """
        # Raw imbalance
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        
        # Smooth with rolling average
        ofi = imbalance.rolling(window=window, min_periods=1).mean()
        
        return ofi
    
    @staticmethod
    def calculate_volume_weighted_spread(high: pd.Series, low: pd.Series,
                                       close: pd.Series, volume: pd.Series,
                                       window: int = 20) -> pd.Series:
        """
        Calculate volume-weighted bid-ask spread proxy.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            window: Rolling window
            
        Returns:
            Volume-weighted spread
        """
        # Spread proxy using high-low
        spread = (high - low) / close
        
        # Volume-weight the spread
        weighted_spread = spread * volume
        sum_weighted = weighted_spread.rolling(window=window).sum()
        sum_volume = volume.rolling(window=window).sum()
        
        vw_spread = sum_weighted / (sum_volume + 1e-10)
        
        return vw_spread
    
    @staticmethod
    def calculate_kyle_lambda(price_changes: pd.Series,
                            volume: pd.Series,
                            window: int = 60) -> pd.Series:
        """
        Calculate Kyle's lambda (price impact coefficient).
        
        Lambda measures how much price moves per unit of volume.
        
        Args:
            price_changes: Price change series
            volume: Volume series
            window: Rolling window
            
        Returns:
            Kyle's lambda series
        """
        # Calculate rolling regression coefficient
        lambdas = []
        
        for i in range(window, len(price_changes)):
            window_changes = price_changes[i-window:i]
            window_volume = volume[i-window:i]
            
            # Remove zero volumes
            mask = window_volume > 0
            if mask.sum() < 10:
                lambdas.append(np.nan)
                continue
                
            # Simple regression: price_change = lambda * volume
            try:
                cov = np.cov(window_changes[mask], window_volume[mask])[0, 1]
                var = np.var(window_volume[mask])
                lambda_est = cov / var if var > 0 else 0
                lambdas.append(abs(lambda_est))  # Use absolute value
            except:
                lambdas.append(np.nan)
                
        # Create series with proper index
        lambda_series = pd.Series(
            [np.nan] * window + lambdas,
            index=price_changes.index
        )
        
        return lambda_series
    
    @staticmethod
    def calculate_amihud_illiquidity(returns: pd.Series,
                                   volume: pd.Series,
                                   window: int = 30) -> pd.Series:
        """
        Calculate Amihud illiquidity measure.
        
        Measures price impact per unit of trading volume.
        
        Args:
            returns: Return series
            volume: Dollar volume series
            window: Rolling window
            
        Returns:
            Amihud illiquidity ratio
        """
        # Absolute return to volume ratio
        illiquidity = abs(returns) / (volume + 1e-10)
        
        # Rolling average
        amihud = illiquidity.rolling(window=window, min_periods=1).mean()
        
        return amihud
    
    @staticmethod
    def calculate_roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Roll's implied spread from price changes.
        
        Args:
            close: Close price series
            window: Rolling window
            
        Returns:
            Roll's spread estimate
        """
        # Price changes
        price_changes = close.diff()
        
        # Rolling covariance
        spreads = []
        
        for i in range(window, len(price_changes)):
            window_changes = price_changes[i-window:i].dropna()
            
            if len(window_changes) < 2:
                spreads.append(np.nan)
                continue
                
            # Autocovariance
            cov = np.cov(window_changes[:-1], window_changes[1:])[0, 1]
            
            # Roll's spread = 2 * sqrt(-cov) if cov < 0
            if cov < 0:
                spread = 2 * np.sqrt(-cov)
            else:
                spread = 0  # No spread detected
                
            spreads.append(spread)
            
        # Create series
        roll_spread = pd.Series(
            [np.nan] * window + spreads,
            index=close.index
        )
        
        return roll_spread
    
    @staticmethod
    def calculate_vpin(volume: pd.Series, price_changes: pd.Series,
                      bucket_size: Optional[float] = None,
                      n_buckets: int = 50) -> pd.Series:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).
        
        Args:
            volume: Volume series
            price_changes: Price change series
            bucket_size: Size of volume buckets
            n_buckets: Number of buckets for calculation
            
        Returns:
            VPIN series
        """
        if bucket_size is None:
            # Use average daily volume / 50
            bucket_size = volume.rolling(1440).sum().mean() / 50
            
        # Classify volume as buy or sell
        buy_volume = volume.copy()
        sell_volume = volume.copy()
        
        buy_volume[price_changes <= 0] = 0
        sell_volume[price_changes > 0] = 0
        
        # Create volume buckets
        cumulative_volume = volume.cumsum()
        bucket_indices = (cumulative_volume // bucket_size).astype(int)
        
        # Calculate VPIN for each bucket
        vpin_values = []
        
        for i in range(n_buckets, bucket_indices.max()):
            # Get volumes for last n_buckets
            mask = (bucket_indices >= i - n_buckets) & (bucket_indices < i)
            
            bucket_buy = buy_volume[mask].sum()
            bucket_sell = sell_volume[mask].sum()
            bucket_total = bucket_buy + bucket_sell
            
            if bucket_total > 0:
                vpin = abs(bucket_buy - bucket_sell) / bucket_total
            else:
                vpin = np.nan
                
            vpin_values.append(vpin)
            
        # Map back to original index
        vpin_series = pd.Series(index=volume.index, dtype=float)
        
        for i, vpin in enumerate(vpin_values):
            bucket_idx = i + n_buckets
            mask = bucket_indices == bucket_idx
            vpin_series[mask] = vpin
            
        return vpin_series.ffill()
    
    @staticmethod
    def calculate_trade_intensity(volume: pd.Series, 
                                num_trades: Optional[pd.Series] = None,
                                window: int = 60) -> pd.DataFrame:
        """
        Calculate various trade intensity metrics.
        
        Args:
            volume: Volume series
            num_trades: Number of trades (if available)
            window: Rolling window
            
        Returns:
            DataFrame with intensity metrics
        """
        intensity_metrics = pd.DataFrame(index=volume.index)
        
        # Volume intensity (current vs average)
        avg_volume = volume.rolling(window=window * 24, min_periods=window).mean()
        intensity_metrics['volume_intensity'] = volume / (avg_volume + 1e-10)
        
        # Volume acceleration
        volume_ma_short = volume.rolling(window=window).mean()
        volume_ma_long = volume.rolling(window=window * 4).mean()
        intensity_metrics['volume_acceleration'] = volume_ma_short / (volume_ma_long + 1e-10)
        
        # Trade size if we have number of trades
        if num_trades is not None:
            intensity_metrics['avg_trade_size'] = volume / (num_trades + 1e-10)
            intensity_metrics['trade_frequency'] = num_trades.rolling(window=window).mean()
            
        # Volume volatility
        intensity_metrics['volume_volatility'] = volume.rolling(window=window).std() / (
            volume.rolling(window=window).mean() + 1e-10
        )
        
        return intensity_metrics
    
    @staticmethod
    def calculate_order_book_features(bid_prices: List[pd.Series],
                                    bid_volumes: List[pd.Series],
                                    ask_prices: List[pd.Series],
                                    ask_volumes: List[pd.Series]) -> pd.DataFrame:
        """
        Calculate order book based features.
        
        Args:
            bid_prices: List of bid price levels
            bid_volumes: List of bid volumes
            ask_prices: List of ask price levels
            ask_volumes: List of ask volumes
            
        Returns:
            DataFrame with order book features
        """
        # Ensure all series have same index
        index = bid_prices[0].index
        features = pd.DataFrame(index=index)
        
        # Best bid-ask spread
        features['spread'] = ask_prices[0] - bid_prices[0]
        features['spread_pct'] = features['spread'] / ((ask_prices[0] + bid_prices[0]) / 2)
        
        # Mid price
        features['mid_price'] = (ask_prices[0] + bid_prices[0]) / 2
        
        # Book imbalance at different levels
        for i in range(min(5, len(bid_prices))):
            features[f'book_imbalance_l{i+1}'] = (
                (bid_volumes[i] - ask_volumes[i]) / 
                (bid_volumes[i] + ask_volumes[i] + 1e-10)
            )
            
        # Depth
        features['bid_depth'] = sum(bid_volumes[:5])
        features['ask_depth'] = sum(ask_volumes[:5])
        features['total_depth'] = features['bid_depth'] + features['ask_depth']
        
        # Weighted average prices
        bid_value = sum(p * v for p, v in zip(bid_prices[:5], bid_volumes[:5]))
        bid_volume = sum(bid_volumes[:5])
        features['weighted_bid'] = bid_value / (bid_volume + 1e-10)
        
        ask_value = sum(p * v for p, v in zip(ask_prices[:5], ask_volumes[:5]))
        ask_volume = sum(ask_volumes[:5])
        features['weighted_ask'] = ask_value / (ask_volume + 1e-10)
        
        # Slope of order book
        if len(bid_prices) >= 5:
            bid_price_array = np.array([p.iloc[-1] for p in bid_prices[:5]])
            bid_volume_array = np.array([v.iloc[-1] for v in bid_volumes[:5]])
            
            # Fit linear regression
            try:
                bid_slope = np.polyfit(bid_price_array, bid_volume_array, 1)[0]
                features['bid_slope'] = bid_slope
            except:
                features['bid_slope'] = 0
                
        return features
    
    @staticmethod
    def calculate_all_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all market microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all microstructure features added
        """
        result = df.copy()
        
        # Basic microstructure features
        if 'volume' in df.columns:
            # Order flow (using volume as proxy)
            high_vol = df['volume'].copy()
            low_vol = df['volume'].copy()
            high_vol[df['close'] <= df['open']] = 0
            low_vol[df['close'] > df['open']] = 0
            
            result['order_flow_imbalance'] = MarketMicrostructureFeatures.calculate_order_flow_imbalance(
                high_vol, low_vol
            )
            
            # Volume-weighted spread
            result['vw_spread'] = MarketMicrostructureFeatures.calculate_volume_weighted_spread(
                df['high'], df['low'], df['close'], df['volume']
            )
            
            # Kyle's lambda
            result['kyle_lambda'] = MarketMicrostructureFeatures.calculate_kyle_lambda(
                df['close'].pct_change(), df['volume']
            )
            
            # Amihud illiquidity
            result['amihud_illiquidity'] = MarketMicrostructureFeatures.calculate_amihud_illiquidity(
                df['returns'], df['volume'] * df['close']
            )
            
            # VPIN
            result['vpin'] = MarketMicrostructureFeatures.calculate_vpin(
                df['volume'], df['close'].diff()
            )
            
            # Trade intensity
            intensity_features = MarketMicrostructureFeatures.calculate_trade_intensity(
                df['volume']
            )
            for col in intensity_features.columns:
                result[f'micro_{col}'] = intensity_features[col]
                
        # Roll's spread
        result['roll_spread'] = MarketMicrostructureFeatures.calculate_roll_spread(df['close'])
        
        # Additional derived features
        result['high_low_ratio'] = df['high'] / df['low']
        result['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        result['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return result