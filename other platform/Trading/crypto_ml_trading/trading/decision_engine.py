import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingDecision:
    """Trading decision with confidence and reasoning."""
    signal: Signal
    confidence: float  # 0-1
    triggered_indicators: List[str]
    buy_score: float
    sell_score: float
    reasoning: str
    timestamp: pd.Timestamp
    price: float


class TradingDecisionEngine:
    """
    Advanced trading decision engine with confidence scoring.
    Combines multiple indicators and patterns for robust signal generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize decision engine with configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)['trading_decisions']
        else:
            self.config = self._default_config()
        
        self.indicator_weights = self._initialize_weights()
        
    def _default_config(self) -> Dict:
        """Default trading decision configuration."""
        return {
            'rule_based': {
                'enabled': True,
                'min_indicators_agreement': 3,
                'confidence_weighting': True
            },
            'signals': {
                'buy_conditions': [
                    {'indicator': 'RSI_14', 'operator': '<', 'value': 30},
                    {'indicator': 'MACD', 'operator': '>', 'compare_to': 'Signal_Line'},
                    {'indicator': 'close', 'operator': '<', 'compare_to': 'Lower_Band'},
                    {'indicator': 'Parabolic_SAR', 'operator': '<', 'compare_to': 'low'},
                    {'indicator': '%K', 'operator': '>', 'compare_to': '%D'},
                    {'indicator': 'CMF_20', 'operator': '>', 'value': 0},
                    {'indicator': 'ADX_14', 'operator': '>', 'value': 25},
                    {'indicator': 'Tenkan_sen', 'operator': '>', 'compare_to': 'Kijun_sen'}
                ],
                'sell_conditions': [
                    {'indicator': 'RSI_14', 'operator': '>', 'value': 70},
                    {'indicator': 'MACD', 'operator': '<', 'compare_to': 'Signal_Line'},
                    {'indicator': 'close', 'operator': '>', 'compare_to': 'Upper_Band'},
                    {'indicator': 'Parabolic_SAR', 'operator': '>', 'compare_to': 'high'},
                    {'indicator': '%K', 'operator': '<', 'compare_to': '%D'},
                    {'indicator': 'CMF_20', 'operator': '<', 'value': 0},
                    {'indicator': 'ADX_14', 'operator': '>', 'value': 25},
                    {'indicator': 'Tenkan_sen', 'operator': '<', 'compare_to': 'Kijun_sen'}
                ]
            },
            'confidence_scoring': {
                'enabled': True,
                'weights': {
                    'trend_indicators': 0.3,
                    'momentum_indicators': 0.3,
                    'volume_indicators': 0.2,
                    'pattern_indicators': 0.2
                }
            }
        }
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize indicator weights for confidence scoring."""
        # Categorize indicators
        trend_indicators = ['MACD', 'Signal_Line', 'ADX_14', 'Parabolic_SAR', 
                          'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B']
        momentum_indicators = ['RSI_14', '%K', '%D', 'Momentum_14']
        volume_indicators = ['CMF_20', 'volume_ratio']
        pattern_indicators = ['Bullish_MACD_Divergence', 'Bearish_MACD_Divergence',
                            'Bullish_RSI_Divergence', 'Bearish_RSI_Divergence',
                            'Elliott_Wave_Sequence']
        
        weights = {}
        weight_config = self.config['confidence_scoring']['weights']
        
        # Assign weights based on category
        for ind in trend_indicators:
            weights[ind] = weight_config['trend_indicators'] / len(trend_indicators)
        for ind in momentum_indicators:
            weights[ind] = weight_config['momentum_indicators'] / len(momentum_indicators)
        for ind in volume_indicators:
            weights[ind] = weight_config['volume_indicators'] / len(volume_indicators)
        for ind in pattern_indicators:
            weights[ind] = weight_config['pattern_indicators'] / len(pattern_indicators)
        
        return weights
    
    def evaluate_condition(self, row: pd.Series, condition: Dict) -> Tuple[bool, str]:
        """
        Evaluate a single trading condition.
        
        Returns:
            Tuple of (condition_met, description)
        """
        indicator = condition['indicator']
        operator = condition['operator']
        
        # Check if indicator exists
        if indicator not in row.index:
            return False, f"{indicator} not available"
        
        # Get indicator value
        ind_value = row[indicator]
        if pd.isna(ind_value):
            return False, f"{indicator} is NaN"
        
        # Get comparison value
        if 'value' in condition:
            compare_value = condition['value']
            compare_desc = str(compare_value)
        elif 'compare_to' in condition:
            compare_to = condition['compare_to']
            if compare_to not in row.index:
                return False, f"{compare_to} not available"
            compare_value = row[compare_to]
            if pd.isna(compare_value):
                return False, f"{compare_to} is NaN"
            compare_desc = compare_to
        else:
            return False, "No comparison value specified"
        
        # Evaluate condition
        if operator == '>':
            result = ind_value > compare_value
        elif operator == '<':
            result = ind_value < compare_value
        elif operator == '>=':
            result = ind_value >= compare_value
        elif operator == '<=':
            result = ind_value <= compare_value
        elif operator == '==':
            result = ind_value == compare_value
        else:
            return False, f"Unknown operator: {operator}"
        
        description = f"{indicator} {operator} {compare_desc}"
        return result, description
    
    def analyze_market_context(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """Analyze broader market context for additional insights."""
        context = {
            'trend': 'neutral',
            'volatility': 'normal',
            'volume_profile': 'average',
            'support_resistance': None
        }
        
        if current_idx < 50:
            return context
        
        # Trend analysis
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            sma50 = df['SMA_50'].iloc[current_idx]
            sma200 = df['SMA_200'].iloc[current_idx]
            if not pd.isna(sma50) and not pd.isna(sma200):
                if sma50 > sma200:
                    context['trend'] = 'bullish'
                else:
                    context['trend'] = 'bearish'
        
        # Volatility analysis
        if 'ATR_14' in df.columns:
            current_atr = df['ATR_14'].iloc[current_idx]
            avg_atr = df['ATR_14'].iloc[current_idx-20:current_idx].mean()
            if not pd.isna(current_atr) and not pd.isna(avg_atr):
                if current_atr > avg_atr * 1.5:
                    context['volatility'] = 'high'
                elif current_atr < avg_atr * 0.5:
                    context['volatility'] = 'low'
        
        # Volume profile
        if 'volume' in df.columns:
            current_vol = df['volume'].iloc[current_idx]
            avg_vol = df['volume'].iloc[current_idx-20:current_idx].mean()
            if current_vol > avg_vol * 2:
                context['volume_profile'] = 'high'
            elif current_vol < avg_vol * 0.5:
                context['volume_profile'] = 'low'
        
        # Support/Resistance proximity
        if 'Support_Level' in df.columns and 'Resistance_Level' in df.columns:
            current_price = df['close'].iloc[current_idx]
            support = df['Support_Level'].iloc[current_idx]
            resistance = df['Resistance_Level'].iloc[current_idx]
            
            if not pd.isna(support) and support > 0:
                support_dist = (current_price - support) / current_price
                if abs(support_dist) < 0.01:  # Within 1%
                    context['support_resistance'] = 'near_support'
            
            if not pd.isna(resistance) and resistance > 0:
                resistance_dist = (resistance - current_price) / current_price
                if abs(resistance_dist) < 0.01:  # Within 1%
                    context['support_resistance'] = 'near_resistance'
        
        return context
    
    def calculate_confidence(self, triggered_buy: List[str], triggered_sell: List[str],
                           market_context: Dict) -> float:
        """
        Calculate confidence score based on triggered indicators and market context.
        
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.5
        
        # Number of indicators agreement
        buy_count = len(triggered_buy)
        sell_count = len(triggered_sell)
        total_triggered = buy_count + sell_count
        
        if total_triggered == 0:
            return 0.0
        
        # Direction clarity (how much buy vs sell agree)
        direction_clarity = abs(buy_count - sell_count) / total_triggered
        base_confidence += direction_clarity * 0.2
        
        # Indicator importance (weighted)
        if self.config['confidence_scoring']['enabled']:
            weighted_score = 0
            for indicator in triggered_buy + triggered_sell:
                # Extract base indicator name (remove comparison part)
                base_ind = indicator.split()[0]
                if base_ind in self.indicator_weights:
                    weighted_score += self.indicator_weights[base_ind]
            base_confidence += weighted_score * 0.2
        
        # Market context adjustments
        if market_context['trend'] == 'bullish' and buy_count > sell_count:
            base_confidence += 0.1
        elif market_context['trend'] == 'bearish' and sell_count > buy_count:
            base_confidence += 0.1
        
        if market_context['volatility'] == 'high':
            base_confidence -= 0.05  # Less confident in high volatility
        
        if market_context['volume_profile'] == 'high':
            base_confidence += 0.05  # More confident with high volume
        
        # Support/Resistance bonus
        if market_context['support_resistance'] == 'near_support' and buy_count > sell_count:
            base_confidence += 0.1
        elif market_context['support_resistance'] == 'near_resistance' and sell_count > buy_count:
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def generate_decision(self, df: pd.DataFrame, index: int = -1) -> TradingDecision:
        """
        Generate trading decision for a specific point in the dataframe.
        
        Args:
            df: DataFrame with OHLCV and indicators
            index: Index to evaluate (default: last row)
            
        Returns:
            TradingDecision object
        """
        if index == -1:
            index = len(df) - 1
        
        row = df.iloc[index]
        
        # Evaluate buy conditions
        triggered_buy = []
        buy_score = 0
        for condition in self.config['signals']['buy_conditions']:
            met, description = self.evaluate_condition(row, condition)
            if met:
                triggered_buy.append(description)
                buy_score += 1
        
        # Evaluate sell conditions
        triggered_sell = []
        sell_score = 0
        for condition in self.config['signals']['sell_conditions']:
            met, description = self.evaluate_condition(row, condition)
            if met:
                triggered_sell.append(description)
                sell_score += 1
        
        # Analyze market context
        market_context = self.analyze_market_context(df, index)
        
        # Calculate confidence
        confidence = self.calculate_confidence(triggered_buy, triggered_sell, market_context)
        
        # Determine signal
        min_agreement = self.config['rule_based']['min_indicators_agreement']
        
        if buy_score > sell_score and buy_score >= min_agreement:
            if buy_score >= 5 and confidence > 0.7:
                signal = Signal.STRONG_BUY
            else:
                signal = Signal.BUY
        elif sell_score > buy_score and sell_score >= min_agreement:
            if sell_score >= 5 and confidence > 0.7:
                signal = Signal.STRONG_SELL
            else:
                signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            signal, triggered_buy, triggered_sell, 
            market_context, confidence
        )
        
        # Create decision
        decision = TradingDecision(
            signal=signal,
            confidence=confidence,
            triggered_indicators=triggered_buy + triggered_sell,
            buy_score=buy_score / len(self.config['signals']['buy_conditions']),
            sell_score=sell_score / len(self.config['signals']['sell_conditions']),
            reasoning=reasoning,
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            price=row['close']
        )
        
        return decision
    
    def _generate_reasoning(self, signal: Signal, triggered_buy: List[str],
                          triggered_sell: List[str], market_context: Dict,
                          confidence: float) -> str:
        """Generate human-readable reasoning for the decision."""
        reasoning_parts = []
        
        # Signal strength
        reasoning_parts.append(f"Signal: {signal.value} (Confidence: {confidence:.2%})")
        
        # Indicator summary
        if triggered_buy:
            reasoning_parts.append(f"Buy indicators ({len(triggered_buy)}): {', '.join(triggered_buy[:3])}")
        if triggered_sell:
            reasoning_parts.append(f"Sell indicators ({len(triggered_sell)}): {', '.join(triggered_sell[:3])}")
        
        # Market context
        context_desc = []
        if market_context['trend'] != 'neutral':
            context_desc.append(f"{market_context['trend']} trend")
        if market_context['volatility'] != 'normal':
            context_desc.append(f"{market_context['volatility']} volatility")
        if market_context['support_resistance']:
            context_desc.append(market_context['support_resistance'])
        
        if context_desc:
            reasoning_parts.append(f"Market context: {', '.join(context_desc)}")
        
        return " | ".join(reasoning_parts)
    
    def batch_decisions(self, df: pd.DataFrame, 
                       lookback_window: int = 100) -> pd.DataFrame:
        """
        Generate decisions for entire dataframe with historical context.
        
        Args:
            df: DataFrame with OHLCV and indicators
            lookback_window: Minimum rows needed before making decisions
            
        Returns:
            DataFrame with decision columns added
        """
        decisions = []
        
        for i in range(lookback_window, len(df)):
            # Use data up to current point
            df_subset = df.iloc[:i+1]
            decision = self.generate_decision(df_subset, -1)
            
            decisions.append({
                'signal': decision.signal.value,
                'confidence': decision.confidence,
                'buy_score': decision.buy_score,
                'sell_score': decision.sell_score,
                'num_triggered': len(decision.triggered_indicators)
            })
        
        # Create decision dataframe
        decision_df = pd.DataFrame(decisions, index=df.index[lookback_window:])
        
        # Merge with original dataframe
        result = df.copy()
        for col in decision_df.columns:
            result[col] = np.nan
            result.loc[decision_df.index, col] = decision_df[col]
        
        return result