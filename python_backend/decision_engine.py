import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class TradingSignal:
    action: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    confidence: float
    reasons: List[str]
    strategy: str
    metadata: Dict[str, Any]

class DecisionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_weights = config.get('signal_weights', {
            'ml_prediction': 0.4,
            'technical': 0.3,
            'sentiment': 0.2,
            'regime': 0.1
        })
        
        self.thresholds = config.get('signal_thresholds', {
            'strong_buy': 0.8,
            'buy': 0.6,
            'sell': -0.6,
            'strong_sell': -0.8
        })
        
    def generate_signal(self, symbol: str, predictions: Dict[str, Any],
                       features: Dict[str, Any], 
                       current_positions: List[Any]) -> Optional[Dict[str, Any]]:
        try:
            # Aggregate ML model predictions
            ml_signal = self._aggregate_ml_predictions(predictions)
            
            # Technical analysis signal
            technical_signal = self._analyze_technical_indicators(features)
            
            # Sentiment signal (if available)
            sentiment_signal = self._analyze_sentiment(predictions.get('sentiment', {}))
            
            # Market regime signal
            regime_signal = self._analyze_market_regime(predictions.get('hmm', {}))
            
            # Combine signals
            combined_signal = self._combine_signals({
                'ml_prediction': ml_signal,
                'technical': technical_signal,
                'sentiment': sentiment_signal,
                'regime': regime_signal
            })
            
            # Apply position management rules
            final_signal = self._apply_position_rules(
                combined_signal, symbol, current_positions
            )
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
            
    def _aggregate_ml_predictions(self, predictions: Dict[str, Any]) -> float:
        if not predictions:
            return 0.0
            
        signals = []
        
        # Statistical models
        if 'arima' in predictions:
            arima_pred = predictions['arima'].get('prediction', 0)
            arima_conf = predictions['arima'].get('confidence', 0.5)
            signals.append(self._normalize_prediction(arima_pred, arima_conf))
            
        if 'garch' in predictions:
            garch_vol = predictions['garch'].get('volatility', 0)
            # Lower volatility -> more confidence in direction
            vol_signal = 1.0 if garch_vol < 0.02 else 0.5
            signals.append(vol_signal)
            
        # Deep learning models
        if 'gru_attention' in predictions:
            gru_pred = predictions['gru_attention'].get('prediction', 0)
            gru_conf = predictions['gru_attention'].get('confidence', 0.5)
            signals.append(self._normalize_prediction(gru_pred, gru_conf))
            
        if 'cnn' in predictions:
            cnn_pred = predictions['cnn'].get('prediction', 0)
            cnn_conf = predictions['cnn'].get('confidence', 0.5)
            signals.append(self._normalize_prediction(cnn_pred, cnn_conf))
            
        # Ensemble prediction
        if 'ensemble' in predictions:
            ensemble_pred = predictions['ensemble'].get('prediction', 0)
            ensemble_conf = predictions['ensemble'].get('confidence', 0.5)
            # Give ensemble higher weight
            signals.append(self._normalize_prediction(ensemble_pred, ensemble_conf) * 1.5)
            
        # Average all signals
        if signals:
            return np.mean(signals)
        return 0.0
        
    def _normalize_prediction(self, prediction: float, confidence: float) -> float:
        # Convert price prediction to directional signal (-1 to 1)
        # This is simplified - in reality would compare to current price
        signal = np.tanh(prediction * 10)  # Sigmoid-like normalization
        return signal * confidence
        
    def _analyze_technical_indicators(self, features: Dict[str, Any]) -> float:
        signals = []
        
        # RSI
        rsi = features.get('rsi', 50)
        if rsi < 30:
            signals.append(0.8)  # Oversold -> Buy
        elif rsi > 70:
            signals.append(-0.8)  # Overbought -> Sell
        else:
            signals.append((50 - rsi) / 50)  # Neutral zone
            
        # MACD
        macd_signal = features.get('macd_signal', 0)
        if macd_signal > 0:
            signals.append(0.6)
        elif macd_signal < 0:
            signals.append(-0.6)
            
        # Bollinger Bands
        bb_position = features.get('bb_position', 0.5)
        if bb_position < 0.2:
            signals.append(0.7)  # Near lower band
        elif bb_position > 0.8:
            signals.append(-0.7)  # Near upper band
            
        # Moving average crossovers
        ma_cross = features.get('ma_crossover', 0)
        if ma_cross > 0:
            signals.append(0.8)
        elif ma_cross < 0:
            signals.append(-0.8)
            
        # Volume confirmation
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            # High volume confirms the signal
            if signals:
                signals = [s * 1.2 for s in signals]
                
        return np.mean(signals) if signals else 0.0
        
    def _analyze_sentiment(self, sentiment_data: Dict[str, Any]) -> float:
        if not sentiment_data:
            return 0.0
            
        sentiment_score = sentiment_data.get('score', 0)
        sentiment_conf = sentiment_data.get('confidence', 0.5)
        
        # Check for sentiment trend if available
        if 'trend' in sentiment_data:
            trend = sentiment_data['trend']
            # Amplify signal if trend aligns with current sentiment
            if (sentiment_score > 0 and trend > 0) or (sentiment_score < 0 and trend < 0):
                sentiment_score *= (1 + abs(trend))
                
        # Check volatility
        if 'volatility' in sentiment_data:
            volatility = sentiment_data['volatility']
            # Reduce confidence in highly volatile sentiment
            if volatility > 0.3:
                sentiment_conf *= 0.7
        
        # Convert sentiment to signal
        # Positive sentiment -> Buy signal
        # Negative sentiment -> Sell signal
        return np.tanh(sentiment_score) * sentiment_conf
        
    def _analyze_market_regime(self, regime_data: Dict[str, Any]) -> float:
        if not regime_data:
            return 0.0
            
        regime = regime_data.get('regime', 'neutral')
        regime_conf = regime_data.get('confidence', 0.5)
        
        regime_signals = {
            'bullish': 0.7,
            'bearish': -0.7,
            'volatile': 0.0,  # Stay out in volatile markets
            'neutral': 0.2
        }
        
        return regime_signals.get(regime, 0.0) * regime_conf
        
    def _combine_signals(self, signals: Dict[str, float]) -> TradingSignal:
        # Weighted average of signals
        weighted_sum = 0.0
        total_weight = 0.0
        active_signals = []
        
        for signal_type, signal_value in signals.items():
            if signal_value != 0:
                weight = self.signal_weights.get(signal_type, 0.1)
                weighted_sum += signal_value * weight
                total_weight += weight
                active_signals.append((signal_type, signal_value))
                
        if total_weight == 0:
            return TradingSignal(
                action='hold',
                strength=SignalStrength.HOLD,
                confidence=0.0,
                reasons=['No signals available'],
                strategy='none',
                metadata={}
            )
            
        final_signal = weighted_sum / total_weight
        
        # Determine action and strength
        if final_signal >= self.thresholds['strong_buy']:
            action = 'buy'
            strength = SignalStrength.STRONG_BUY
        elif final_signal >= self.thresholds['buy']:
            action = 'buy'
            strength = SignalStrength.BUY
        elif final_signal <= self.thresholds['strong_sell']:
            action = 'sell'
            strength = SignalStrength.STRONG_SELL
        elif final_signal <= self.thresholds['sell']:
            action = 'sell'
            strength = SignalStrength.SELL
        else:
            action = 'hold'
            strength = SignalStrength.HOLD
            
        # Generate reasons
        reasons = []
        for signal_type, value in active_signals:
            if abs(value) > 0.5:
                direction = "bullish" if value > 0 else "bearish"
                reasons.append(f"{signal_type} is {direction} ({value:.2f})")
                
        # Calculate confidence
        confidence = min(abs(final_signal), 1.0)
        
        # Determine strategy
        dominant_signal = max(active_signals, key=lambda x: abs(x[1]))[0]
        strategy = f"{dominant_signal}_driven"
        
        return TradingSignal(
            action=action,
            strength=strength,
            confidence=confidence,
            reasons=reasons,
            strategy=strategy,
            metadata={
                'raw_signals': signals,
                'final_signal': final_signal,
                'active_signals': active_signals
            }
        )
        
    def _apply_position_rules(self, signal: TradingSignal, symbol: str,
                            current_positions: List[Any]) -> Dict[str, Any]:
        # Check if we already have a position in this symbol
        existing_position = None
        for pos in current_positions:
            if pos.symbol == symbol and pos.status.value == 'open':
                existing_position = pos
                break
                
        # Apply position management rules
        if existing_position:
            if signal.action == 'buy' and existing_position.side.value == 'SELL':
                # Reverse position
                signal.action = 'buy'
                signal.reasons.append("Reversing short position")
            elif signal.action == 'sell' and existing_position.side.value == 'BUY':
                # Close long position
                signal.action = 'sell'
                signal.reasons.append("Closing long position")
            elif signal.action == signal.action == existing_position.side.value.lower():
                # Already in position in same direction
                if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
                    # Only add to position on strong signals
                    signal.reasons.append("Adding to existing position")
                else:
                    signal.action = 'hold'
                    signal.reasons.append("Already in position")
                    
        # Risk checks
        open_positions = len([p for p in current_positions if p.status.value == 'open'])
        max_positions = self.config.get('risk_management', {}).get('max_positions', 10)
        
        if open_positions >= max_positions and signal.action != 'hold':
            if not existing_position:
                signal.action = 'hold'
                signal.reasons.append(f"Max positions limit reached ({max_positions})")
                
        return {
            'action': signal.action,
            'strength': signal.strength.value,
            'confidence': signal.confidence,
            'reasons': signal.reasons,
            'strategy': signal.strategy,
            'metadata': signal.metadata
        }
        
    def backtest_decisions(self, historical_data: pd.DataFrame,
                          predictions_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple backtest of decision engine
        results = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'correct_signals': 0,
            'profit_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        positions = []
        equity_curve = []
        
        for i, (idx, row) in enumerate(historical_data.iterrows()):
            if i >= len(predictions_history):
                break
                
            predictions = predictions_history[i]
            features = row.to_dict()
            
            signal = self.generate_signal('BTCUSDT', predictions, features, positions)
            
            if signal:
                results['total_signals'] += 1
                
                if signal['action'] == 'buy':
                    results['buy_signals'] += 1
                    # Simulate position entry
                    positions.append({
                        'entry_price': row['close'],
                        'entry_time': idx,
                        'type': 'long'
                    })
                    
                elif signal['action'] == 'sell':
                    results['sell_signals'] += 1
                    # Close long positions or open short
                    if positions and positions[-1]['type'] == 'long':
                        pnl = (row['close'] - positions[-1]['entry_price']) / positions[-1]['entry_price']
                        results['profit_loss'] += pnl
                        positions.pop()
                        
                equity_curve.append(results['profit_loss'])
                
        # Calculate metrics
        if equity_curve:
            returns = np.diff(equity_curve)
            if len(returns) > 0:
                results['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                
            # Max drawdown
            cumulative = np.array(equity_curve)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-8)
            results['max_drawdown'] = np.min(drawdown)
            
        return results