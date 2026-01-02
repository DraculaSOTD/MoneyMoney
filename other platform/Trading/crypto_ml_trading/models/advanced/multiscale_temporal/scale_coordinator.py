"""
Scale Coordinator for Multi-Scale Temporal Analysis.

Coordinates analysis and decision-making across multiple temporal scales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.multiscale_temporal.wavelet_analyzer import WaveletAnalyzer, WaveletFeatures
from models.advanced.multiscale_temporal.multiscale_decomposition import MultiScaleDecomposition, MultiScaleFeatures
from models.advanced.multiscale_temporal.temporal_patterns import TemporalPatternDetector, TemporalPattern
from models.advanced.multiscale_temporal.hierarchical_model import HierarchicalTemporalModel


@dataclass
class ScaleSignal:
    """Signal generated at a specific temporal scale."""
    scale: int
    timestamp: datetime
    signal_type: str
    strength: float
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    components: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossScaleAnalysis:
    """Results of cross-scale analysis."""
    dominant_trend: str
    trend_strength: float
    scale_agreement: float
    conflicting_scales: List[int]
    consensus_scales: List[int]
    overall_signal: ScaleSignal
    scale_contributions: Dict[int, float]


@dataclass
class ScaleConfig:
    """Configuration for a temporal scale."""
    scale_id: int
    time_horizon: str  # '1m', '5m', '1h', '4h', '1d', etc.
    weight: float
    enabled: bool
    analysis_methods: List[str]
    update_frequency: int  # seconds
    lookback_periods: int


class ScaleCoordinator:
    """
    Coordinates multi-scale temporal analysis and decision-making.
    
    Features:
    - Multi-scale signal aggregation
    - Cross-scale consensus analysis
    - Temporal hierarchy coordination
    - Real-time scale monitoring
    - Adaptive scale weighting
    - Signal reconciliation
    - Regime-dependent scaling
    """
    
    def __init__(self,
                 scale_configs: Optional[List[ScaleConfig]] = None,
                 reconciliation_method: str = 'weighted_voting',
                 update_interval: float = 1.0):
        """
        Initialize scale coordinator.
        
        Args:
            scale_configs: Configuration for each temporal scale
            reconciliation_method: Method for reconciling cross-scale signals
            update_interval: Update interval in seconds
        """
        self.scale_configs = scale_configs or self._create_default_scales()
        self.reconciliation_method = reconciliation_method
        self.update_interval = update_interval
        
        # Analysis components
        self.wavelet_analyzer = WaveletAnalyzer()
        self.decomposition = MultiScaleDecomposition()
        self.pattern_detector = TemporalPatternDetector()
        self.hierarchical_model = HierarchicalTemporalModel()
        
        # Scale-specific data and signals
        self.scale_data: Dict[int, deque] = {}
        self.scale_signals: Dict[int, List[ScaleSignal]] = defaultdict(list)
        self.scale_features: Dict[int, Dict] = {}
        
        # Cross-scale analysis
        self.cross_scale_history: List[CrossScaleAnalysis] = []
        self.scale_weights: Dict[int, float] = {}
        self.scale_performance: Dict[int, Dict[str, float]] = defaultdict(dict)
        
        # Real-time coordination
        self.is_running = False
        self.coordination_thread = None
        self.data_locks: Dict[int, threading.Lock] = defaultdict(threading.Lock)
        
        # Initialize scales
        self._initialize_scales()
    
    def _create_default_scales(self) -> List[ScaleConfig]:
        """Create default scale configurations."""
        return [
            ScaleConfig(
                scale_id=1,
                time_horizon='1m',
                weight=0.1,
                enabled=True,
                analysis_methods=['wavelet', 'patterns'],
                update_frequency=60,
                lookback_periods=100
            ),
            ScaleConfig(
                scale_id=2,
                time_horizon='5m',
                weight=0.15,
                enabled=True,
                analysis_methods=['wavelet', 'decomposition', 'patterns'],
                update_frequency=300,
                lookback_periods=200
            ),
            ScaleConfig(
                scale_id=3,
                time_horizon='15m',
                weight=0.2,
                enabled=True,
                analysis_methods=['wavelet', 'decomposition', 'hierarchical'],
                update_frequency=900,
                lookback_periods=150
            ),
            ScaleConfig(
                scale_id=4,
                time_horizon='1h',
                weight=0.25,
                enabled=True,
                analysis_methods=['decomposition', 'hierarchical', 'patterns'],
                update_frequency=3600,
                lookback_periods=168
            ),
            ScaleConfig(
                scale_id=5,
                time_horizon='4h',
                weight=0.2,
                enabled=True,
                analysis_methods=['hierarchical', 'decomposition'],
                update_frequency=14400,
                lookback_periods=60
            ),
            ScaleConfig(
                scale_id=6,
                time_horizon='1d',
                weight=0.1,
                enabled=True,
                analysis_methods=['hierarchical', 'decomposition'],
                update_frequency=86400,
                lookback_periods=30
            )
        ]
    
    def _initialize_scales(self) -> None:
        """Initialize scale-specific components."""
        for config in self.scale_configs:
            if config.enabled:
                self.scale_data[config.scale_id] = deque(maxlen=config.lookback_periods)
                self.scale_weights[config.scale_id] = config.weight
                
                # Initialize performance tracking
                self.scale_performance[config.scale_id] = {
                    'accuracy': 0.5,
                    'precision': 0.5,
                    'recall': 0.5,
                    'sharpe_ratio': 0.0,
                    'win_rate': 0.5,
                    'avg_return': 0.0
                }
    
    def add_data_point(self, 
                      scale_id: int,
                      data_point: Union[float, np.ndarray],
                      timestamp: Optional[datetime] = None) -> None:
        """
        Add data point for a specific scale.
        
        Args:
            scale_id: Scale identifier
            data_point: Data point to add
            timestamp: Timestamp of data point
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if scale_id not in self.scale_data:
            return
        
        with self.data_locks[scale_id]:
            # Store data with timestamp
            self.scale_data[scale_id].append({
                'data': data_point,
                'timestamp': timestamp
            })
        
        # Trigger analysis if enough data
        self._trigger_scale_analysis(scale_id)
    
    def _trigger_scale_analysis(self, scale_id: int) -> None:
        """Trigger analysis for a specific scale."""
        config = self._get_scale_config(scale_id)
        if not config or not config.enabled:
            return
        
        # Check if we have enough data
        if len(self.scale_data[scale_id]) < 10:
            return
        
        # Extract time series
        with self.data_locks[scale_id]:
            data_points = [item['data'] for item in self.scale_data[scale_id]]
            timestamps = [item['timestamp'] for item in self.scale_data[scale_id]]
        
        # Convert to numpy array
        if isinstance(data_points[0], (int, float)):
            time_series = np.array(data_points)
        else:
            time_series = np.array(data_points)
            if len(time_series.shape) > 1:
                time_series = time_series[:, 0]  # Use first column if multivariate
        
        # Perform scale-specific analysis
        self._analyze_scale(scale_id, time_series, timestamps[-1])
    
    def _get_scale_config(self, scale_id: int) -> Optional[ScaleConfig]:
        """Get configuration for a scale."""
        for config in self.scale_configs:
            if config.scale_id == scale_id:
                return config
        return None
    
    def _analyze_scale(self, 
                      scale_id: int, 
                      time_series: np.ndarray,
                      timestamp: datetime) -> None:
        """Perform analysis for a specific scale."""
        config = self._get_scale_config(scale_id)
        if not config:
            return
        
        signals = []
        features = {}
        
        # Wavelet analysis
        if 'wavelet' in config.analysis_methods:
            try:
                wavelet_features = self.wavelet_analyzer.extract_features(time_series)
                features['wavelet'] = wavelet_features.__dict__
                
                # Generate wavelet-based signals
                wavelet_signals = self._generate_wavelet_signals(
                    scale_id, wavelet_features, timestamp
                )
                signals.extend(wavelet_signals)
                
            except Exception as e:
                print(f"Wavelet analysis failed for scale {scale_id}: {e}")
        
        # Multi-scale decomposition
        if 'decomposition' in config.analysis_methods:
            try:
                decomp_features = self.decomposition.extract_multiscale_features(time_series)
                features['decomposition'] = decomp_features.__dict__
                
                # Generate decomposition-based signals
                decomp_signals = self._generate_decomposition_signals(
                    scale_id, decomp_features, timestamp
                )
                signals.extend(decomp_signals)
                
            except Exception as e:
                print(f"Decomposition analysis failed for scale {scale_id}: {e}")
        
        # Pattern detection
        if 'patterns' in config.analysis_methods:
            try:
                patterns = self.pattern_detector.detect_patterns_multiscale(
                    time_series, [1, 2, 4]
                )
                features['patterns'] = self._summarize_patterns(patterns)
                
                # Generate pattern-based signals
                pattern_signals = self._generate_pattern_signals(
                    scale_id, patterns, timestamp
                )
                signals.extend(pattern_signals)
                
            except Exception as e:
                print(f"Pattern analysis failed for scale {scale_id}: {e}")
        
        # Hierarchical modeling
        if 'hierarchical' in config.analysis_methods:
            try:
                if not self.hierarchical_model.is_fitted and len(time_series) > 20:
                    self.hierarchical_model.fit(time_series)
                
                if self.hierarchical_model.is_fitted:
                    forecasts = self.hierarchical_model.forecast([1, 5, 10])
                    features['hierarchical'] = {
                        'forecasts': {h: f.tolist() if hasattr(f, 'tolist') else f 
                                    for h, f in forecasts.items()},
                        'structure': self.hierarchical_model.get_hierarchy_summary()
                    }
                    
                    # Generate hierarchical signals
                    hier_signals = self._generate_hierarchical_signals(
                        scale_id, forecasts, timestamp
                    )
                    signals.extend(hier_signals)
                    
            except Exception as e:
                print(f"Hierarchical analysis failed for scale {scale_id}: {e}")
        
        # Store results
        self.scale_features[scale_id] = features
        self.scale_signals[scale_id].extend(signals)
        
        # Keep only recent signals
        cutoff_time = timestamp - timedelta(hours=24)
        self.scale_signals[scale_id] = [
            s for s in self.scale_signals[scale_id]
            if s.timestamp > cutoff_time
        ]
        
        # Trigger cross-scale analysis
        self._trigger_cross_scale_analysis(timestamp)
    
    def _generate_wavelet_signals(self,
                                 scale_id: int,
                                 features: WaveletFeatures,
                                 timestamp: datetime) -> List[ScaleSignal]:
        """Generate signals from wavelet analysis."""
        signals = []
        
        # Trend strength signal
        if features.trend_strength > 0.7:
            signals.append(ScaleSignal(
                scale=scale_id,
                timestamp=timestamp,
                signal_type='trend',
                strength=features.trend_strength,
                confidence=min(1.0, features.trend_strength * 1.2),
                direction='bullish' if features.trend_strength > 0 else 'bearish',
                components={'trend_strength': features.trend_strength},
                metadata={'source': 'wavelet', 'method': 'trend_analysis'}
            ))
        
        # Volatility signal
        if features.noise_level > 0.3:
            signals.append(ScaleSignal(
                scale=scale_id,
                timestamp=timestamp,
                signal_type='volatility',
                strength=features.noise_level,
                confidence=0.8,
                direction='neutral',
                components={'noise_level': features.noise_level},
                metadata={'source': 'wavelet', 'method': 'noise_analysis'}
            ))
        
        # Dominant frequency signals
        for freq in features.dominant_frequencies[:3]:  # Top 3 frequencies
            if freq > 0.1:  # High frequency component
                signals.append(ScaleSignal(
                    scale=scale_id,
                    timestamp=timestamp,
                    signal_type='oscillation',
                    strength=0.6,
                    confidence=0.7,
                    direction='neutral',
                    components={'frequency': freq},
                    metadata={'source': 'wavelet', 'method': 'frequency_analysis'}
                ))
        
        return signals
    
    def _generate_decomposition_signals(self,
                                      scale_id: int,
                                      features: MultiScaleFeatures,
                                      timestamp: datetime) -> List[ScaleSignal]:
        """Generate signals from multi-scale decomposition."""
        signals = []
        
        # Trend strength signal
        if features.trend_strength > 0.6:
            direction = 'bullish' if features.trend_strength > 0.5 else 'bearish'
            signals.append(ScaleSignal(
                scale=scale_id,
                timestamp=timestamp,
                signal_type='trend',
                strength=features.trend_strength,
                confidence=0.8,
                direction=direction,
                components={'trend_strength': features.trend_strength},
                metadata={'source': 'decomposition', 'method': 'trend_extraction'}
            ))
        
        # Cycle component signals
        for cycle_name, cycle_strength in features.cycle_components.items():
            if cycle_strength > 0.2:
                signals.append(ScaleSignal(
                    scale=scale_id,
                    timestamp=timestamp,
                    signal_type='cycle',
                    strength=cycle_strength,
                    confidence=0.7,
                    direction='neutral',
                    components={cycle_name: cycle_strength},
                    metadata={'source': 'decomposition', 'method': 'cycle_analysis'}
                ))
        
        # Complexity signal
        if features.complexity_measure > 0.5:
            signals.append(ScaleSignal(
                scale=scale_id,
                timestamp=timestamp,
                signal_type='complexity',
                strength=features.complexity_measure,
                confidence=0.6,
                direction='neutral',
                components={'complexity': features.complexity_measure},
                metadata={'source': 'decomposition', 'method': 'complexity_analysis'}
            ))
        
        return signals
    
    def _generate_pattern_signals(self,
                                scale_id: int,
                                patterns: Dict[int, List[TemporalPattern]],
                                timestamp: datetime) -> List[ScaleSignal]:
        """Generate signals from pattern detection."""
        signals = []
        
        for pattern_scale, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern.confidence > 0.7:
                    # Determine direction based on pattern type and features
                    direction = self._determine_pattern_direction(pattern)
                    
                    signals.append(ScaleSignal(
                        scale=scale_id,
                        timestamp=timestamp,
                        signal_type=pattern.pattern_type,
                        strength=pattern.confidence,
                        confidence=pattern.confidence,
                        direction=direction,
                        components=pattern.features,
                        metadata={
                            'source': 'patterns',
                            'pattern_id': pattern.pattern_id,
                            'pattern_scale': pattern_scale
                        }
                    ))
        
        return signals
    
    def _determine_pattern_direction(self, pattern: TemporalPattern) -> str:
        """Determine directional signal from pattern."""
        if pattern.pattern_type == 'trend_change':
            change_magnitude = pattern.features.get('change_magnitude', 0)
            return 'bullish' if change_magnitude > 0 else 'bearish'
        elif pattern.pattern_type == 'motif':
            trend = pattern.features.get('trend', 0)
            if abs(trend) > 0.1:
                return 'bullish' if trend > 0 else 'bearish'
        
        return 'neutral'
    
    def _generate_hierarchical_signals(self,
                                     scale_id: int,
                                     forecasts: Dict[int, Dict[str, np.ndarray]],
                                     timestamp: datetime) -> List[ScaleSignal]:
        """Generate signals from hierarchical modeling."""
        signals = []
        
        for horizon, level_forecasts in forecasts.items():
            for level_name, forecast_array in level_forecasts.items():
                if len(forecast_array) > 0:
                    # Calculate trend from forecast
                    if len(forecast_array) > 1:
                        trend = np.mean(np.diff(forecast_array))
                        trend_strength = abs(trend) / (np.std(forecast_array) + 1e-8)
                        
                        if trend_strength > 0.3:
                            direction = 'bullish' if trend > 0 else 'bearish'
                            
                            signals.append(ScaleSignal(
                                scale=scale_id,
                                timestamp=timestamp,
                                signal_type='forecast_trend',
                                strength=min(1.0, trend_strength),
                                confidence=0.7,
                                direction=direction,
                                components={
                                    'trend': trend,
                                    'horizon': horizon,
                                    'level': level_name
                                },
                                metadata={
                                    'source': 'hierarchical',
                                    'method': 'forecast_analysis'
                                }
                            ))
        
        return signals
    
    def _summarize_patterns(self, patterns: Dict[int, List[TemporalPattern]]) -> Dict[str, Any]:
        """Summarize pattern detection results."""
        summary = {
            'total_patterns': sum(len(pattern_list) for pattern_list in patterns.values()),
            'patterns_by_scale': {scale: len(pattern_list) for scale, pattern_list in patterns.items()},
            'pattern_types': defaultdict(int),
            'avg_confidence': 0.0
        }
        
        all_patterns = []
        for pattern_list in patterns.values():
            all_patterns.extend(pattern_list)
        
        if all_patterns:
            for pattern in all_patterns:
                summary['pattern_types'][pattern.pattern_type] += 1
            
            summary['avg_confidence'] = np.mean([p.confidence for p in all_patterns])
            summary['pattern_types'] = dict(summary['pattern_types'])
        
        return summary
    
    def _trigger_cross_scale_analysis(self, timestamp: datetime) -> None:
        """Trigger cross-scale analysis and signal reconciliation."""
        # Check if we have recent signals from multiple scales
        active_scales = []
        recent_cutoff = timestamp - timedelta(minutes=30)
        
        for scale_id, signals in self.scale_signals.items():
            recent_signals = [s for s in signals if s.timestamp > recent_cutoff]
            if recent_signals:
                active_scales.append(scale_id)
        
        if len(active_scales) < 2:
            return  # Need at least 2 scales for cross-scale analysis
        
        # Perform cross-scale analysis
        analysis = self._perform_cross_scale_analysis(active_scales, timestamp)
        
        # Store analysis
        self.cross_scale_history.append(analysis)
        
        # Keep only recent analyses
        cutoff_time = timestamp - timedelta(hours=24)
        self.cross_scale_history = [
            a for a in self.cross_scale_history
            if a.overall_signal.timestamp > cutoff_time
        ]
    
    def _perform_cross_scale_analysis(self,
                                    active_scales: List[int],
                                    timestamp: datetime) -> CrossScaleAnalysis:
        """Perform cross-scale analysis."""
        # Collect recent signals from all scales
        recent_cutoff = timestamp - timedelta(minutes=30)
        scale_signals_dict = {}
        
        for scale_id in active_scales:
            recent_signals = [
                s for s in self.scale_signals[scale_id]
                if s.timestamp > recent_cutoff
            ]
            scale_signals_dict[scale_id] = recent_signals
        
        # Analyze signal consensus
        signal_consensus = self._analyze_signal_consensus(scale_signals_dict)
        
        # Determine dominant trend
        trend_votes = defaultdict(float)
        total_weight = 0
        
        for scale_id, signals in scale_signals_dict.items():
            scale_weight = self.scale_weights.get(scale_id, 1.0)
            
            for signal in signals:
                if signal.signal_type in ['trend', 'forecast_trend']:
                    vote_weight = signal.strength * signal.confidence * scale_weight
                    trend_votes[signal.direction] += vote_weight
                    total_weight += vote_weight
        
        # Determine dominant trend
        if total_weight > 0:
            dominant_trend = max(trend_votes.keys(), key=lambda k: trend_votes[k])
            trend_strength = trend_votes[dominant_trend] / total_weight
        else:
            dominant_trend = 'neutral'
            trend_strength = 0.0
        
        # Calculate scale agreement
        scale_agreement = self._calculate_scale_agreement(scale_signals_dict, dominant_trend)
        
        # Identify conflicting and consensus scales
        conflicting_scales = []
        consensus_scales = []
        
        for scale_id, signals in scale_signals_dict.items():
            scale_trend = self._get_scale_dominant_trend(signals)
            if scale_trend == dominant_trend:
                consensus_scales.append(scale_id)
            elif scale_trend != 'neutral':
                conflicting_scales.append(scale_id)
        
        # Calculate scale contributions
        scale_contributions = {}
        for scale_id in active_scales:
            contribution = self._calculate_scale_contribution(
                scale_id, scale_signals_dict.get(scale_id, []), dominant_trend
            )
            scale_contributions[scale_id] = contribution
        
        # Generate overall signal
        overall_signal = self._generate_overall_signal(
            scale_signals_dict, dominant_trend, trend_strength, timestamp
        )
        
        return CrossScaleAnalysis(
            dominant_trend=dominant_trend,
            trend_strength=trend_strength,
            scale_agreement=scale_agreement,
            conflicting_scales=conflicting_scales,
            consensus_scales=consensus_scales,
            overall_signal=overall_signal,
            scale_contributions=scale_contributions
        )
    
    def _analyze_signal_consensus(self, scale_signals: Dict[int, List[ScaleSignal]]) -> Dict[str, float]:
        """Analyze consensus across scale signals."""
        signal_types = set()
        for signals in scale_signals.values():
            for signal in signals:
                signal_types.add(signal.signal_type)
        
        consensus = {}
        
        for signal_type in signal_types:
            type_signals = []
            for signals in scale_signals.values():
                type_signals.extend([s for s in signals if s.signal_type == signal_type])
            
            if type_signals:
                # Calculate consensus metrics
                strengths = [s.strength for s in type_signals]
                confidences = [s.confidence for s in type_signals]
                
                consensus[signal_type] = {
                    'count': len(type_signals),
                    'avg_strength': np.mean(strengths),
                    'avg_confidence': np.mean(confidences),
                    'agreement': self._calculate_signal_agreement(type_signals)
                }
        
        return consensus
    
    def _calculate_signal_agreement(self, signals: List[ScaleSignal]) -> float:
        """Calculate agreement level among signals."""
        if len(signals) < 2:
            return 1.0
        
        # Group by direction
        direction_counts = defaultdict(int)
        for signal in signals:
            direction_counts[signal.direction] += 1
        
        # Calculate agreement as proportion of majority direction
        max_count = max(direction_counts.values())
        agreement = max_count / len(signals)
        
        return agreement
    
    def _calculate_scale_agreement(self,
                                 scale_signals: Dict[int, List[ScaleSignal]],
                                 dominant_trend: str) -> float:
        """Calculate agreement level across scales."""
        if not scale_signals:
            return 0.0
        
        agreeing_scales = 0
        total_scales = len(scale_signals)
        
        for scale_id, signals in scale_signals.items():
            scale_trend = self._get_scale_dominant_trend(signals)
            if scale_trend == dominant_trend or scale_trend == 'neutral':
                agreeing_scales += 1
        
        return agreeing_scales / total_scales if total_scales > 0 else 0.0
    
    def _get_scale_dominant_trend(self, signals: List[ScaleSignal]) -> str:
        """Get dominant trend for a scale's signals."""
        if not signals:
            return 'neutral'
        
        trend_signals = [s for s in signals if s.signal_type in ['trend', 'forecast_trend']]
        
        if not trend_signals:
            return 'neutral'
        
        # Weight by strength and confidence
        direction_weights = defaultdict(float)
        
        for signal in trend_signals:
            weight = signal.strength * signal.confidence
            direction_weights[signal.direction] += weight
        
        if not direction_weights:
            return 'neutral'
        
        return max(direction_weights.keys(), key=lambda k: direction_weights[k])
    
    def _calculate_scale_contribution(self,
                                    scale_id: int,
                                    signals: List[ScaleSignal],
                                    dominant_trend: str) -> float:
        """Calculate scale's contribution to overall signal."""
        if not signals:
            return 0.0
        
        scale_weight = self.scale_weights.get(scale_id, 1.0)
        
        # Calculate contribution based on signal alignment and strength
        contribution = 0.0
        total_signals = 0
        
        for signal in signals:
            signal_weight = signal.strength * signal.confidence
            
            if signal.direction == dominant_trend:
                contribution += signal_weight * scale_weight
            elif signal.direction != 'neutral':
                contribution -= signal_weight * scale_weight * 0.5  # Penalty for conflict
            
            total_signals += 1
        
        # Normalize by number of signals
        if total_signals > 0:
            contribution /= total_signals
        
        return max(0.0, min(1.0, contribution))
    
    def _generate_overall_signal(self,
                               scale_signals: Dict[int, List[ScaleSignal]],
                               dominant_trend: str,
                               trend_strength: float,
                               timestamp: datetime) -> ScaleSignal:
        """Generate overall coordinated signal."""
        # Aggregate signal components
        all_signals = []
        for signals in scale_signals.values():
            all_signals.extend(signals)
        
        if not all_signals:
            return ScaleSignal(
                scale=0,  # Overall signal
                timestamp=timestamp,
                signal_type='coordinated',
                strength=0.0,
                confidence=0.0,
                direction='neutral',
                components={},
                metadata={'source': 'coordinator', 'method': 'cross_scale_analysis'}
            )
        
        # Calculate weighted components
        components = defaultdict(float)
        total_weight = 0
        
        for scale_id, signals in scale_signals.items():
            scale_weight = self.scale_weights.get(scale_id, 1.0)
            
            for signal in signals:
                signal_weight = signal.strength * signal.confidence * scale_weight
                
                for comp_name, comp_value in signal.components.items():
                    components[f"{comp_name}_scale_{signal.scale}"] = comp_value * signal_weight
                
                total_weight += signal_weight
        
        # Normalize components
        if total_weight > 0:
            components = {k: v / total_weight for k, v in components.items()}
        
        # Calculate overall confidence
        confidences = [s.confidence for s in all_signals]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Adjust confidence based on scale agreement
        scale_agreement = self._calculate_scale_agreement(scale_signals, dominant_trend)
        adjusted_confidence = overall_confidence * scale_agreement
        
        return ScaleSignal(
            scale=0,  # Overall signal
            timestamp=timestamp,
            signal_type='coordinated',
            strength=trend_strength,
            confidence=adjusted_confidence,
            direction=dominant_trend,
            components=dict(components),
            metadata={
                'source': 'coordinator',
                'method': 'cross_scale_analysis',
                'num_scales': len(scale_signals),
                'num_signals': len(all_signals),
                'scale_agreement': scale_agreement
            }
        )
    
    def get_current_analysis(self) -> Optional[CrossScaleAnalysis]:
        """Get most recent cross-scale analysis."""
        if self.cross_scale_history:
            return self.cross_scale_history[-1]
        return None
    
    def get_scale_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all scales."""
        status = {}
        
        for config in self.scale_configs:
            scale_id = config.scale_id
            
            # Basic info
            scale_status = {
                'scale_id': scale_id,
                'time_horizon': config.time_horizon,
                'enabled': config.enabled,
                'weight': self.scale_weights.get(scale_id, 0.0),
                'data_points': len(self.scale_data.get(scale_id, [])),
                'recent_signals': len([
                    s for s in self.scale_signals.get(scale_id, [])
                    if (datetime.now() - s.timestamp).total_seconds() < 3600
                ])
            }
            
            # Performance metrics
            if scale_id in self.scale_performance:
                scale_status['performance'] = self.scale_performance[scale_id]
            
            # Recent features
            if scale_id in self.scale_features:
                scale_status['features'] = self.scale_features[scale_id]
            
            status[scale_id] = scale_status
        
        return status
    
    def update_scale_weights(self, new_weights: Dict[int, float]) -> None:
        """Update scale weights."""
        for scale_id, weight in new_weights.items():
            if scale_id in self.scale_weights:
                self.scale_weights[scale_id] = max(0.0, min(1.0, weight))
        
        # Normalize weights
        total_weight = sum(self.scale_weights.values())
        if total_weight > 0:
            for scale_id in self.scale_weights:
                self.scale_weights[scale_id] /= total_weight
    
    def get_signal_history(self,
                          scale_id: Optional[int] = None,
                          hours_back: int = 24) -> List[ScaleSignal]:
        """Get signal history for specified scale or all scales."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        if scale_id is not None:
            return [
                s for s in self.scale_signals.get(scale_id, [])
                if s.timestamp > cutoff_time
            ]
        else:
            all_signals = []
            for signals in self.scale_signals.values():
                all_signals.extend([
                    s for s in signals if s.timestamp > cutoff_time
                ])
            return sorted(all_signals, key=lambda s: s.timestamp)
    
    def start_coordination(self) -> None:
        """Start real-time coordination thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        print("Scale coordination started")
    
    def stop_coordination(self) -> None:
        """Stop real-time coordination."""
        self.is_running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        print("Scale coordination stopped")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.is_running:
            try:
                # Perform periodic maintenance
                self._update_scale_performance()
                self._adapt_scale_weights()
                self._cleanup_old_data()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_scale_performance(self) -> None:
        """Update performance metrics for each scale."""
        # This would be implemented with actual trading results
        # For now, just placeholder
        pass
    
    def _adapt_scale_weights(self) -> None:
        """Adapt scale weights based on performance."""
        # Simple adaptation based on recent signal accuracy
        # In practice, this would use more sophisticated methods
        
        total_performance = 0
        performance_scores = {}
        
        for scale_id in self.scale_weights:
            # Get recent performance score
            perf = self.scale_performance.get(scale_id, {})
            accuracy = perf.get('accuracy', 0.5)
            sharpe = perf.get('sharpe_ratio', 0.0)
            
            # Combined performance score
            score = accuracy * 0.7 + min(1.0, max(0.0, (sharpe + 1) / 2)) * 0.3
            performance_scores[scale_id] = score
            total_performance += score
        
        # Update weights based on performance
        if total_performance > 0:
            for scale_id in self.scale_weights:
                new_weight = performance_scores[scale_id] / total_performance
                # Smooth weight updates
                current_weight = self.scale_weights[scale_id]
                self.scale_weights[scale_id] = 0.9 * current_weight + 0.1 * new_weight
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data and signals."""
        cutoff_time = datetime.now() - timedelta(hours=48)
        
        # Clean up signal history
        for scale_id in self.scale_signals:
            self.scale_signals[scale_id] = [
                s for s in self.scale_signals[scale_id]
                if s.timestamp > cutoff_time
            ]
        
        # Clean up cross-scale history
        self.cross_scale_history = [
            analysis for analysis in self.cross_scale_history
            if analysis.overall_signal.timestamp > cutoff_time
        ]