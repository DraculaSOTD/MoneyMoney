"""
Temporal Pattern Detection Across Multiple Scales.

Implements pattern recognition algorithms for multi-scale temporal analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class TemporalPattern:
    """Represents a detected temporal pattern."""
    pattern_id: str
    scale: int
    start_index: int
    end_index: int
    pattern_type: str
    confidence: float
    features: Dict[str, float]
    template: np.ndarray
    metadata: Dict


@dataclass
class ScalePattern:
    """Pattern detected at a specific scale."""
    scale: int
    frequency: float
    amplitude: float
    phase: float
    duration: int
    strength: float
    pattern_class: str


class TemporalPatternDetector:
    """
    Multi-scale temporal pattern detection system.
    
    Features:
    - Pattern discovery across different time scales
    - Motif detection using Matrix Profile
    - Seasonal pattern recognition
    - Trend change detection
    - Cycle identification
    - Pattern classification and clustering
    - Real-time pattern matching
    """
    
    def __init__(self,
                 min_pattern_length: int = 4,
                 max_pattern_length: int = 100,
                 similarity_threshold: float = 0.8,
                 min_confidence: float = 0.6):
        """
        Initialize temporal pattern detector.
        
        Args:
            min_pattern_length: Minimum pattern length
            max_pattern_length: Maximum pattern length
            similarity_threshold: Threshold for pattern similarity
            min_confidence: Minimum confidence for pattern detection
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        
        # Pattern storage
        self.discovered_patterns: Dict[int, List[TemporalPattern]] = defaultdict(list)
        self.pattern_templates: Dict[str, np.ndarray] = {}
        self.pattern_statistics: Dict[str, Dict] = {}
        
        # Real-time detection
        self.real_time_buffer: deque = deque(maxlen=1000)
        self.active_patterns: List[TemporalPattern] = []
        
    def detect_patterns_multiscale(self, 
                                  signal: np.ndarray,
                                  scales: List[int]) -> Dict[int, List[TemporalPattern]]:
        """
        Detect patterns across multiple scales.
        
        Args:
            signal: Input time series
            scales: List of scales to analyze
            
        Returns:
            Patterns detected at each scale
        """
        patterns_by_scale = {}
        
        for scale in scales:
            # Downsample signal to the target scale
            downsampled = self._downsample_signal(signal, scale)
            
            # Detect patterns at this scale
            scale_patterns = self._detect_patterns_single_scale(downsampled, scale)
            
            patterns_by_scale[scale] = scale_patterns
            
        return patterns_by_scale
    
    def _downsample_signal(self, signal: np.ndarray, scale: int) -> np.ndarray:
        """Downsample signal to target scale."""
        if scale <= 1:
            return signal
            
        # Simple downsampling by taking every nth point
        return signal[::scale]
    
    def _detect_patterns_single_scale(self, signal: np.ndarray, scale: int) -> List[TemporalPattern]:
        """Detect patterns at a single scale."""
        patterns = []
        
        # 1. Matrix Profile for motif discovery
        motif_patterns = self._discover_motifs(signal, scale)
        patterns.extend(motif_patterns)
        
        # 2. Seasonal patterns
        seasonal_patterns = self._detect_seasonal_patterns(signal, scale)
        patterns.extend(seasonal_patterns)
        
        # 3. Trend changes
        trend_patterns = self._detect_trend_changes(signal, scale)
        patterns.extend(trend_patterns)
        
        # 4. Cyclical patterns
        cycle_patterns = self._detect_cycles(signal, scale)
        patterns.extend(cycle_patterns)
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        
        return patterns
    
    def _discover_motifs(self, signal: np.ndarray, scale: int) -> List[TemporalPattern]:
        """Discover repeated motifs using Matrix Profile."""
        if len(signal) < self.min_pattern_length * 2:
            return []
            
        patterns = []
        
        # Try different pattern lengths
        for pattern_length in range(self.min_pattern_length, 
                                   min(self.max_pattern_length, len(signal) // 2)):
            
            # Compute matrix profile
            matrix_profile, profile_index = self._compute_matrix_profile(signal, pattern_length)
            
            # Find motifs (minimum values in matrix profile)
            motif_indices = self._find_motifs(matrix_profile, profile_index)
            
            for motif_idx in motif_indices:
                if motif_idx + pattern_length <= len(signal):
                    pattern = self._create_motif_pattern(
                        signal, motif_idx, pattern_length, scale, matrix_profile[motif_idx]
                    )
                    patterns.append(pattern)
                    
        return patterns
    
    def _compute_matrix_profile(self, signal: np.ndarray, 
                               pattern_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute matrix profile for motif discovery."""
        n = len(signal)
        matrix_profile = np.full(n - pattern_length + 1, np.inf)
        profile_index = np.zeros(n - pattern_length + 1, dtype=int)
        
        # Compute distances between all subsequences
        for i in range(n - pattern_length + 1):
            query = signal[i:i + pattern_length]
            
            for j in range(n - pattern_length + 1):
                if abs(i - j) < pattern_length:  # Avoid trivial matches
                    continue
                    
                candidate = signal[j:j + pattern_length]
                distance = self._compute_z_normalized_distance(query, candidate)
                
                if distance < matrix_profile[i]:
                    matrix_profile[i] = distance
                    profile_index[i] = j
                    
        return matrix_profile, profile_index
    
    def _compute_z_normalized_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute z-normalized Euclidean distance."""
        # Z-normalize sequences
        seq1_norm = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-8)
        seq2_norm = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-8)
        
        # Euclidean distance
        return np.sqrt(np.sum((seq1_norm - seq2_norm) ** 2))
    
    def _find_motifs(self, matrix_profile: np.ndarray, 
                    profile_index: np.ndarray) -> List[int]:
        """Find motif locations from matrix profile."""
        # Find local minima in matrix profile
        motifs = []
        
        for i in range(1, len(matrix_profile) - 1):
            if (matrix_profile[i] < matrix_profile[i-1] and 
                matrix_profile[i] < matrix_profile[i+1] and
                matrix_profile[i] < self.similarity_threshold):
                motifs.append(i)
                
        return motifs
    
    def _create_motif_pattern(self, signal: np.ndarray, start_idx: int,
                             length: int, scale: int, distance: float) -> TemporalPattern:
        """Create temporal pattern from motif detection."""
        template = signal[start_idx:start_idx + length]
        
        # Calculate features
        features = {
            'mean': np.mean(template),
            'std': np.std(template),
            'min': np.min(template),
            'max': np.max(template),
            'range': np.max(template) - np.min(template),
            'trend': np.polyfit(range(len(template)), template, 1)[0],
            'autocorr_lag1': np.corrcoef(template[:-1], template[1:])[0, 1] if len(template) > 1 else 0
        }
        
        # Confidence based on distance (lower distance = higher confidence)
        confidence = max(0, 1 - distance)
        
        return TemporalPattern(
            pattern_id=f"motif_{scale}_{start_idx}",
            scale=scale,
            start_index=start_idx * scale,  # Scale back to original indices
            end_index=(start_idx + length) * scale,
            pattern_type="motif",
            confidence=confidence,
            features=features,
            template=template,
            metadata={'distance': distance, 'length': length}
        )
    
    def _detect_seasonal_patterns(self, signal: np.ndarray, scale: int) -> List[TemporalPattern]:
        """Detect seasonal patterns."""
        patterns = []
        
        if len(signal) < 12:  # Need minimum data for seasonality
            return patterns
            
        # Test different seasonal periods
        potential_periods = [7, 24, 30, 168]  # Weekly, daily, monthly patterns
        potential_periods = [p for p in potential_periods if p < len(signal) // 3]
        
        for period in potential_periods:
            seasonal_strength = self._measure_seasonality(signal, period)
            
            if seasonal_strength > 0.3:  # Threshold for significant seasonality
                pattern = TemporalPattern(
                    pattern_id=f"seasonal_{scale}_{period}",
                    scale=scale,
                    start_index=0,
                    end_index=len(signal) * scale,
                    pattern_type="seasonal",
                    confidence=seasonal_strength,
                    features={'period': period, 'strength': seasonal_strength},
                    template=self._extract_seasonal_template(signal, period),
                    metadata={'period': period}
                )
                patterns.append(pattern)
                
        return patterns
    
    def _measure_seasonality(self, signal: np.ndarray, period: int) -> float:
        """Measure seasonal strength for a given period."""
        if len(signal) < period * 2:
            return 0.0
            
        # Calculate seasonal means
        seasonal_means = []
        for i in range(period):
            positions = np.arange(i, len(signal), period)
            if len(positions) > 1:
                seasonal_means.append(np.mean(signal[positions]))
            else:
                seasonal_means.append(signal[i] if i < len(signal) else 0)
                
        # Calculate strength as ratio of seasonal variance to total variance
        seasonal_var = np.var(seasonal_means)
        total_var = np.var(signal)
        
        if total_var == 0:
            return 0.0
            
        return min(1.0, seasonal_var / total_var)
    
    def _extract_seasonal_template(self, signal: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal template."""
        template = np.zeros(period)
        
        for i in range(period):
            positions = np.arange(i, len(signal), period)
            if len(positions) > 0:
                template[i] = np.mean(signal[positions])
                
        return template
    
    def _detect_trend_changes(self, signal: np.ndarray, scale: int) -> List[TemporalPattern]:
        """Detect trend change points."""
        patterns = []
        
        if len(signal) < 10:
            return patterns
            
        # Calculate rolling trend using linear regression
        window_size = min(20, len(signal) // 4)
        trends = []
        
        for i in range(window_size, len(signal) - window_size):
            window = signal[i-window_size:i+window_size]
            trend = np.polyfit(range(len(window)), window, 1)[0]
            trends.append(trend)
            
        trends = np.array(trends)
        
        # Detect significant trend changes
        trend_changes = []
        change_threshold = np.std(trends) * 2
        
        for i in range(1, len(trends)):
            if abs(trends[i] - trends[i-1]) > change_threshold:
                change_point = i + window_size
                
                # Create pattern around change point
                start_idx = max(0, change_point - window_size)
                end_idx = min(len(signal), change_point + window_size)
                
                pattern = TemporalPattern(
                    pattern_id=f"trend_change_{scale}_{change_point}",
                    scale=scale,
                    start_index=start_idx * scale,
                    end_index=end_idx * scale,
                    pattern_type="trend_change",
                    confidence=min(1.0, abs(trends[i] - trends[i-1]) / change_threshold),
                    features={
                        'trend_before': trends[i-1],
                        'trend_after': trends[i],
                        'change_magnitude': trends[i] - trends[i-1]
                    },
                    template=signal[start_idx:end_idx],
                    metadata={'change_point': change_point}
                )
                patterns.append(pattern)
                
        return patterns
    
    def _detect_cycles(self, signal: np.ndarray, scale: int) -> List[TemporalPattern]:
        """Detect cyclical patterns using FFT."""
        patterns = []
        
        if len(signal) < 8:
            return patterns
            
        # Perform FFT
        fft_signal = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft_signal) ** 2
        
        # Exclude DC component and very high frequencies
        valid_indices = np.where((frequencies > 0) & (frequencies < 0.5))[0]
        
        if len(valid_indices) == 0:
            return patterns
            
        # Find peaks in power spectrum
        peak_indices = []
        for i in range(1, len(valid_indices) - 1):
            idx = valid_indices[i]
            if (power_spectrum[idx] > power_spectrum[valid_indices[i-1]] and
                power_spectrum[idx] > power_spectrum[valid_indices[i+1]]):
                peak_indices.append(idx)
                
        # Create patterns for significant cycles
        for peak_idx in peak_indices:
            frequency = frequencies[peak_idx]
            period = int(1 / frequency) if frequency > 0 else len(signal)
            
            if period >= 4 and period <= len(signal) // 2:
                # Calculate cycle strength
                cycle_strength = power_spectrum[peak_idx] / np.sum(power_spectrum)
                
                if cycle_strength > 0.05:  # Minimum significance threshold
                    pattern = TemporalPattern(
                        pattern_id=f"cycle_{scale}_{period}",
                        scale=scale,
                        start_index=0,
                        end_index=len(signal) * scale,
                        pattern_type="cycle",
                        confidence=min(1.0, cycle_strength * 10),
                        features={
                            'frequency': frequency,
                            'period': period,
                            'amplitude': np.sqrt(power_spectrum[peak_idx]),
                            'strength': cycle_strength
                        },
                        template=self._extract_cycle_template(signal, period),
                        metadata={'frequency': frequency, 'period': period}
                    )
                    patterns.append(pattern)
                    
        return patterns
    
    def _extract_cycle_template(self, signal: np.ndarray, period: int) -> np.ndarray:
        """Extract representative cycle template."""
        if period >= len(signal):
            return signal
            
        # Extract multiple cycles and average them
        cycles = []
        for start in range(0, len(signal) - period, period):
            cycle = signal[start:start + period]
            cycles.append(cycle)
            
        if cycles:
            return np.mean(cycles, axis=0)
        else:
            return signal[:period]
    
    def classify_patterns(self, patterns: List[TemporalPattern]) -> Dict[str, List[TemporalPattern]]:
        """Classify patterns into categories."""
        classified = defaultdict(list)
        
        for pattern in patterns:
            # Primary classification by type
            pattern_class = pattern.pattern_type
            
            # Secondary classification by characteristics
            if pattern_class == "motif":
                # Classify motifs by shape characteristics
                if pattern.features.get('trend', 0) > 0.1:
                    pattern_class += "_upward"
                elif pattern.features.get('trend', 0) < -0.1:
                    pattern_class += "_downward"
                else:
                    pattern_class += "_flat"
                    
            elif pattern_class == "seasonal":
                # Classify by period
                period = pattern.features.get('period', 0)
                if period <= 24:
                    pattern_class += "_intraday"
                elif period <= 168:
                    pattern_class += "_weekly"
                else:
                    pattern_class += "_long_term"
                    
            elif pattern_class == "cycle":
                # Classify by frequency
                frequency = pattern.features.get('frequency', 0)
                if frequency > 0.1:
                    pattern_class += "_high_freq"
                elif frequency > 0.01:
                    pattern_class += "_medium_freq"
                else:
                    pattern_class += "_low_freq"
                    
            classified[pattern_class].append(pattern)
            
        return dict(classified)
    
    def find_pattern_relationships(self, patterns: List[TemporalPattern]) -> List[Tuple[str, str, float]]:
        """Find relationships between different patterns."""
        relationships = []
        
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Check temporal overlap
                overlap = self._calculate_temporal_overlap(pattern1, pattern2)
                
                # Check similarity
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                
                # Check frequency relationships
                freq_relationship = self._analyze_frequency_relationship(pattern1, pattern2)
                
                if overlap > 0.1 or similarity > 0.7 or freq_relationship > 0.8:
                    relationship_strength = max(overlap, similarity, freq_relationship)
                    relationships.append((pattern1.pattern_id, pattern2.pattern_id, relationship_strength))
                    
        return relationships
    
    def _calculate_temporal_overlap(self, pattern1: TemporalPattern, pattern2: TemporalPattern) -> float:
        """Calculate temporal overlap between patterns."""
        start1, end1 = pattern1.start_index, pattern1.end_index
        start2, end2 = pattern2.start_index, pattern2.end_index
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
            
        overlap_length = overlap_end - overlap_start
        total_length = max(end1, end2) - min(start1, start2)
        
        return overlap_length / total_length if total_length > 0 else 0.0
    
    def _calculate_pattern_similarity(self, pattern1: TemporalPattern, pattern2: TemporalPattern) -> float:
        """Calculate similarity between pattern templates."""
        template1 = pattern1.template
        template2 = pattern2.template
        
        # Normalize templates
        template1_norm = (template1 - np.mean(template1)) / (np.std(template1) + 1e-8)
        template2_norm = (template2 - np.mean(template2)) / (np.std(template2) + 1e-8)
        
        # Interpolate to same length
        if len(template1_norm) != len(template2_norm):
            max_length = max(len(template1_norm), len(template2_norm))
            template1_interp = np.interp(np.linspace(0, 1, max_length), 
                                       np.linspace(0, 1, len(template1_norm)), 
                                       template1_norm)
            template2_interp = np.interp(np.linspace(0, 1, max_length), 
                                       np.linspace(0, 1, len(template2_norm)), 
                                       template2_norm)
        else:
            template1_interp = template1_norm
            template2_interp = template2_norm
            
        # Calculate correlation
        if len(template1_interp) > 1:
            correlation = np.corrcoef(template1_interp, template2_interp)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def _analyze_frequency_relationship(self, pattern1: TemporalPattern, pattern2: TemporalPattern) -> float:
        """Analyze frequency relationships between patterns."""
        # Extract frequencies from pattern features
        freq1 = pattern1.features.get('frequency', 0)
        freq2 = pattern2.features.get('frequency', 0)
        
        if freq1 == 0 or freq2 == 0:
            return 0.0
            
        # Check for harmonic relationships
        ratio = max(freq1, freq2) / min(freq1, freq2)
        
        # Check if ratio is close to integer (harmonic relationship)
        nearest_integer = round(ratio)
        if abs(ratio - nearest_integer) < 0.1 and nearest_integer > 1:
            return 0.9  # Strong harmonic relationship
        elif abs(ratio - 1) < 0.1:
            return 0.8  # Same frequency
        else:
            return 0.0
    
    def update_real_time(self, new_data_point: float) -> List[TemporalPattern]:
        """Update real-time pattern detection with new data point."""
        self.real_time_buffer.append(new_data_point)
        
        if len(self.real_time_buffer) < self.min_pattern_length:
            return []
            
        # Convert buffer to array for analysis
        signal = np.array(list(self.real_time_buffer))
        
        # Detect patterns in recent data
        recent_patterns = self._detect_patterns_single_scale(signal, scale=1)
        
        # Update active patterns
        self._update_active_patterns(recent_patterns)
        
        return self.active_patterns
    
    def _update_active_patterns(self, new_patterns: List[TemporalPattern]):
        """Update list of active patterns."""
        current_time = len(self.real_time_buffer) - 1
        
        # Remove expired patterns
        self.active_patterns = [
            p for p in self.active_patterns
            if current_time - p.end_index < self.max_pattern_length
        ]
        
        # Add new patterns
        for pattern in new_patterns:
            # Check if pattern is still active
            if pattern.end_index >= current_time - 10:  # Pattern ended recently
                self.active_patterns.append(pattern)
                
    def get_pattern_summary(self, patterns: List[TemporalPattern]) -> Dict[str, Any]:
        """Generate summary statistics for detected patterns."""
        if not patterns:
            return {'total_patterns': 0}
            
        # Basic statistics
        summary = {
            'total_patterns': len(patterns),
            'pattern_types': defaultdict(int),
            'scales': defaultdict(int),
            'confidence_stats': {
                'mean': np.mean([p.confidence for p in patterns]),
                'min': np.min([p.confidence for p in patterns]),
                'max': np.max([p.confidence for p in patterns]),
                'std': np.std([p.confidence for p in patterns])
            }
        }
        
        # Count by type and scale
        for pattern in patterns:
            summary['pattern_types'][pattern.pattern_type] += 1
            summary['scales'][pattern.scale] += 1
            
        # Convert defaultdicts to regular dicts
        summary['pattern_types'] = dict(summary['pattern_types'])
        summary['scales'] = dict(summary['scales'])
        
        # Duration statistics
        durations = [p.end_index - p.start_index for p in patterns]
        if durations:
            summary['duration_stats'] = {
                'mean': np.mean(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'std': np.std(durations)
            }
            
        return summary