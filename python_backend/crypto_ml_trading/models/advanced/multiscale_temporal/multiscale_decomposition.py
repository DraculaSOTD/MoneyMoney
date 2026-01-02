"""
Multi-Scale Decomposition for Financial Time Series.

Implements various multi-scale decomposition techniques beyond wavelets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class EMDComponents:
    """Empirical Mode Decomposition results."""
    imfs: List[np.ndarray]  # Intrinsic Mode Functions
    residue: np.ndarray
    frequencies: List[float]
    energy_ratios: List[float]


@dataclass
class VMDComponents:
    """Variational Mode Decomposition results."""
    modes: List[np.ndarray]
    center_frequencies: List[float]
    bandwidths: List[float]
    reconstruction_error: float


@dataclass
class MultiScaleFeatures:
    """Features extracted from multi-scale decomposition."""
    trend_strength: float
    cycle_components: Dict[str, float]
    noise_level: float
    complexity_measure: float
    dominant_periods: List[int]
    scale_entropy: Dict[int, float]
    cross_scale_correlations: np.ndarray


class MultiScaleDecomposition:
    """
    Multi-scale decomposition using various techniques.
    
    Features:
    - Empirical Mode Decomposition (EMD)
    - Variational Mode Decomposition (VMD)
    - Singular Spectrum Analysis (SSA)
    - Multi-resolution filtering
    - Seasonal decomposition
    - Trend-cycle-noise separation
    """
    
    def __init__(self,
                 max_imfs: int = 10,
                 sifting_threshold: float = 0.05,
                 max_sifting_iterations: int = 100):
        """
        Initialize multi-scale decomposition.
        
        Args:
            max_imfs: Maximum number of IMFs for EMD
            sifting_threshold: Convergence threshold for sifting
            max_sifting_iterations: Maximum sifting iterations
        """
        self.max_imfs = max_imfs
        self.sifting_threshold = sifting_threshold
        self.max_sifting_iterations = max_sifting_iterations
        
    def empirical_mode_decomposition(self, signal: np.ndarray) -> EMDComponents:
        """
        Perform Empirical Mode Decomposition (EMD).
        
        Args:
            signal: Input signal
            
        Returns:
            EMD components
        """
        signal = np.array(signal).flatten()
        imfs = []
        residue = signal.copy()
        
        for imf_idx in range(self.max_imfs):
            imf = self._extract_imf(residue)
            
            if imf is None:
                break
                
            imfs.append(imf)
            residue = residue - imf
            
            # Check if residue is monotonic (trend)
            if self._is_monotonic(residue):
                break
                
        # Calculate frequencies and energy ratios
        frequencies = []
        energy_ratios = []
        total_energy = np.sum(signal ** 2)
        
        for imf in imfs:
            freq = self._estimate_frequency(imf)
            energy = np.sum(imf ** 2) / total_energy
            frequencies.append(freq)
            energy_ratios.append(energy)
            
        return EMDComponents(
            imfs=imfs,
            residue=residue,
            frequencies=frequencies,
            energy_ratios=energy_ratios
        )
    
    def _extract_imf(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single Intrinsic Mode Function."""
        if len(signal) < 4:
            return None
            
        h = signal.copy()
        
        for iteration in range(self.max_sifting_iterations):
            # Find local maxima and minima
            maxima_indices, maxima_values = self._find_extrema(h, 'max')
            minima_indices, minima_values = self._find_extrema(h, 'min')
            
            if len(maxima_indices) < 2 or len(minima_indices) < 2:
                break
                
            # Create envelopes
            upper_envelope = self._interpolate_envelope(
                maxima_indices, maxima_values, len(h)
            )
            lower_envelope = self._interpolate_envelope(
                minima_indices, minima_values, len(h)
            )
            
            # Calculate mean envelope
            mean_envelope = (upper_envelope + lower_envelope) / 2
            
            # Update h
            h_new = h - mean_envelope
            
            # Check stopping criterion
            if self._check_imf_criterion(h, h_new):
                h = h_new
                break
                
            h = h_new
            
        return h if not self._is_monotonic(h) else None
    
    def _find_extrema(self, signal: np.ndarray, extrema_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Find local extrema in signal."""
        indices = []
        values = []
        
        for i in range(1, len(signal) - 1):
            if extrema_type == 'max':
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                    indices.append(i)
                    values.append(signal[i])
            else:  # min
                if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                    indices.append(i)
                    values.append(signal[i])
                    
        return np.array(indices), np.array(values)
    
    def _interpolate_envelope(self, indices: np.ndarray, values: np.ndarray, 
                            signal_length: int) -> np.ndarray:
        """Interpolate envelope from extrema points."""
        if len(indices) < 2:
            return np.zeros(signal_length)
            
        # Add boundary points
        extended_indices = np.concatenate([[0], indices, [signal_length - 1]])
        extended_values = np.concatenate([[values[0]], values, [values[-1]]])
        
        # Linear interpolation (could use spline for better results)
        envelope = np.interp(np.arange(signal_length), extended_indices, extended_values)
        
        return envelope
    
    def _check_imf_criterion(self, h_old: np.ndarray, h_new: np.ndarray) -> bool:
        """Check if IMF extraction should stop."""
        if len(h_old) == 0:
            return True
            
        # Standard deviation criterion
        numerator = np.sum((h_old - h_new) ** 2)
        denominator = np.sum(h_old ** 2)
        
        if denominator == 0:
            return True
            
        sd = numerator / denominator
        
        return sd < self.sifting_threshold
    
    def _is_monotonic(self, signal: np.ndarray) -> bool:
        """Check if signal is monotonic."""
        if len(signal) < 2:
            return True
            
        diff = np.diff(signal)
        
        # Check if all differences have the same sign
        return np.all(diff >= 0) or np.all(diff <= 0)
    
    def _estimate_frequency(self, imf: np.ndarray) -> float:
        """Estimate dominant frequency of an IMF."""
        if len(imf) < 4:
            return 0.0
            
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        
        # Frequency is approximately zero_crossings / (2 * length)
        frequency = zero_crossings / (2 * len(imf))
        
        return frequency
    
    def variational_mode_decomposition(self, signal: np.ndarray, 
                                     num_modes: int = 4,
                                     alpha: float = 2000) -> VMDComponents:
        """
        Perform Variational Mode Decomposition (VMD).
        
        This is a simplified implementation of VMD.
        
        Args:
            signal: Input signal
            num_modes: Number of modes to extract
            alpha: Balancing parameter
            
        Returns:
            VMD components
        """
        signal = np.array(signal).flatten()
        n = len(signal)
        
        # Initialize modes and their center frequencies
        modes = []
        center_frequencies = []
        bandwidths = []
        
        # Simple implementation: use bandpass filtering at different frequencies
        frequency_bands = np.linspace(0.01, 0.5, num_modes)
        
        for i, center_freq in enumerate(frequency_bands):
            # Create bandpass filter
            bandwidth = 0.5 / num_modes
            mode = self._bandpass_filter(signal, center_freq, bandwidth)
            
            modes.append(mode)
            center_frequencies.append(center_freq)
            bandwidths.append(bandwidth)
            
        # Calculate reconstruction error
        reconstructed = np.sum(modes, axis=0)
        reconstruction_error = np.mean((signal - reconstructed) ** 2)
        
        return VMDComponents(
            modes=modes,
            center_frequencies=center_frequencies,
            bandwidths=bandwidths,
            reconstruction_error=reconstruction_error
        )
    
    def _bandpass_filter(self, signal: np.ndarray, center_freq: float, 
                        bandwidth: float) -> np.ndarray:
        """Simple bandpass filter implementation."""
        n = len(signal)
        
        # Create frequency domain representation
        fft_signal = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(n)
        
        # Create bandpass mask
        low_freq = center_freq - bandwidth / 2
        high_freq = center_freq + bandwidth / 2
        
        mask = np.zeros(n)
        mask[(np.abs(frequencies) >= low_freq) & (np.abs(frequencies) <= high_freq)] = 1
        
        # Apply filter
        filtered_fft = fft_signal * mask
        
        # Convert back to time domain
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_signal
    
    def singular_spectrum_analysis(self, signal: np.ndarray, 
                                  window_length: Optional[int] = None,
                                  num_components: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Perform Singular Spectrum Analysis (SSA).
        
        Args:
            signal: Input signal
            window_length: Length of embedding window
            num_components: Number of components to extract
            
        Returns:
            SSA components
        """
        signal = np.array(signal).flatten()
        n = len(signal)
        
        if window_length is None:
            window_length = min(n // 2, 50)
            
        if num_components is None:
            num_components = min(10, window_length)
            
        # Step 1: Embedding
        trajectory_matrix = self._create_trajectory_matrix(signal, window_length)
        
        # Step 2: SVD
        U, sigma, Vt = np.linalg.svd(trajectory_matrix, full_matrices=False)
        
        # Step 3: Grouping and reconstruction
        components = {}
        
        for i in range(min(num_components, len(sigma))):
            # Reconstruct component
            component_matrix = sigma[i] * np.outer(U[:, i], Vt[i, :])
            component_series = self._diagonal_averaging(component_matrix)
            
            components[f'component_{i+1}'] = component_series
            
        # Add residual
        total_reconstructed = np.sum(list(components.values()), axis=0)
        if len(total_reconstructed) == len(signal):
            components['residual'] = signal - total_reconstructed
        else:
            # Handle length mismatch
            min_length = min(len(signal), len(total_reconstructed))
            components['residual'] = signal[:min_length] - total_reconstructed[:min_length]
            
        return components
    
    def _create_trajectory_matrix(self, signal: np.ndarray, window_length: int) -> np.ndarray:
        """Create trajectory matrix for SSA."""
        n = len(signal)
        k = n - window_length + 1
        
        trajectory_matrix = np.zeros((window_length, k))
        
        for i in range(k):
            trajectory_matrix[:, i] = signal[i:i + window_length]
            
        return trajectory_matrix
    
    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """Perform diagonal averaging to convert matrix back to time series."""
        m, n = matrix.shape
        series_length = m + n - 1
        series = np.zeros(series_length)
        
        for k in range(series_length):
            if k < m:
                # Upper triangle
                diag_sum = np.sum([matrix[k-j, j] for j in range(max(0, k-m+1), min(k+1, n))])
                diag_count = min(k+1, n) - max(0, k-m+1)
            else:
                # Lower triangle
                diag_sum = np.sum([matrix[k-j, j] for j in range(k-m+1, min(k+1, n))])
                diag_count = min(k+1, n) - (k-m+1)
                
            if diag_count > 0:
                series[k] = diag_sum / diag_count
                
        return series
    
    def seasonal_decomposition(self, signal: np.ndarray, 
                             period: int,
                             decomposition_type: str = 'additive') -> Dict[str, np.ndarray]:
        """
        Perform seasonal decomposition.
        
        Args:
            signal: Input signal
            period: Seasonal period
            decomposition_type: 'additive' or 'multiplicative'
            
        Returns:
            Decomposition components
        """
        signal = np.array(signal).flatten()
        n = len(signal)
        
        # Calculate trend using moving average
        trend = self._calculate_trend(signal, period)
        
        # Calculate seasonal component
        if decomposition_type == 'additive':
            detrended = signal - trend
        else:  # multiplicative
            detrended = signal / np.where(trend != 0, trend, 1)
            
        seasonal = self._calculate_seasonal(detrended, period)
        
        # Calculate residual
        if decomposition_type == 'additive':
            residual = signal - trend - seasonal
        else:
            residual = signal / (trend * np.where(seasonal != 0, seasonal, 1))
            
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'original': signal
        }
    
    def _calculate_trend(self, signal: np.ndarray, period: int) -> np.ndarray:
        """Calculate trend component using centered moving average."""
        n = len(signal)
        trend = np.full(n, np.nan)
        
        # Use centered moving average
        half_period = period // 2
        
        for i in range(half_period, n - half_period):
            if period % 2 == 1:
                # Odd period
                trend[i] = np.mean(signal[i - half_period:i + half_period + 1])
            else:
                # Even period - use weighted average
                left_sum = np.sum(signal[i - half_period:i + half_period])
                weight_sum = 2 * half_period
                
                if i - half_period > 0:
                    left_sum += 0.5 * signal[i - half_period - 1]
                    weight_sum += 0.5
                    
                if i + half_period < n:
                    left_sum += 0.5 * signal[i + half_period]
                    weight_sum += 0.5
                    
                trend[i] = left_sum / weight_sum
                
        # Fill in edges with linear interpolation
        valid_indices = ~np.isnan(trend)
        if np.any(valid_indices):
            trend[~valid_indices] = np.interp(
                np.where(~valid_indices)[0],
                np.where(valid_indices)[0],
                trend[valid_indices]
            )
            
        return trend
    
    def _calculate_seasonal(self, detrended: np.ndarray, period: int) -> np.ndarray:
        """Calculate seasonal component."""
        n = len(detrended)
        seasonal = np.zeros(n)
        
        # Calculate average for each position in the cycle
        for i in range(period):
            positions = np.arange(i, n, period)
            seasonal_value = np.nanmean(detrended[positions])
            seasonal[positions] = seasonal_value
            
        return seasonal
    
    def extract_multiscale_features(self, signal: np.ndarray) -> MultiScaleFeatures:
        """
        Extract comprehensive multi-scale features.
        
        Args:
            signal: Input signal
            
        Returns:
            Multi-scale features
        """
        # EMD decomposition
        emd_components = self.empirical_mode_decomposition(signal)
        
        # Seasonal decomposition (assume daily period for crypto)
        period = min(24, len(signal) // 4)  # Daily period or quarter of signal
        if period > 2:
            seasonal_components = self.seasonal_decomposition(signal, period)
        else:
            seasonal_components = {'trend': signal, 'seasonal': np.zeros_like(signal)}
            
        # SSA decomposition
        ssa_components = self.singular_spectrum_analysis(signal)
        
        # Calculate features
        
        # 1. Trend strength
        trend_component = seasonal_components['trend']
        signal_var = np.var(signal)
        trend_var = np.var(trend_component)
        trend_strength = trend_var / signal_var if signal_var > 0 else 0
        
        # 2. Cycle components (from EMD)
        cycle_components = {}
        for i, (imf, freq) in enumerate(zip(emd_components.imfs, emd_components.frequencies)):
            if 0.01 < freq < 0.5:  # Filter relevant frequencies
                cycle_components[f'cycle_{i+1}'] = np.var(imf) / signal_var if signal_var > 0 else 0
                
        # 3. Noise level (high-frequency IMFs)
        noise_imfs = [imf for imf, freq in zip(emd_components.imfs, emd_components.frequencies) if freq > 0.3]
        noise_level = sum(np.var(imf) for imf in noise_imfs) / signal_var if signal_var > 0 else 0
        
        # 4. Complexity measure (number of significant IMFs)
        significant_imfs = sum(1 for energy in emd_components.energy_ratios if energy > 0.05)
        complexity_measure = significant_imfs / len(emd_components.imfs) if emd_components.imfs else 0
        
        # 5. Dominant periods
        dominant_periods = []
        for freq in emd_components.frequencies:
            if freq > 0:
                period = int(1 / freq)
                if 2 <= period <= len(signal) // 4:
                    dominant_periods.append(period)
        dominant_periods = sorted(list(set(dominant_periods)))
        
        # 6. Scale entropy
        scale_entropy = {}
        for i, imf in enumerate(emd_components.imfs):
            entropy = self._calculate_sample_entropy(imf)
            scale_entropy[i+1] = entropy
            
        # 7. Cross-scale correlations
        cross_correlations = self._calculate_cross_scale_correlations(emd_components.imfs)
        
        return MultiScaleFeatures(
            trend_strength=trend_strength,
            cycle_components=cycle_components,
            noise_level=noise_level,
            complexity_measure=complexity_measure,
            dominant_periods=dominant_periods,
            scale_entropy=scale_entropy,
            cross_scale_correlations=cross_correlations
        )
    
    def _calculate_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy of a signal."""
        if len(signal) < m + 1:
            return 0.0
            
        n = len(signal)
        
        # Calculate standard deviation
        std_signal = np.std(signal)
        if std_signal == 0:
            return 0.0
            
        tolerance = r * std_signal
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(n - m + 1)])
            c = np.zeros(n - m + 1)
            
            for i in range(n - m + 1):
                template = patterns[i]
                for j in range(n - m + 1):
                    if _maxdist(template, patterns[j], m) <= tolerance:
                        c[i] += 1
                        
            phi = np.mean(np.log(c / (n - m + 1)))
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _calculate_cross_scale_correlations(self, imfs: List[np.ndarray]) -> np.ndarray:
        """Calculate correlations between different scales."""
        n_imfs = len(imfs)
        if n_imfs < 2:
            return np.array([[1.0]])
            
        correlations = np.eye(n_imfs)
        
        for i in range(n_imfs):
            for j in range(i + 1, n_imfs):
                # Ensure same length
                min_length = min(len(imfs[i]), len(imfs[j]))
                if min_length > 1:
                    corr = np.corrcoef(imfs[i][:min_length], imfs[j][:min_length])[0, 1]
                    if not np.isnan(corr):
                        correlations[i, j] = correlations[j, i] = corr
                        
        return correlations
    
    def reconstruct_filtered_signal(self, components: Union[EMDComponents, Dict[str, np.ndarray]],
                                   filter_specs: Dict[str, bool]) -> np.ndarray:
        """
        Reconstruct signal with filtered components.
        
        Args:
            components: Decomposition components
            filter_specs: Which components to include
            
        Returns:
            Filtered signal
        """
        if isinstance(components, EMDComponents):
            # EMD reconstruction
            reconstructed = np.zeros_like(components.residue)
            
            if filter_specs.get('residue', True):
                reconstructed += components.residue
                
            for i, imf in enumerate(components.imfs):
                if filter_specs.get(f'imf_{i+1}', True):
                    reconstructed += imf
                    
        else:
            # Generic component reconstruction
            reconstructed = None
            
            for component_name, component_signal in components.items():
                if filter_specs.get(component_name, True):
                    if reconstructed is None:
                        reconstructed = component_signal.copy()
                    else:
                        # Ensure same length
                        min_length = min(len(reconstructed), len(component_signal))
                        reconstructed = reconstructed[:min_length] + component_signal[:min_length]
                        
            if reconstructed is None:
                reconstructed = np.zeros(1)
                
        return reconstructed