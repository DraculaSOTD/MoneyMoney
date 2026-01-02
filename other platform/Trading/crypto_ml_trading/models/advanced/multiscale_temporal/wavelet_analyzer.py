"""
Wavelet Analysis for Multi-Scale Time Series Decomposition.

Implements wavelet-based analysis for financial time series across multiple scales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class WaveletCoefficients:
    """Container for wavelet decomposition coefficients."""
    approximation: np.ndarray
    details: List[np.ndarray]
    scales: List[int]
    wavelet_type: str
    reconstruction_error: float


@dataclass
class WaveletFeatures:
    """Features extracted from wavelet analysis."""
    energy_by_scale: Dict[int, float]
    entropy_by_scale: Dict[int, float]
    variance_by_scale: Dict[int, float]
    dominant_frequencies: List[float]
    trend_strength: float
    noise_level: float
    seasonality_strength: Dict[int, float]


class WaveletAnalyzer:
    """
    Wavelet-based multi-scale analysis for financial time series.
    
    Features:
    - Discrete Wavelet Transform (DWT) implementation
    - Multiple wavelet types (Haar, Daubechies, etc.)
    - Multi-resolution decomposition
    - Denoising and trend extraction
    - Feature extraction across scales
    - Reconstruction and filtering
    """
    
    def __init__(self,
                 wavelet_type: str = 'db4',
                 max_levels: int = 6,
                 mode: str = 'symmetric',
                 threshold_method: str = 'soft'):
        """
        Initialize wavelet analyzer.
        
        Args:
            wavelet_type: Type of wavelet ('haar', 'db4', 'db8', 'coif2', etc.)
            max_levels: Maximum decomposition levels
            mode: Boundary condition mode
            threshold_method: Thresholding method for denoising
        """
        self.wavelet_type = wavelet_type
        self.max_levels = max_levels
        self.mode = mode
        self.threshold_method = threshold_method
        
        # Initialize wavelet coefficients
        self.wavelet_coeffs = self._get_wavelet_coefficients(wavelet_type)
        
    def _get_wavelet_coefficients(self, wavelet_type: str) -> Dict[str, np.ndarray]:
        """Get wavelet filter coefficients."""
        if wavelet_type == 'haar':
            return {
                'low_pass': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'high_pass': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
            }
        elif wavelet_type == 'db4':
            # Daubechies 4-tap wavelet
            sqrt3 = np.sqrt(3)
            sqrt2 = np.sqrt(2)
            
            h0 = (1 + sqrt3) / (4 * sqrt2)
            h1 = (3 + sqrt3) / (4 * sqrt2)
            h2 = (3 - sqrt3) / (4 * sqrt2)
            h3 = (1 - sqrt3) / (4 * sqrt2)
            
            return {
                'low_pass': np.array([h0, h1, h2, h3]),
                'high_pass': np.array([h3, -h2, h1, -h0])
            }
        elif wavelet_type == 'db8':
            # Daubechies 8-tap wavelet (simplified coefficients)
            return {
                'low_pass': np.array([
                    0.23037781, 0.71484657, 0.63088077, -0.02798376,
                    -0.18703481, 0.03084138, 0.03288301, -0.01059740
                ]),
                'high_pass': np.array([
                    -0.01059740, -0.03288301, 0.03084138, 0.18703481,
                    -0.02798376, -0.63088077, 0.71484657, -0.23037781
                ])
            }
        else:
            # Default to Haar
            return {
                'low_pass': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'high_pass': np.array([1/np.sqrt(2), -1/np.sqrt(2)])
            }
    
    def decompose(self, signal: np.ndarray, levels: Optional[int] = None) -> WaveletCoefficients:
        """
        Perform wavelet decomposition of the signal.
        
        Args:
            signal: Input signal
            levels: Number of decomposition levels (default: auto)
            
        Returns:
            Wavelet coefficients
        """
        signal = np.array(signal).flatten()
        
        if levels is None:
            levels = min(self.max_levels, int(np.log2(len(signal))))
            
        # Ensure signal length is appropriate
        min_length = 2 ** levels
        if len(signal) < min_length:
            # Pad signal
            pad_length = min_length - len(signal)
            signal = np.pad(signal, (0, pad_length), mode='symmetric')
            
        # Perform multi-level decomposition
        approximation = signal.copy()
        details = []
        scales = []
        
        for level in range(levels):
            # Decompose current approximation
            approx, detail = self._single_level_decompose(approximation)
            
            details.append(detail)
            scales.append(2 ** (level + 1))
            approximation = approx
            
        # Calculate reconstruction error
        reconstructed = self.reconstruct(WaveletCoefficients(
            approximation=approximation,
            details=details,
            scales=scales,
            wavelet_type=self.wavelet_type,
            reconstruction_error=0.0
        ))
        
        # Trim to original length
        original_length = len(signal) - (len(signal) - len(np.array(signal).flatten()))
        if len(reconstructed) > original_length:
            reconstructed = reconstructed[:original_length]
            
        reconstruction_error = np.mean((signal[:len(reconstructed)] - reconstructed) ** 2)
        
        return WaveletCoefficients(
            approximation=approximation,
            details=details,
            scales=scales,
            wavelet_type=self.wavelet_type,
            reconstruction_error=reconstruction_error
        )
    
    def _single_level_decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform single-level wavelet decomposition."""
        low_pass = self.wavelet_coeffs['low_pass']
        high_pass = self.wavelet_coeffs['high_pass']
        
        # Convolution with downsampling
        approx = self._convolve_downsample(signal, low_pass)
        detail = self._convolve_downsample(signal, high_pass)
        
        return approx, detail
    
    def _convolve_downsample(self, signal: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
        """Convolve signal with filter and downsample."""
        # Extend signal based on mode
        extended_signal = self._extend_signal(signal, len(filter_coeffs))
        
        # Convolution
        convolved = np.convolve(extended_signal, filter_coeffs[::-1], mode='valid')
        
        # Downsample by 2
        downsampled = convolved[::2]
        
        return downsampled
    
    def _extend_signal(self, signal: np.ndarray, filter_length: int) -> np.ndarray:
        """Extend signal according to boundary mode."""
        if self.mode == 'symmetric':
            # Symmetric extension
            left_ext = signal[1:filter_length][::-1]
            right_ext = signal[-filter_length:-1][::-1]
            return np.concatenate([left_ext, signal, right_ext])
        elif self.mode == 'periodization':
            # Periodic extension
            left_ext = signal[-filter_length+1:]
            right_ext = signal[:filter_length-1]
            return np.concatenate([left_ext, signal, right_ext])
        else:
            # Zero padding
            left_ext = np.zeros(filter_length - 1)
            right_ext = np.zeros(filter_length - 1)
            return np.concatenate([left_ext, signal, right_ext])
    
    def reconstruct(self, coeffs: WaveletCoefficients) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients.
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Reconstructed signal
        """
        # Start with approximation at deepest level
        reconstruction = coeffs.approximation.copy()
        
        # Reconstruct level by level
        for i in range(len(coeffs.details) - 1, -1, -1):
            detail = coeffs.details[i]
            reconstruction = self._single_level_reconstruct(reconstruction, detail)
            
        return reconstruction
    
    def _single_level_reconstruct(self, approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        """Perform single-level reconstruction."""
        low_pass = self.wavelet_coeffs['low_pass']
        high_pass = self.wavelet_coeffs['high_pass']
        
        # Upsample and convolve
        approx_upsampled = self._upsample_convolve(approx, low_pass)
        detail_upsampled = self._upsample_convolve(detail, high_pass)
        
        # Ensure same length
        min_length = min(len(approx_upsampled), len(detail_upsampled))
        
        return approx_upsampled[:min_length] + detail_upsampled[:min_length]
    
    def _upsample_convolve(self, signal: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
        """Upsample signal and convolve with filter."""
        # Upsample by inserting zeros
        upsampled = np.zeros(2 * len(signal))
        upsampled[::2] = signal
        
        # Convolve with reconstruction filter
        convolved = np.convolve(upsampled, filter_coeffs, mode='full')
        
        return convolved
    
    def denoise(self, signal: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            signal: Input signal
            threshold: Threshold value (auto-computed if None)
            
        Returns:
            Denoised signal
        """
        # Decompose signal
        coeffs = self.decompose(signal)
        
        # Estimate noise level from finest detail coefficients
        if threshold is None:
            finest_detail = coeffs.details[0]
            sigma = np.median(np.abs(finest_detail)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply thresholding to detail coefficients
        denoised_details = []
        for detail in coeffs.details:
            if self.threshold_method == 'soft':
                denoised_detail = self._soft_threshold(detail, threshold)
            else:
                denoised_detail = self._hard_threshold(detail, threshold)
            denoised_details.append(denoised_detail)
        
        # Reconstruct with denoised coefficients
        denoised_coeffs = WaveletCoefficients(
            approximation=coeffs.approximation,
            details=denoised_details,
            scales=coeffs.scales,
            wavelet_type=coeffs.wavelet_type,
            reconstruction_error=0.0
        )
        
        return self.reconstruct(denoised_coeffs)
    
    def _soft_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft thresholding."""
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def _hard_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """Apply hard thresholding."""
        return coeffs * (np.abs(coeffs) > threshold)
    
    def extract_features(self, signal: np.ndarray) -> WaveletFeatures:
        """
        Extract features from wavelet decomposition.
        
        Args:
            signal: Input signal
            
        Returns:
            Wavelet features
        """
        coeffs = self.decompose(signal)
        
        # Energy by scale
        energy_by_scale = {}
        for i, detail in enumerate(coeffs.details):
            scale = coeffs.scales[i]
            energy_by_scale[scale] = np.sum(detail ** 2)
        
        # Add approximation energy
        energy_by_scale[0] = np.sum(coeffs.approximation ** 2)
        
        # Entropy by scale
        entropy_by_scale = {}
        for i, detail in enumerate(coeffs.details):
            scale = coeffs.scales[i]
            entropy_by_scale[scale] = self._calculate_entropy(detail)
        entropy_by_scale[0] = self._calculate_entropy(coeffs.approximation)
        
        # Variance by scale
        variance_by_scale = {}
        for i, detail in enumerate(coeffs.details):
            scale = coeffs.scales[i]
            variance_by_scale[scale] = np.var(detail)
        variance_by_scale[0] = np.var(coeffs.approximation)
        
        # Dominant frequencies (inverse of scales with highest energy)
        total_energy = sum(energy_by_scale.values())
        dominant_frequencies = []
        for scale, energy in energy_by_scale.items():
            if scale > 0 and energy / total_energy > 0.1:  # Significant energy
                frequency = 1.0 / scale if scale > 0 else 0
                dominant_frequencies.append(frequency)
        dominant_frequencies.sort(reverse=True)
        
        # Trend strength (energy in approximation vs total)
        trend_strength = energy_by_scale[0] / total_energy
        
        # Noise level (energy in finest detail)
        finest_scale = min([s for s in coeffs.scales if s > 0])
        noise_level = energy_by_scale[finest_scale] / total_energy
        
        # Seasonality strength by scale
        seasonality_strength = {}
        for scale, energy in energy_by_scale.items():
            if scale > 0:
                # Measure periodicity in the detail coefficients
                detail_idx = coeffs.scales.index(scale)
                detail = coeffs.details[detail_idx]
                seasonality_strength[scale] = self._measure_periodicity(detail)
        
        return WaveletFeatures(
            energy_by_scale=energy_by_scale,
            entropy_by_scale=entropy_by_scale,
            variance_by_scale=variance_by_scale,
            dominant_frequencies=dominant_frequencies,
            trend_strength=trend_strength,
            noise_level=noise_level,
            seasonality_strength=seasonality_strength
        )
    
    def _calculate_entropy(self, coeffs: np.ndarray) -> float:
        """Calculate Shannon entropy of coefficients."""
        # Normalize coefficients to create probability distribution
        abs_coeffs = np.abs(coeffs)
        if np.sum(abs_coeffs) == 0:
            return 0.0
            
        probs = abs_coeffs / np.sum(abs_coeffs)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        
        return entropy
    
    def _measure_periodicity(self, signal: np.ndarray) -> float:
        """Measure periodicity in signal using autocorrelation."""
        if len(signal) < 4:
            return 0.0
            
        # Autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # Find peaks in autocorrelation (excluding lag 0)
        if len(autocorr) > 1:
            peaks = []
            for i in range(1, min(len(autocorr), len(signal)//2)):
                if (i == 1 or autocorr[i] > autocorr[i-1]) and \
                   (i == len(autocorr)-1 or autocorr[i] > autocorr[i+1]):
                    peaks.append(autocorr[i])
            
            # Periodicity strength is maximum peak value
            return max(peaks) if peaks else 0.0
        
        return 0.0
    
    def get_scale_components(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get signal components at different scales.
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary with components at different scales
        """
        coeffs = self.decompose(signal)
        components = {}
        
        # Add trend component (approximation)
        components['trend'] = self.reconstruct(WaveletCoefficients(
            approximation=coeffs.approximation,
            details=[np.zeros_like(d) for d in coeffs.details],
            scales=coeffs.scales,
            wavelet_type=coeffs.wavelet_type,
            reconstruction_error=0.0
        ))
        
        # Add detail components at each scale
        for i, scale in enumerate(coeffs.scales):
            detail_coeffs = [np.zeros_like(d) for d in coeffs.details]
            detail_coeffs[i] = coeffs.details[i]
            
            component = self.reconstruct(WaveletCoefficients(
                approximation=np.zeros_like(coeffs.approximation),
                details=detail_coeffs,
                scales=coeffs.scales,
                wavelet_type=coeffs.wavelet_type,
                reconstruction_error=0.0
            ))
            
            components[f'detail_scale_{scale}'] = component
        
        return components
    
    def detect_regime_changes(self, signal: np.ndarray, window_size: int = 50) -> List[int]:
        """
        Detect regime changes using wavelet analysis.
        
        Args:
            signal: Input signal
            window_size: Window size for change detection
            
        Returns:
            List of change point indices
        """
        # Decompose signal
        coeffs = self.decompose(signal)
        
        # Use finest detail coefficients for change detection
        finest_detail = coeffs.details[0]
        
        # Calculate local variance in sliding window
        change_points = []
        
        for i in range(window_size, len(finest_detail) - window_size):
            # Variance before and after
            var_before = np.var(finest_detail[i-window_size:i])
            var_after = np.var(finest_detail[i:i+window_size])
            
            # Change score
            if var_before + var_after > 0:
                change_score = abs(var_before - var_after) / (var_before + var_after)
                
                # Threshold for change detection
                if change_score > 0.5:  # Configurable threshold
                    # Map back to original signal indices
                    original_idx = i * (len(signal) // len(finest_detail))
                    change_points.append(original_idx)
        
        return change_points
    
    def analyze_volatility_clustering(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Analyze volatility clustering using multi-scale wavelet analysis.
        
        Args:
            returns: Return series
            
        Returns:
            Volatility clustering metrics
        """
        # Calculate absolute returns (proxy for volatility)
        abs_returns = np.abs(returns)
        
        # Decompose absolute returns
        coeffs = self.decompose(abs_returns)
        
        # Analyze persistence at each scale
        persistence_by_scale = {}
        
        for i, scale in enumerate(coeffs.scales):
            detail = coeffs.details[i]
            
            # Calculate autocorrelation at lag 1
            if len(detail) > 1:
                correlation = np.corrcoef(detail[:-1], detail[1:])[0, 1]
                persistence_by_scale[scale] = correlation if not np.isnan(correlation) else 0
            else:
                persistence_by_scale[scale] = 0
        
        # Overall clustering measure
        clustering_strength = np.mean(list(persistence_by_scale.values()))
        
        # Multi-scale Hurst exponent estimation
        hurst_exponent = self._estimate_multiscale_hurst(abs_returns, coeffs)
        
        return {
            'clustering_strength': clustering_strength,
            'persistence_by_scale': persistence_by_scale,
            'hurst_exponent': hurst_exponent,
            'long_memory_indicator': hurst_exponent > 0.5
        }
    
    def _estimate_multiscale_hurst(self, signal: np.ndarray, coeffs: WaveletCoefficients) -> float:
        """Estimate Hurst exponent using wavelet-based method."""
        # Calculate variance at each scale
        scales = []
        variances = []
        
        for i, scale in enumerate(coeffs.scales):
            detail = coeffs.details[i]
            if len(detail) > 0:
                scales.append(np.log2(scale))
                variances.append(np.log2(np.var(detail) + 1e-12))
        
        if len(scales) < 2:
            return 0.5  # Default value
        
        # Linear regression of log(variance) vs log(scale)
        # Slope is related to Hurst exponent: H = (slope + 1) / 2
        A = np.vstack([scales, np.ones(len(scales))]).T
        slope, _ = np.linalg.lstsq(A, variances, rcond=None)[0]
        
        hurst = (slope + 1) / 2
        
        # Clamp to valid range
        return max(0, min(1, hurst))
    
    def adaptive_decomposition(self, signal: np.ndarray, 
                             target_scales: List[int]) -> WaveletCoefficients:
        """
        Perform adaptive wavelet decomposition targeting specific scales.
        
        Args:
            signal: Input signal
            target_scales: Desired scales for decomposition
            
        Returns:
            Wavelet coefficients optimized for target scales
        """
        # Determine optimal decomposition level
        max_target_scale = max(target_scales)
        levels = int(np.log2(max_target_scale)) + 1
        
        # Perform standard decomposition
        coeffs = self.decompose(signal, levels)
        
        # Filter coefficients to keep only target scales
        filtered_details = []
        filtered_scales = []
        
        for i, scale in enumerate(coeffs.scales):
            if scale in target_scales:
                filtered_details.append(coeffs.details[i])
                filtered_scales.append(scale)
            else:
                # Zero out non-target scales
                filtered_details.append(np.zeros_like(coeffs.details[i]))
                filtered_scales.append(scale)
        
        return WaveletCoefficients(
            approximation=coeffs.approximation,
            details=filtered_details,
            scales=filtered_scales,
            wavelet_type=coeffs.wavelet_type,
            reconstruction_error=0.0
        )