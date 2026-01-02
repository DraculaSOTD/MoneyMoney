"""
Statistical Significance Testing for Model Validation.

Implements various statistical tests to determine if model performance
differences are statistically significant.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, nemenyi
import warnings
from dataclasses import dataclass
from itertools import combinations


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class StatisticalSignificanceTester:
    """
    Performs statistical significance tests for model comparison.
    
    Features:
    - Paired t-tests
    - Wilcoxon signed-rank tests
    - Diebold-Mariano test for forecast comparison
    - Multiple comparison corrections
    - Bootstrap confidence intervals
    - Effect size calculations
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize significance tester.
        
        Args:
            significance_level: Significance level for tests (alpha)
        """
        self.significance_level = significance_level
        
    def compare_two_models(self,
                          y_true: np.ndarray,
                          predictions_1: np.ndarray,
                          predictions_2: np.ndarray,
                          test_type: str = 'auto') -> Dict[str, TestResult]:
        """
        Compare predictions from two models.
        
        Args:
            y_true: True values
            predictions_1: Predictions from model 1
            predictions_2: Predictions from model 2
            test_type: Type of test ('auto', 'parametric', 'non-parametric')
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Calculate errors
        errors_1 = y_true - predictions_1
        errors_2 = y_true - predictions_2
        
        # Paired difference
        diff = errors_1 - errors_2
        
        # Choose test type
        if test_type == 'auto':
            # Check normality
            _, p_normal = stats.normaltest(diff)
            test_type = 'parametric' if p_normal > 0.05 else 'non-parametric'
        
        # Parametric tests
        if test_type == 'parametric':
            results['paired_t_test'] = self._paired_t_test(errors_1, errors_2)
            results['paired_t_test_corrected'] = self._paired_t_test_corrected(errors_1, errors_2)
        
        # Non-parametric tests
        results['wilcoxon_test'] = self._wilcoxon_test(errors_1, errors_2)
        
        # Diebold-Mariano test
        results['diebold_mariano'] = self._diebold_mariano_test(errors_1, errors_2)
        
        # Effect size
        results['effect_size'] = self._calculate_effect_size(errors_1, errors_2)
        
        # Bootstrap confidence interval
        results['bootstrap_ci'] = self._bootstrap_confidence_interval(errors_1, errors_2)
        
        return results
    
    def compare_multiple_models(self,
                              y_true: np.ndarray,
                              predictions_dict: Dict[str, np.ndarray],
                              test_type: str = 'friedman') -> Dict[str, Any]:
        """
        Compare predictions from multiple models.
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of model predictions
            test_type: Type of test ('friedman', 'anova')
            
        Returns:
            Dictionary of test results
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        if n_models < 3:
            raise ValueError("Use compare_two_models for fewer than 3 models")
        
        results = {
            'omnibus_test': None,
            'pairwise_comparisons': {},
            'rankings': None
        }
        
        # Calculate errors for each model
        errors_dict = {
            name: y_true - preds 
            for name, preds in predictions_dict.items()
        }
        
        # Omnibus test
        if test_type == 'friedman':
            results['omnibus_test'] = self._friedman_test(errors_dict)
            
            # If significant, perform post-hoc tests
            if results['omnibus_test'].significant:
                results['post_hoc'] = self._nemenyi_test(errors_dict)
        elif test_type == 'anova':
            results['omnibus_test'] = self._repeated_measures_anova(errors_dict)
        
        # Pairwise comparisons with correction
        results['pairwise_comparisons'] = self._all_pairwise_comparisons(
            y_true, predictions_dict
        )
        
        # Model rankings
        results['rankings'] = self._rank_models(errors_dict)
        
        return results
    
    def _paired_t_test(self, errors_1: np.ndarray, errors_2: np.ndarray) -> TestResult:
        """Perform paired t-test."""
        diff = errors_1 - errors_2
        
        # Remove NaN values
        diff = diff[~np.isnan(diff)]
        
        if len(diff) < 2:
            return TestResult(
                test_name="Paired t-test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                interpretation="Insufficient data for test"
            )
        
        t_stat, p_value = stats.ttest_rel(errors_1, errors_2)
        
        # Effect size (Cohen's d)
        effect_size = np.mean(diff) / np.std(diff, ddof=1)
        
        # Confidence interval
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci = stats.t.interval(
            1 - self.significance_level,
            len(diff) - 1,
            loc=mean_diff,
            scale=se_diff
        )
        
        return TestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=self._interpret_effect_size(effect_size)
        )
    
    def _paired_t_test_corrected(self, errors_1: np.ndarray, 
                                errors_2: np.ndarray) -> TestResult:
        """Perform paired t-test with Nadeau-Bengio correction."""
        diff = errors_1 - errors_2
        n = len(diff)
        
        # Training/test split ratio (assuming 80/20)
        rho = 0.2
        
        # Corrected variance
        var_corrected = np.var(diff, ddof=1) * (1/n + rho/(1-rho))
        
        # Corrected t-statistic
        t_stat_corrected = np.mean(diff) / np.sqrt(var_corrected)
        
        # Degrees of freedom
        df = n - 1
        
        # P-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat_corrected), df))
        
        return TestResult(
            test_name="Paired t-test (Nadeau-Bengio corrected)",
            statistic=t_stat_corrected,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation="Accounts for training/test dependence"
        )
    
    def _wilcoxon_test(self, errors_1: np.ndarray, errors_2: np.ndarray) -> TestResult:
        """Perform Wilcoxon signed-rank test."""
        try:
            statistic, p_value = wilcoxon(errors_1, errors_2)
            
            # Effect size (r = Z / sqrt(N))
            n = len(errors_1)
            z_score = stats.norm.ppf(1 - p_value/2)
            effect_size = z_score / np.sqrt(n)
            
            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                effect_size=effect_size,
                interpretation=f"Non-parametric test, r={effect_size:.3f}"
            )
        except:
            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                interpretation="Test failed"
            )
    
    def _diebold_mariano_test(self, errors_1: np.ndarray, 
                             errors_2: np.ndarray,
                             horizon: int = 1) -> TestResult:
        """
        Diebold-Mariano test for predictive accuracy.
        
        Tests if the forecast accuracy of two models is significantly different.
        """
        # Loss differential
        d = errors_1**2 - errors_2**2  # Using squared loss
        
        # Mean
        mean_d = np.mean(d)
        
        # Long-run variance estimation
        T = len(d)
        gamma = []
        
        for k in range(horizon):
            if k < T:
                gamma_k = np.sum((d[:-k-1] - mean_d) * (d[k+1:] - mean_d)) / T
                gamma.append(gamma_k)
        
        # Variance
        var_d = gamma[0] + 2 * sum(gamma[1:])
        
        # DM statistic
        dm_stat = mean_d / np.sqrt(var_d / T)
        
        # P-value (two-sided)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return TestResult(
            test_name="Diebold-Mariano test",
            statistic=dm_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation="Tests equal predictive accuracy"
        )
    
    def _calculate_effect_size(self, errors_1: np.ndarray, 
                             errors_2: np.ndarray) -> TestResult:
        """Calculate various effect size measures."""
        diff = errors_1 - errors_2
        
        # Cohen's d
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # Hedge's g (corrected Cohen's d)
        n = len(diff)
        correction = 1 - 3 / (4 * n - 9)
        hedges_g = cohens_d * correction
        
        # Glass's delta (using first model's SD)
        glass_delta = np.mean(diff) / np.std(errors_1, ddof=1)
        
        # Probability of superiority
        prob_superiority = np.mean(errors_1 < errors_2)
        
        interpretation = self._interpret_effect_size(cohens_d)
        
        return TestResult(
            test_name="Effect sizes",
            statistic=cohens_d,
            p_value=np.nan,
            significant=abs(cohens_d) > 0.2,  # Small effect threshold
            effect_size=cohens_d,
            interpretation=f"{interpretation}; Hedge's g={hedges_g:.3f}, P(Model1<Model2)={prob_superiority:.3f}"
        )
    
    def _bootstrap_confidence_interval(self, errors_1: np.ndarray,
                                     errors_2: np.ndarray,
                                     n_bootstrap: int = 1000) -> TestResult:
        """Calculate bootstrap confidence interval for difference."""
        diff = errors_1 - errors_2
        n = len(diff)
        
        # Bootstrap
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = diff[indices]
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Confidence interval
        alpha = self.significance_level
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        # Check if 0 is in the interval
        significant = not (lower <= 0 <= upper)
        
        return TestResult(
            test_name="Bootstrap confidence interval",
            statistic=np.mean(diff),
            p_value=np.nan,
            significant=significant,
            confidence_interval=(lower, upper),
            interpretation=f"95% CI: [{lower:.4f}, {upper:.4f}]"
        )
    
    def _friedman_test(self, errors_dict: Dict[str, np.ndarray]) -> TestResult:
        """Friedman test for multiple models."""
        # Create matrix of errors
        error_matrix = np.column_stack(list(errors_dict.values()))
        
        try:
            statistic, p_value = friedmanchisquare(*error_matrix.T)
            
            return TestResult(
                test_name="Friedman test",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.significance_level,
                interpretation="Non-parametric test for multiple models"
            )
        except:
            return TestResult(
                test_name="Friedman test",
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                interpretation="Test failed"
            )
    
    def _nemenyi_test(self, errors_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Nemenyi post-hoc test after Friedman."""
        model_names = list(errors_dict.keys())
        n_models = len(model_names)
        n_samples = len(next(iter(errors_dict.values())))
        
        # Rank errors for each sample
        error_matrix = np.column_stack(list(errors_dict.values()))
        ranks = np.array([stats.rankdata(row) for row in error_matrix])
        mean_ranks = np.mean(ranks, axis=0)
        
        # Critical difference
        q_alpha = 2.569  # For alpha=0.05, k=5 models (approximate)
        cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_samples))
        
        # Pairwise comparisons
        results = {
            'mean_ranks': dict(zip(model_names, mean_ranks)),
            'critical_difference': cd,
            'significant_pairs': []
        }
        
        for i, j in combinations(range(n_models), 2):
            if abs(mean_ranks[i] - mean_ranks[j]) > cd:
                results['significant_pairs'].append((model_names[i], model_names[j]))
        
        return results
    
    def _repeated_measures_anova(self, errors_dict: Dict[str, np.ndarray]) -> TestResult:
        """Repeated measures ANOVA for multiple models."""
        # Simplified version - would need proper implementation
        error_matrix = np.column_stack(list(errors_dict.values()))
        
        # One-way ANOVA as approximation
        f_stat, p_value = stats.f_oneway(*error_matrix.T)
        
        return TestResult(
            test_name="Repeated measures ANOVA",
            statistic=f_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation="Parametric test for multiple models"
        )
    
    def _all_pairwise_comparisons(self,
                                 y_true: np.ndarray,
                                 predictions_dict: Dict[str, np.ndarray]) -> Dict[str, TestResult]:
        """Perform all pairwise comparisons with multiple testing correction."""
        model_names = list(predictions_dict.keys())
        comparisons = {}
        p_values = []
        
        # All pairwise comparisons
        for model1, model2 in combinations(model_names, 2):
            pred1 = predictions_dict[model1]
            pred2 = predictions_dict[model2]
            
            # Wilcoxon test
            result = self._wilcoxon_test(y_true - pred1, y_true - pred2)
            comparisons[f"{model1}_vs_{model2}"] = result
            p_values.append(result.p_value)
        
        # Apply Bonferroni correction
        n_comparisons = len(p_values)
        corrected_alpha = self.significance_level / n_comparisons
        
        # Update significance based on corrected alpha
        for key, result in comparisons.items():
            result.significant = result.p_value < corrected_alpha
            result.interpretation += f" (Bonferroni corrected, α={corrected_alpha:.4f})"
        
        return comparisons
    
    def _rank_models(self, errors_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Rank models based on performance."""
        # Calculate mean absolute error for each model
        mae_scores = {
            name: np.mean(np.abs(errors))
            for name, errors in errors_dict.items()
        }
        
        # Rank models (lower MAE is better)
        ranked_models = sorted(mae_scores.items(), key=lambda x: x[1])
        
        # Calculate average ranks across samples
        error_matrix = np.column_stack(list(errors_dict.values()))
        ranks = np.array([stats.rankdata(np.abs(row)) for row in error_matrix])
        mean_ranks = np.mean(ranks, axis=0)
        
        model_names = list(errors_dict.keys())
        
        return {
            'ranking_by_mae': ranked_models,
            'mean_ranks': dict(zip(model_names, mean_ranks)),
            'best_model': ranked_models[0][0],
            'mae_scores': mae_scores
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def mcnemar_test(self, predictions_1: np.ndarray, 
                    predictions_2: np.ndarray,
                    y_true: np.ndarray) -> TestResult:
        """
        McNemar's test for binary classification models.
        
        Tests if two models have similar error rates.
        """
        # Ensure binary predictions
        if predictions_1.dtype != bool:
            predictions_1 = predictions_1 > 0.5
        if predictions_2.dtype != bool:
            predictions_2 = predictions_2 > 0.5
        
        # Correct predictions
        correct_1 = predictions_1 == y_true
        correct_2 = predictions_2 == y_true
        
        # Contingency table
        n00 = np.sum(~correct_1 & ~correct_2)  # Both wrong
        n01 = np.sum(~correct_1 & correct_2)   # 1 wrong, 2 correct
        n10 = np.sum(correct_1 & ~correct_2)   # 1 correct, 2 wrong
        n11 = np.sum(correct_1 & correct_2)    # Both correct
        
        # McNemar statistic
        if n01 + n10 == 0:
            statistic = 0
            p_value = 1.0
        else:
            # With continuity correction
            statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return TestResult(
            test_name="McNemar's test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            interpretation=f"Tests if models have different error rates (n01={n01}, n10={n10})"
        )
    
    def permutation_test(self, y_true: np.ndarray,
                        predictions_1: np.ndarray,
                        predictions_2: np.ndarray,
                        n_permutations: int = 1000,
                        metric_func: Optional[callable] = None) -> TestResult:
        """
        Permutation test for model comparison.
        
        Non-parametric test that doesn't assume any distribution.
        """
        if metric_func is None:
            # Default to MAE
            metric_func = lambda y, p: np.mean(np.abs(y - p))
        
        # Original difference
        score_1 = metric_func(y_true, predictions_1)
        score_2 = metric_func(y_true, predictions_2)
        original_diff = score_1 - score_2
        
        # Permutation distribution
        perm_diffs = []
        n = len(y_true)
        
        for _ in range(n_permutations):
            # Randomly swap predictions
            mask = np.random.rand(n) > 0.5
            perm_pred_1 = np.where(mask, predictions_1, predictions_2)
            perm_pred_2 = np.where(mask, predictions_2, predictions_1)
            
            perm_score_1 = metric_func(y_true, perm_pred_1)
            perm_score_2 = metric_func(y_true, perm_pred_2)
            perm_diffs.append(perm_score_1 - perm_score_2)
        
        perm_diffs = np.array(perm_diffs)
        
        # P-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(original_diff))
        
        # Confidence interval
        ci_lower = np.percentile(perm_diffs, self.significance_level/2 * 100)
        ci_upper = np.percentile(perm_diffs, (1 - self.significance_level/2) * 100)
        
        return TestResult(
            test_name="Permutation test",
            statistic=original_diff,
            p_value=p_value,
            significant=p_value < self.significance_level,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=f"Non-parametric test, observed diff={original_diff:.4f}"
        )