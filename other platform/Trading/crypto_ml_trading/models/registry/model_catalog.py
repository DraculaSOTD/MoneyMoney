"""
Model Catalog for easy discovery and selection.

Provides a searchable catalog of all available models with
filtering, ranking, and recommendation capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from collections import defaultdict

from .model_registry import ModelRegistry, ModelMetadata, ModelStatus, ModelType

logger = logging.getLogger(__name__)


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning."""
    model_id: str
    model_name: str
    score: float
    reasons: List[str]
    pros: List[str]
    cons: List[str]
    similar_models: List[str]
    metadata: ModelMetadata


@dataclass
class ModelFilter:
    """Filters for model search."""
    model_types: Optional[List[ModelType]] = None
    status: Optional[List[ModelStatus]] = None
    min_performance: Optional[Dict[str, float]] = None
    max_performance: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None
    asset_classes: Optional[List[str]] = None
    time_horizons: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    authors: Optional[List[str]] = None
    min_validation_score: Optional[float] = None


@dataclass
class ModelRanking:
    """Ranking criteria for models."""
    metric: str
    weight: float
    higher_is_better: bool = True
    
    
class ModelCatalog:
    """
    Searchable catalog of trading models.
    
    Features:
    - Advanced search and filtering
    - Model recommendations
    - Performance-based ranking
    - Use case matching
    - Model comparison
    - Ensemble suggestions
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize model catalog.
        
        Args:
            registry: Model registry instance
        """
        self.registry = registry
        
        # Default ranking criteria
        self.default_rankings = [
            ModelRanking("sharpe_ratio", 0.3, True),
            ModelRanking("accuracy", 0.2, True),
            ModelRanking("max_drawdown", 0.2, False),
            ModelRanking("win_rate", 0.15, True),
            ModelRanking("validation_score", 0.15, True)
        ]
        
        # Model similarity cache
        self.similarity_cache = {}
        
        # Performance benchmarks
        self.benchmarks = self._load_benchmarks()
        
        logger.info("Model catalog initialized")
    
    def _load_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load performance benchmarks for different model types."""
        return {
            ModelType.DEEP_LEARNING: {
                "sharpe_ratio": 1.5,
                "accuracy": 0.6,
                "max_drawdown": -0.15,
                "win_rate": 0.55
            },
            ModelType.REINFORCEMENT_LEARNING: {
                "sharpe_ratio": 2.0,
                "accuracy": 0.55,
                "max_drawdown": -0.20,
                "win_rate": 0.52
            },
            ModelType.ENSEMBLE: {
                "sharpe_ratio": 2.5,
                "accuracy": 0.65,
                "max_drawdown": -0.10,
                "win_rate": 0.60
            },
            ModelType.STATISTICAL: {
                "sharpe_ratio": 1.2,
                "accuracy": 0.55,
                "max_drawdown": -0.12,
                "win_rate": 0.53
            }
        }
    
    def search(self, 
               query: Optional[str] = None,
               filters: Optional[ModelFilter] = None,
               rankings: Optional[List[ModelRanking]] = None,
               limit: int = 20) -> List[ModelMetadata]:
        """
        Search for models with advanced filtering and ranking.
        
        Args:
            query: Text search query
            filters: Filter criteria
            rankings: Ranking criteria
            limit: Maximum results
            
        Returns:
            List of matching models
        """
        # Start with all models
        models = list(self.registry.registry.values())
        
        # Apply filters
        if filters:
            models = self._apply_filters(models, filters)
        
        # Apply text search
        if query:
            models = self._text_search(models, query)
        
        # Rank models
        rankings = rankings or self.default_rankings
        models = self._rank_models(models, rankings)
        
        # Apply limit
        return models[:limit]
    
    def _apply_filters(self, models: List[ModelMetadata], 
                      filters: ModelFilter) -> List[ModelMetadata]:
        """Apply filters to model list."""
        filtered = []
        
        for model in models:
            # Model type filter
            if filters.model_types and model.model_type not in filters.model_types:
                continue
            
            # Status filter
            if filters.status and model.status not in filters.status:
                continue
            
            # Performance filters
            if filters.min_performance:
                meets_min = all(
                    model.performance_metrics.get(metric, -float('inf')) >= value
                    for metric, value in filters.min_performance.items()
                )
                if not meets_min:
                    continue
            
            if filters.max_performance:
                meets_max = all(
                    model.performance_metrics.get(metric, float('inf')) <= value
                    for metric, value in filters.max_performance.items()
                )
                if not meets_max:
                    continue
            
            # Tag filter
            if filters.tags:
                has_tags = any(tag in model.tags for tag in filters.tags)
                if not has_tags:
                    continue
            
            # Asset class filter
            if filters.asset_classes:
                has_assets = any(
                    asset in model.asset_classes 
                    for asset in filters.asset_classes
                )
                if not has_assets:
                    continue
            
            # Time horizon filter
            if filters.time_horizons:
                has_horizons = any(
                    horizon in model.time_horizons 
                    for horizon in filters.time_horizons
                )
                if not has_horizons:
                    continue
            
            # Date filters
            if filters.created_after and model.created_at < filters.created_after:
                continue
            
            if filters.created_before and model.created_at > filters.created_before:
                continue
            
            # Author filter
            if filters.authors and model.author not in filters.authors:
                continue
            
            # Validation score filter
            if filters.min_validation_score:
                val_score = model.validation_metrics.get('overall_score', 0)
                if val_score < filters.min_validation_score:
                    continue
            
            filtered.append(model)
        
        return filtered
    
    def _text_search(self, models: List[ModelMetadata], 
                    query: str) -> List[ModelMetadata]:
        """Search models by text query."""
        query_lower = query.lower()
        results = []
        
        for model in models:
            # Calculate relevance score
            score = 0
            
            # Name match
            if query_lower in model.model_name.lower():
                score += 10
            
            # Description match
            if query_lower in model.description.lower():
                score += 5
            
            # Tag match
            for tag in model.tags:
                if query_lower in tag.lower():
                    score += 3
            
            # Architecture match
            arch_str = json.dumps(model.architecture).lower()
            if query_lower in arch_str:
                score += 2
            
            if score > 0:
                results.append((score, model))
        
        # Sort by relevance
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [model for _, model in results]
    
    def _rank_models(self, models: List[ModelMetadata], 
                    rankings: List[ModelRanking]) -> List[ModelMetadata]:
        """Rank models based on criteria."""
        scores = []
        
        for model in models:
            total_score = 0
            
            for ranking in rankings:
                metric_value = model.performance_metrics.get(ranking.metric, 0)
                
                # Normalize metric (0-1 scale)
                benchmark = self.benchmarks.get(
                    model.model_type, {}
                ).get(ranking.metric, 1)
                
                if ranking.higher_is_better:
                    normalized = min(metric_value / (benchmark + 1e-8), 2.0)
                else:
                    # For metrics where lower is better (like drawdown)
                    normalized = min(benchmark / (abs(metric_value) + 1e-8), 2.0)
                
                # Apply weight
                total_score += normalized * ranking.weight
            
            scores.append((total_score, model))
        
        # Sort by score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [model for _, model in scores]
    
    def recommend_models(self, 
                        use_case: str,
                        requirements: Dict[str, Any],
                        current_models: Optional[List[str]] = None,
                        limit: int = 5) -> List[ModelRecommendation]:
        """
        Recommend models for specific use case.
        
        Args:
            use_case: Use case description
            requirements: Specific requirements
            current_models: Currently used models (for ensemble)
            limit: Maximum recommendations
            
        Returns:
            List of model recommendations
        """
        recommendations = []
        
        # Define use case mappings
        use_case_filters = self._get_use_case_filters(use_case, requirements)
        
        # Search for matching models
        candidates = self.search(
            filters=use_case_filters,
            limit=limit * 3  # Get more candidates for filtering
        )
        
        # Generate recommendations
        for model in candidates[:limit]:
            recommendation = self._generate_recommendation(
                model, use_case, requirements, current_models
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_use_case_filters(self, use_case: str, 
                            requirements: Dict[str, Any]) -> ModelFilter:
        """Map use case to model filters."""
        filters = ModelFilter()
        
        # Common use cases
        if "high frequency" in use_case.lower():
            filters.model_types = [
                ModelType.DEEP_LEARNING, 
                ModelType.REINFORCEMENT_LEARNING
            ]
            filters.tags = ["low_latency", "high_frequency"]
            filters.time_horizons = ["1m", "5m", "15m"]
            
        elif "long term" in use_case.lower():
            filters.model_types = [
                ModelType.STATISTICAL,
                ModelType.ENSEMBLE
            ]
            filters.time_horizons = ["1d", "1w", "1M"]
            
        elif "risk management" in use_case.lower():
            filters.model_types = [ModelType.RISK_MANAGEMENT]
            filters.tags = ["risk", "portfolio", "hedging"]
            
        elif "sentiment" in use_case.lower():
            filters.model_types = [
                ModelType.SENTIMENT,
                ModelType.ALTERNATIVE_DATA
            ]
            filters.tags = ["sentiment", "news", "social"]
        
        # Apply specific requirements
        if "min_sharpe" in requirements:
            filters.min_performance = {"sharpe_ratio": requirements["min_sharpe"]}
        
        if "max_drawdown" in requirements:
            filters.max_performance = {"max_drawdown": requirements["max_drawdown"]}
        
        if "assets" in requirements:
            filters.asset_classes = requirements["assets"]
        
        # Only production models for live trading
        if requirements.get("production_only", False):
            filters.status = [ModelStatus.PRODUCTION]
        
        return filters
    
    def _generate_recommendation(self, 
                               model: ModelMetadata,
                               use_case: str,
                               requirements: Dict[str, Any],
                               current_models: Optional[List[str]]) -> ModelRecommendation:
        """Generate detailed recommendation for a model."""
        score = 0
        reasons = []
        pros = []
        cons = []
        
        # Performance-based scoring
        sharpe = model.performance_metrics.get("sharpe_ratio", 0)
        if sharpe > 2.0:
            score += 20
            pros.append(f"Excellent Sharpe ratio: {sharpe:.2f}")
        elif sharpe > 1.5:
            score += 10
            pros.append(f"Good Sharpe ratio: {sharpe:.2f}")
        else:
            cons.append(f"Low Sharpe ratio: {sharpe:.2f}")
        
        # Drawdown scoring
        max_dd = model.performance_metrics.get("max_drawdown", -1)
        if max_dd > -0.10:
            score += 15
            pros.append(f"Low drawdown: {max_dd:.2%}")
        elif max_dd > -0.20:
            score += 5
        else:
            cons.append(f"High drawdown: {max_dd:.2%}")
        
        # Validation scoring
        val_score = model.validation_metrics.get("overall_score", 0)
        if val_score > 0.8:
            score += 10
            pros.append("Strong validation performance")
        
        # Use case fit
        if self._matches_use_case(model, use_case):
            score += 15
            reasons.append(f"Good fit for {use_case}")
        
        # Ensemble compatibility
        if current_models:
            compatibility = self._calculate_ensemble_compatibility(
                model, current_models
            )
            if compatibility > 0.7:
                score += 10
                reasons.append("High ensemble compatibility")
                pros.append("Complements existing models well")
        
        # Find similar models
        similar_models = self._find_similar_models(model.model_id, limit=3)
        
        return ModelRecommendation(
            model_id=model.model_id,
            model_name=model.model_name,
            score=score,
            reasons=reasons,
            pros=pros,
            cons=cons,
            similar_models=similar_models,
            metadata=model
        )
    
    def _matches_use_case(self, model: ModelMetadata, use_case: str) -> bool:
        """Check if model matches use case."""
        use_case_lower = use_case.lower()
        
        # Check tags
        for tag in model.tags:
            if tag.lower() in use_case_lower:
                return True
        
        # Check description
        if any(word in model.description.lower() 
               for word in use_case_lower.split()):
            return True
        
        # Check time horizons
        if "short" in use_case_lower and "1m" in model.time_horizons:
            return True
        if "long" in use_case_lower and "1d" in model.time_horizons:
            return True
        
        return False
    
    def _calculate_ensemble_compatibility(self, 
                                        model: ModelMetadata,
                                        current_model_ids: List[str]) -> float:
        """Calculate how well model fits with existing ensemble."""
        if not current_model_ids:
            return 1.0
        
        compatibility_scores = []
        
        for model_id in current_model_ids:
            if model_id in self.registry.registry:
                other_model = self.registry.registry[model_id]
                
                # Different model types are more compatible
                if model.model_type != other_model.model_type:
                    compatibility_scores.append(0.8)
                else:
                    compatibility_scores.append(0.4)
                
                # Different architectures within same type
                if model.model_type == other_model.model_type:
                    arch_similarity = self._calculate_architecture_similarity(
                        model.architecture, other_model.architecture
                    )
                    compatibility_scores.append(1 - arch_similarity)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.5
    
    def _calculate_architecture_similarity(self, 
                                         arch1: Dict[str, Any],
                                         arch2: Dict[str, Any]) -> float:
        """Calculate similarity between architectures."""
        # Simple Jaccard similarity on architecture keys
        keys1 = set(str(k) for k in arch1.keys())
        keys2 = set(str(k) for k in arch2.keys())
        
        if not keys1 and not keys2:
            return 1.0
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        return len(intersection) / len(union) if union else 0
    
    def _find_similar_models(self, model_id: str, limit: int = 5) -> List[str]:
        """Find similar models based on various criteria."""
        if model_id not in self.registry.registry:
            return []
        
        target_model = self.registry.registry[model_id]
        similarities = []
        
        for other_id, other_model in self.registry.registry.items():
            if other_id == model_id:
                continue
            
            # Calculate similarity score
            score = 0
            
            # Same model type
            if target_model.model_type == other_model.model_type:
                score += 0.3
            
            # Similar performance
            perf_similarity = self._calculate_performance_similarity(
                target_model.performance_metrics,
                other_model.performance_metrics
            )
            score += 0.4 * perf_similarity
            
            # Common tags
            common_tags = set(target_model.tags) & set(other_model.tags)
            if target_model.tags:
                tag_similarity = len(common_tags) / len(target_model.tags)
                score += 0.2 * tag_similarity
            
            # Similar time horizons
            common_horizons = set(target_model.time_horizons) & set(other_model.time_horizons)
            if target_model.time_horizons:
                horizon_similarity = len(common_horizons) / len(target_model.time_horizons)
                score += 0.1 * horizon_similarity
            
            similarities.append((score, other_id))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [model_id for _, model_id in similarities[:limit]]
    
    def _calculate_performance_similarity(self, 
                                        perf1: Dict[str, float],
                                        perf2: Dict[str, float]) -> float:
        """Calculate similarity between performance metrics."""
        common_metrics = set(perf1.keys()) & set(perf2.keys())
        
        if not common_metrics:
            return 0
        
        distances = []
        for metric in common_metrics:
            val1 = perf1[metric]
            val2 = perf2[metric]
            
            # Normalize by benchmark
            benchmark = 1.0  # Default
            for model_type, benchmarks in self.benchmarks.items():
                if metric in benchmarks:
                    benchmark = benchmarks[metric]
                    break
            
            # Calculate normalized distance
            distance = abs(val1 - val2) / (abs(benchmark) + 1e-8)
            distances.append(distance)
        
        # Convert average distance to similarity
        avg_distance = np.mean(distances)
        similarity = 1 / (1 + avg_distance)
        
        return similarity
    
    def create_ensemble_suggestions(self, 
                                  base_models: List[str],
                                  target_performance: Dict[str, float],
                                  max_models: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest models to add to ensemble.
        
        Args:
            base_models: Current models in ensemble
            target_performance: Target performance metrics
            max_models: Maximum ensemble size
            
        Returns:
            List of ensemble suggestions
        """
        suggestions = []
        
        # Analyze current ensemble
        current_types = set()
        current_performance = defaultdict(list)
        
        for model_id in base_models:
            if model_id in self.registry.registry:
                model = self.registry.registry[model_id]
                current_types.add(model.model_type)
                
                for metric, value in model.performance_metrics.items():
                    current_performance[metric].append(value)
        
        # Calculate ensemble gaps
        gaps = {}
        for metric, target in target_performance.items():
            current_avg = np.mean(current_performance[metric]) if current_performance[metric] else 0
            gaps[metric] = target - current_avg
        
        # Find complementary models
        filters = ModelFilter(
            status=[ModelStatus.PRODUCTION, ModelStatus.STAGING]
        )
        
        # Prioritize different model types
        missing_types = set(ModelType) - current_types
        if missing_types:
            filters.model_types = list(missing_types)
        
        candidates = self.search(filters=filters, limit=20)
        
        # Score candidates
        for candidate in candidates:
            if candidate.model_id in base_models:
                continue
            
            score = 0
            improvement_metrics = []
            
            # Check if candidate improves gaps
            for metric, gap in gaps.items():
                candidate_value = candidate.performance_metrics.get(metric, 0)
                current_avg = np.mean(current_performance[metric]) if current_performance[metric] else 0
                
                if gap > 0 and candidate_value > current_avg:
                    score += 10
                    improvement_metrics.append(metric)
            
            # Diversity bonus
            if candidate.model_type not in current_types:
                score += 15
            
            # Compatibility score
            compatibility = self._calculate_ensemble_compatibility(
                candidate, base_models
            )
            score += compatibility * 20
            
            if score > 0 and len(base_models) < max_models:
                suggestions.append({
                    "model_id": candidate.model_id,
                    "model_name": candidate.model_name,
                    "score": score,
                    "improves": improvement_metrics,
                    "adds_diversity": candidate.model_type not in current_types,
                    "compatibility": compatibility,
                    "expected_ensemble_size": len(base_models) + 1
                })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:3]  # Top 3 suggestions
    
    def export_catalog(self, output_path: str, 
                      format: str = "markdown") -> None:
        """
        Export model catalog in various formats.
        
        Args:
            output_path: Output file path
            format: Export format (markdown, json, html)
        """
        if format == "markdown":
            self._export_markdown(output_path)
        elif format == "json":
            self._export_json(output_path)
        elif format == "html":
            self._export_html(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, output_path: str):
        """Export catalog as Markdown."""
        lines = ["# Model Catalog\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Total Models: {len(self.registry.registry)}\n")
        
        # Group by type
        by_type = defaultdict(list)
        for model in self.registry.registry.values():
            by_type[model.model_type].append(model)
        
        for model_type, models in by_type.items():
            lines.append(f"\n## {model_type.value.replace('_', ' ').title()}\n")
            
            # Sort by performance
            models.sort(
                key=lambda m: m.performance_metrics.get('sharpe_ratio', 0),
                reverse=True
            )
            
            for model in models:
                lines.append(f"### {model.model_name}\n")
                lines.append(f"- **ID**: {model.model_id}\n")
                lines.append(f"- **Status**: {model.status.value}\n")
                lines.append(f"- **Author**: {model.author}\n")
                lines.append(f"- **Created**: {model.created_at.strftime('%Y-%m-%d')}\n")
                lines.append(f"- **Description**: {model.description}\n")
                
                # Performance metrics
                lines.append("- **Performance**:\n")
                for metric, value in model.performance_metrics.items():
                    lines.append(f"  - {metric}: {value:.4f}\n")
                
                lines.append("\n")
        
        with open(output_path, 'w') as f:
            f.writelines(lines)
    
    def _export_json(self, output_path: str):
        """Export catalog as JSON."""
        catalog_data = {
            "generated": datetime.now().isoformat(),
            "total_models": len(self.registry.registry),
            "models": []
        }
        
        for model in self.registry.registry.values():
            catalog_data["models"].append(model.to_dict())
        
        with open(output_path, 'w') as f:
            json.dump(catalog_data, f, indent=2)
    
    def _export_html(self, output_path: str):
        """Export catalog as HTML."""
        # Simple HTML table
        html = ["<html><head><title>Model Catalog</title>"]
        html.append("<style>")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("</style></head><body>")
        html.append("<h1>Model Catalog</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p>Total Models: {len(self.registry.registry)}</p>")
        
        # Create table
        html.append("<table>")
        html.append("<tr>")
        html.append("<th>Model Name</th>")
        html.append("<th>Type</th>")
        html.append("<th>Status</th>")
        html.append("<th>Sharpe Ratio</th>")
        html.append("<th>Accuracy</th>")
        html.append("<th>Max Drawdown</th>")
        html.append("<th>Author</th>")
        html.append("<th>Created</th>")
        html.append("</tr>")
        
        for model in self.registry.registry.values():
            html.append("<tr>")
            html.append(f"<td>{model.model_name}</td>")
            html.append(f"<td>{model.model_type.value}</td>")
            html.append(f"<td>{model.status.value}</td>")
            html.append(f"<td>{model.performance_metrics.get('sharpe_ratio', 'N/A'):.3f}</td>")
            html.append(f"<td>{model.performance_metrics.get('accuracy', 'N/A'):.3f}</td>")
            html.append(f"<td>{model.performance_metrics.get('max_drawdown', 'N/A'):.3f}</td>")
            html.append(f"<td>{model.author}</td>")
            html.append(f"<td>{model.created_at.strftime('%Y-%m-%d')}</td>")
            html.append("</tr>")
        
        html.append("</table>")
        html.append("</body></html>")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(html))