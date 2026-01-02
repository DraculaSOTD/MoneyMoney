"""
Alternative Data Sources Module for Cryptocurrency Trading.

This module implements advanced alternative data integration and analysis for capturing
non-traditional signals and market intelligence in cryptocurrency markets.

Key Features:
- Social media sentiment analysis and monitoring
- On-chain analytics and blockchain data processing
- News sentiment analysis and event detection
- Satellite data and economic indicators integration
- Google Trends and search volume analysis
- Regulatory and compliance data monitoring
"""

from models.advanced.alternative_data.social_media_analyzer import SocialMediaAnalyzer
from models.advanced.alternative_data.onchain_analytics import OnChainAnalytics
from models.advanced.alternative_data.news_sentiment_analyzer import NewsSentimentAnalyzer
from models.advanced.alternative_data.economic_indicators import EconomicIndicators
from models.advanced.alternative_data.alternative_data_coordinator import AlternativeDataCoordinator

__all__ = [
    'SocialMediaAnalyzer',
    'OnChainAnalytics',
    'NewsSentimentAnalyzer', 
    'EconomicIndicators',
    'AlternativeDataCoordinator'
]