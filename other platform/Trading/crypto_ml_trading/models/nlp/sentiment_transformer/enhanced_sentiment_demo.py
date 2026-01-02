"""
Enhanced Sentiment Analysis System Demonstration.

Comprehensive demo showcasing all components of the enhanced sentiment system.
"""

import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.nlp.sentiment_transformer.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from models.nlp.sentiment_transformer.multi_source_aggregator import MultiSourceAggregator
from models.nlp.sentiment_transformer.sentiment_features import SentimentFeatureExtractor
from models.nlp.sentiment_transformer.stream_processor import SentimentStreamProcessor
from models.nlp.sentiment_transformer.sentiment_strategies import (
    MomentumSentimentStrategy, ContrarianSentimentStrategy,
    EntityFocusedStrategy, SentimentDivergenceStrategy,
    SentimentStrategyManager
)
from models.nlp.sentiment_transformer.sentiment_dashboard import SentimentDashboard

def demonstrate_enhanced_sentiment_analysis():
    """Demonstrate enhanced sentiment analysis capabilities."""
    print("=" * 80)
    print("ENHANCED SENTIMENT ANALYSIS SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    print("\n1. Initializing System Components...")
    
    # Create analyzer
    analyzer = EnhancedSentimentAnalyzer(
        enable_streaming=True,
        alert_threshold=0.7
    )
    print("✓ Enhanced Sentiment Analyzer initialized")
    
    # Create aggregator
    aggregator = MultiSourceAggregator(
        enable_streaming=True,
        cache_size=10000
    )
    print("✓ Multi-Source Aggregator initialized")
    
    # Create feature extractor
    feature_extractor = SentimentFeatureExtractor()
    print("✓ Sentiment Feature Extractor initialized")
    
    # Create stream processor
    processor = SentimentStreamProcessor(
        analyzer=analyzer,
        aggregator=aggregator,
        feature_extractor=feature_extractor,
        batch_size=32,
        num_workers=4
    )
    print("✓ Stream Processor initialized")
    
    # Create trading strategies
    strategies = [
        MomentumSentimentStrategy(momentum_threshold=0.3),
        ContrarianSentimentStrategy(extreme_threshold=0.8),
        EntityFocusedStrategy(target_entities=['BTC', 'ETH']),
        SentimentDivergenceStrategy(divergence_threshold=0.3)
    ]
    
    strategy_manager = SentimentStrategyManager(
        strategies=strategies,
        max_positions=10,
        max_position_size=0.1
    )
    print("✓ Strategy Manager initialized with 4 strategies")
    
    # Basic sentiment analysis demo
    print("\n2. Basic Sentiment Analysis Demo")
    print("-" * 50)
    
    test_texts = [
        "Bitcoin is absolutely mooning! 🚀🚀🚀 This is incredible! BTC to 100k!",
        "ETH looking bearish, might dump soon. Getting out of my positions 📉",
        "Solid technical analysis: BTC forming bullish pattern at support level",
        "This is definitely NOT a scam guys... sure thing... 🙄",
        "Breaking: Major exchange hack! Funds are NOT SAFU! 😱"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Extract features
        features = feature_extractor.extract_features(text)
        
        # Analyze with entities
        analysis = analyzer.analyze_with_entities(text, source='demo', author='user123')
        
        print(f"Sentiment Score: {analysis['sentiment_score']:+.3f}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        
        # Show detected entities
        if analysis['entities']:
            print("Detected Entities:")
            for entity, data in analysis['entities'].items():
                print(f"  - {entity}: {data['sentiment']:+.3f} ({data['mentions']} mentions)")
        
        # Show key features
        print(f"Features:")
        print(f"  - Sarcasm: {features.sarcasm_probability:.2%}")
        print(f"  - Urgency: {features.urgency_score:.2%}")
        print(f"  - Technical Score: {features.technical_score:.2%}")
        print(f"  - Emotion: {features.dominant_emotion}")
    
    # Multi-source aggregation demo
    print("\n\n3. Multi-Source Data Aggregation Demo")
    print("-" * 50)
    
    # Fetch data from mock sources
    all_data = aggregator.fetch_all_sources("BTC", limit_per_source=10)
    
    print(f"Fetched {len(all_data)} data points from {len(aggregator.sources)} sources")
    
    # Show source statistics
    stats = aggregator.get_source_statistics()
    for source, source_stats in stats.items():
        print(f"\n{source.upper()}:")
        print(f"  Total Items: {source_stats['total_items']}")
        print(f"  Quality Score: {source_stats['quality_score']:.2f}")
        print(f"  Error Count: {source_stats['error_count']}")
    
    # Calculate source weights
    weights = aggregator.calculate_source_weights()
    print("\nOptimal Source Weights:")
    for source, weight in weights.items():
        print(f"  {source}: {weight:.2%}")
    
    # Stream processing demo
    print("\n\n4. Real-Time Stream Processing Demo")
    print("-" * 50)
    
    print("Starting stream processing...")
    
    # Register callbacks
    alerts_received = []
    
    def alert_callback(alert):
        alerts_received.append(alert)
        print(f"\n🚨 ALERT: {alert['type']} - {alert.get('entity', 'market')} "
              f"sentiment: {alert.get('sentiment', 0):+.3f}")
    
    analyzer.register_alert_callback(alert_callback)
    processor.register_alert_callback(alert_callback)
    
    # Start streaming
    processor.start_processing(['BTC', 'ETH'])
    
    print("Processing stream for 10 seconds...")
    time.sleep(10)
    
    # Get stream statistics
    stream_stats = processor.get_real_time_sentiment()
    
    print("\nStream Processing Results:")
    print(f"Processed: {stream_stats['metrics']['processed_count']} messages")
    print(f"Throughput: {stream_stats['metrics']['throughput']:.1f} msg/s")
    print(f"Error Rate: {stream_stats['metrics']['error_rate']:.1%}")
    print(f"Alerts Generated: {len(alerts_received)}")
    
    # Stop streaming
    processor.stop_processing()
    
    # Trading strategies demo
    print("\n\n5. Trading Strategies Demo")
    print("-" * 50)
    
    # Simulate market data
    market_data = {
        'price': 45000,
        'volume': 1000000,
        'volatility': 0.02
    }
    
    # Process sentiment through strategies
    sentiment_data = {
        'asset': 'BTC',
        'sentiment_score': 0.65,
        'velocity': 0.4,
        'acceleration': 0.1,
        'confidence': 0.8,
        'entities': {
            'BTC': {'sentiment': 0.7, 'mentions': 50, 'type': 'coin'}
        }
    }
    
    signals = strategy_manager.process_sentiment_update(sentiment_data, market_data)
    
    print(f"Generated {len(signals)} trading signals:")
    for signal in signals:
        print(f"\n{signal.metadata['strategy']} Strategy:")
        print(f"  Action: {signal.action.upper()}")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Position Size: {signal.position_size:.2%}")
        print(f"  Reason: {signal.reason}")
    
    # Entity sentiment tracking demo
    print("\n\n6. Entity Sentiment Tracking Demo")
    print("-" * 50)
    
    # Get entity sentiment history
    btc_history = analyzer.get_entity_sentiment_history('BTC', hours=1)
    if not btc_history.empty:
        print(f"BTC Sentiment History ({len(btc_history)} data points):")
        print(f"  Current: {btc_history['sentiment'].iloc[-1]:+.3f}")
        print(f"  Average: {btc_history['sentiment'].mean():+.3f}")
        print(f"  Volatility: {btc_history['sentiment'].std():.3f}")
    
    # Timeframe analysis
    print("\nMulti-Timeframe Analysis:")
    for tf in ['5m', '15m', '1h']:
        tf_data = analyzer.get_timeframe_sentiment(tf)
        if tf_data:
            print(f"\n{tf} timeframe:")
            print(f"  Current: {tf_data['current_sentiment']:+.3f}")
            print(f"  Average: {tf_data['average_sentiment']:+.3f}")
            print(f"  Trend: {tf_data['trend']:+.3f}")
    
    # Anomaly detection demo
    print("\n\n7. Anomaly Detection Demo")
    print("-" * 50)
    
    anomalies = analyzer.detect_sentiment_anomalies_enhanced(['BTC', 'ETH'])
    
    if anomalies:
        print(f"Detected {len(anomalies)} anomalies:")
        for anomaly in anomalies[:5]:
            print(f"\n{anomaly['type']}:")
            if 'entity' in anomaly:
                print(f"  Entity: {anomaly['entity']}")
            print(f"  Severity: {anomaly['severity']}")
            if 'change' in anomaly:
                print(f"  Change: {anomaly['change']:+.3f}")
    else:
        print("No anomalies detected")
    
    # Dashboard demo
    print("\n\n8. Real-Time Dashboard Demo")
    print("-" * 50)
    
    dashboard = SentimentDashboard(
        analyzer=analyzer,
        processor=processor,
        strategy_manager=strategy_manager
    )
    
    print("Starting dashboard for 15 seconds...")
    print("(Dashboard will appear in a moment)")
    
    # Start dashboard
    dashboard.start()
    processor.start_processing(['BTC', 'ETH'])
    
    # Run for 15 seconds
    time.sleep(15)
    
    # Stop dashboard
    processor.stop_processing()
    dashboard.stop()
    
    # Final report
    print("\n\n9. Final Sentiment Report")
    print("-" * 50)
    
    report = analyzer.get_sentiment_report(['BTC', 'ETH'])
    
    print(f"Report Generated: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nAsset Analysis:")
    for asset, data in report['assets'].items():
        print(f"\n{asset}:")
        print(f"  Signal: {data['signal'].upper()}")
        print(f"  Sentiment: {data['sentiment']:+.3f}")
        print(f"  Confidence: {data['confidence']:.2%}")
    
    print("\nTop Entities:")
    for entity_data in report['top_entities'][:5]:
        print(f"  {entity_data['entity']}: {entity_data['mentions']} mentions, "
              f"sentiment: {entity_data['recent_sentiment']:+.3f}")
    
    print("\nSentiment Momentum:")
    for tf, momentum in report['sentiment_momentum'].items():
        direction = "↑" if momentum > 0 else "↓" if momentum < 0 else "→"
        print(f"  {tf}: {momentum:+.3f} {direction}")
    
    # Performance summary
    if strategy_manager:
        perf_report = strategy_manager.get_performance_report()
        print("\n\nStrategy Performance Summary:")
        print("-" * 50)
        for strategy, metrics in perf_report['strategy_performance'].items():
            print(f"\n{strategy}:")
            print(f"  Total Signals: {metrics['total_signals']}")
            print(f"  Execution Rate: {metrics['execution_rate']:.1%}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main():
    """Run the enhanced sentiment analysis demo."""
    try:
        demonstrate_enhanced_sentiment_analysis()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()