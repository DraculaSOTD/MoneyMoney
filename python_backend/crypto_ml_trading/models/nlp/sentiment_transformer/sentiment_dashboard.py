"""
Sentiment Analysis Dashboard.

Provides real-time monitoring and visualization of sentiment analysis
with console-based dashboard display.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import os
import sys

from enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from stream_processor import SentimentStreamProcessor
from sentiment_strategies import SentimentStrategyManager

class SentimentDashboard:
    """
    Console-based sentiment monitoring dashboard.
    
    Features:
    - Real-time sentiment display
    - Multi-source monitoring
    - Entity tracking
    - Alert display
    - Performance metrics
    """
    
    def __init__(self,
                 analyzer: EnhancedSentimentAnalyzer,
                 processor: SentimentStreamProcessor,
                 strategy_manager: Optional[SentimentStrategyManager] = None,
                 update_interval: int = 1):
        """
        Initialize dashboard.
        
        Args:
            analyzer: Enhanced sentiment analyzer
            processor: Stream processor
            strategy_manager: Optional strategy manager
            update_interval: Update interval in seconds
        """
        self.analyzer = analyzer
        self.processor = processor
        self.strategy_manager = strategy_manager
        self.update_interval = update_interval
        
        # Display state
        self.is_running = False
        self.display_thread = None
        
        # Data buffers
        self.sentiment_history = deque(maxlen=100)
        self.entity_sentiments = defaultdict(lambda: deque(maxlen=50))
        self.recent_alerts = deque(maxlen=10)
        self.source_stats = defaultdict(int)
        
        # Display configuration
        self.terminal_width = 120
        self.sections = [
            'header',
            'overall_sentiment',
            'entity_sentiments',
            'source_distribution',
            'recent_alerts',
            'trading_signals',
            'performance_metrics'
        ]
    
    def start(self):
        """Start dashboard display."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Register callbacks
        self.processor.register_stream_callback(self._on_stream_update)
        self.processor.register_alert_callback(self._on_alert)
        
        # Start display thread
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True
        )
        self.display_thread.start()
    
    def stop(self):
        """Stop dashboard display."""
        self.is_running = False
        if self.display_thread:
            self.display_thread.join()
    
    def _on_stream_update(self, results: List[Dict[str, Any]]):
        """Handle stream updates."""
        for result in results:
            # Update sentiment history
            self.sentiment_history.append({
                'timestamp': result['timestamp'],
                'sentiment': result['sentiment'],
                'confidence': result['confidence']
            })
            
            # Update entity sentiments
            for entity, entity_data in result.get('entities', {}).items():
                self.entity_sentiments[entity].append({
                    'timestamp': result['timestamp'],
                    'sentiment': entity_data['sentiment']
                })
            
            # Update source statistics
            self.source_stats[result['source']] += 1
    
    def _on_alert(self, alert: Dict[str, Any]):
        """Handle new alert."""
        self.recent_alerts.append(alert)
    
    def _display_loop(self):
        """Main display loop."""
        while self.is_running:
            try:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Display sections
                self._display_header()
                self._display_overall_sentiment()
                self._display_entity_sentiments()
                self._display_source_distribution()
                self._display_recent_alerts()
                
                if self.strategy_manager:
                    self._display_trading_signals()
                    self._display_performance_metrics()
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(self.update_interval)
    
    def _display_header(self):
        """Display dashboard header."""
        print("=" * self.terminal_width)
        print(f"{'Cryptocurrency Sentiment Analysis Dashboard':^{self.terminal_width}}")
        print(f"{'Last Update: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{self.terminal_width}}")
        print("=" * self.terminal_width)
        print()
    
    def _display_overall_sentiment(self):
        """Display overall sentiment metrics."""
        print("📊 OVERALL SENTIMENT")
        print("-" * 50)
        
        if not self.sentiment_history:
            print("No sentiment data available")
            print()
            return
        
        # Get recent sentiments
        recent_sentiments = list(self.sentiment_history)
        current_sentiment = recent_sentiments[-1]['sentiment']
        avg_sentiment = np.mean([s['sentiment'] for s in recent_sentiments])
        
        # Sentiment trend
        if len(recent_sentiments) > 10:
            older = np.mean([s['sentiment'] for s in recent_sentiments[-20:-10]])
            recent = np.mean([s['sentiment'] for s in recent_sentiments[-10:]])
            trend = "📈" if recent > older else "📉" if recent < older else "➡️"
        else:
            trend = "➡️"
        
        # Display metrics
        sentiment_emoji = self._get_sentiment_emoji(current_sentiment)
        print(f"Current: {sentiment_emoji} {current_sentiment:+.3f} {trend}")
        print(f"Average: {avg_sentiment:+.3f}")
        print(f"Volatility: {np.std([s['sentiment'] for s in recent_sentiments]):.3f}")
        
        # Mini chart
        self._display_mini_chart(recent_sentiments)
        print()
    
    def _display_entity_sentiments(self):
        """Display top entity sentiments."""
        print("🏷️  TOP ENTITIES")
        print("-" * 50)
        
        if not self.entity_sentiments:
            print("No entity data available")
            print()
            return
        
        # Get top entities by mention count
        entity_stats = []
        for entity, history in self.entity_sentiments.items():
            if history:
                recent_sentiment = np.mean([h['sentiment'] for h in list(history)[-10:]])
                entity_stats.append({
                    'entity': entity,
                    'sentiment': recent_sentiment,
                    'mentions': len(history)
                })
        
        # Sort by mentions
        entity_stats.sort(key=lambda x: x['mentions'], reverse=True)
        
        # Display top 5
        for i, stats in enumerate(entity_stats[:5]):
            sentiment_bar = self._create_sentiment_bar(stats['sentiment'])
            print(f"{i+1}. {stats['entity']:<12} {sentiment_bar} {stats['sentiment']:+.3f} ({stats['mentions']} mentions)")
        
        print()
    
    def _display_source_distribution(self):
        """Display source distribution."""
        print("📡 SOURCE DISTRIBUTION")
        print("-" * 50)
        
        if not self.source_stats:
            print("No source data available")
            print()
            return
        
        total = sum(self.source_stats.values())
        
        for source, count in sorted(self.source_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"{source:<10} {bar:<50} {percentage:>5.1f}% ({count})")
        
        print()
    
    def _display_recent_alerts(self):
        """Display recent alerts."""
        print("🚨 RECENT ALERTS")
        print("-" * 50)
        
        if not self.recent_alerts:
            print("No recent alerts")
            print()
            return
        
        for alert in list(self.recent_alerts)[-5:]:
            alert_icon = "⚠️" if alert['severity'] == 'high' else "ℹ️"
            timestamp = alert['timestamp'].strftime('%H:%M:%S')
            
            if alert['type'] == 'sentiment_spike':
                print(f"{alert_icon} [{timestamp}] Sentiment spike: {alert['sentiment']:+.3f}")
            elif alert['type'] == 'entity_sentiment':
                print(f"{alert_icon} [{timestamp}] {alert['entity']} sentiment: {alert['sentiment']:+.3f}")
            elif alert['type'] == 'volume_spike':
                print(f"{alert_icon} [{timestamp}] Volume spike: {alert['volume']} messages")
            else:
                print(f"{alert_icon} [{timestamp}] {alert['type']}")
        
        print()
    
    def _display_trading_signals(self):
        """Display recent trading signals."""
        print("📈 TRADING SIGNALS")
        print("-" * 50)
        
        if not self.strategy_manager:
            return
        
        recent_signals = list(self.strategy_manager.signal_history)[-5:]
        
        if not recent_signals:
            print("No recent trading signals")
            print()
            return
        
        for signal in recent_signals:
            action_emoji = "🟢" if signal.action == 'buy' else "🔴" if signal.action == 'sell' else "⚪"
            timestamp = signal.timestamp.strftime('%H:%M:%S')
            
            print(f"{action_emoji} [{timestamp}] {signal.asset} - {signal.action.upper()}")
            print(f"   Strategy: {signal.metadata.get('strategy', 'unknown')}")
            print(f"   Confidence: {signal.confidence:.1%} | Sentiment: {signal.sentiment_score:+.3f}")
            print(f"   Reason: {signal.reason}")
            print()
    
    def _display_performance_metrics(self):
        """Display strategy performance metrics."""
        print("📊 PERFORMANCE METRICS")
        print("-" * 50)
        
        if not self.strategy_manager:
            return
        
        report = self.strategy_manager.get_performance_report()
        
        print(f"Active Positions: {report['active_positions']}")
        print(f"Daily PnL: {report['daily_pnl']:+.2%}")
        print()
        
        for strategy, metrics in report['strategy_performance'].items():
            print(f"{strategy}:")
            print(f"  Signals: {metrics['total_signals']} | Executed: {metrics['executed_signals']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%} | Total PnL: {metrics['total_pnl']:+.2%}")
        
        print()
    
    def _display_mini_chart(self, sentiment_history: List[Dict]):
        """Display mini sentiment chart."""
        if len(sentiment_history) < 20:
            return
        
        # Get last 20 sentiment values
        values = [s['sentiment'] for s in sentiment_history[-20:]]
        
        # Normalize to 0-5 range for display
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 0.001:
            return
        
        normalized = [(v - min_val) / (max_val - min_val) * 5 for v in values]
        
        # Create chart
        print("\nTrend (last 20):")
        for row in range(5, -1, -1):
            line = ""
            for val in normalized:
                if val >= row:
                    line += "█"
                else:
                    line += " "
            print(f"  {line}")
        print(f"  {min_val:+.3f}" + " " * 14 + f"{max_val:+.3f}")
    
    def _create_sentiment_bar(self, sentiment: float) -> str:
        """Create sentiment visualization bar."""
        # Normalize to -1 to 1 range
        sentiment = max(-1, min(1, sentiment))
        
        # Create bar
        if sentiment > 0:
            bar_length = int(sentiment * 20)
            return "🟢" + "█" * bar_length + " " * (20 - bar_length)
        else:
            bar_length = int(abs(sentiment) * 20)
            return "🔴" + "█" * bar_length + " " * (20 - bar_length)
    
    def _get_sentiment_emoji(self, sentiment: float) -> str:
        """Get emoji for sentiment value."""
        if sentiment > 0.5:
            return "😊"
        elif sentiment > 0.2:
            return "🙂"
        elif sentiment > -0.2:
            return "😐"
        elif sentiment > -0.5:
            return "😟"
        else:
            return "😰"
    
    def display_summary_report(self):
        """Display summary report."""
        print("\n" + "=" * self.terminal_width)
        print(f"{'SENTIMENT ANALYSIS SUMMARY REPORT':^{self.terminal_width}}")
        print("=" * self.terminal_width)
        
        # Time windows
        windows = self.processor.get_window_statistics('1h')
        if windows:
            print("\n📊 HOURLY STATISTICS")
            print("-" * 50)
            print(f"Data Points: {windows['data_points']}")
            print(f"Average Sentiment: {windows['avg_sentiment']:+.3f}")
            print(f"Sentiment Volatility: {windows['sentiment_std']:.3f}")
            print(f"Positive Ratio: {windows['positive_ratio']:.1%}")
            print(f"Negative Ratio: {windows['negative_ratio']:.1%}")
        
        # Top entities
        print("\n🏷️  TOP ENTITIES (BY MENTIONS)")
        print("-" * 50)
        for i, (entity, count) in enumerate(windows.get('top_entities', [])[:10]):
            sentiment = windows['entity_sentiments'].get(entity, 0)
            print(f"{i+1:>2}. {entity:<15} {count:>4} mentions | Sentiment: {sentiment:+.3f}")
        
        # Processing metrics
        metrics = self.processor.get_real_time_sentiment()['metrics']
        print("\n⚙️  PROCESSING METRICS")
        print("-" * 50)
        print(f"Processed: {metrics['processed_count']:,}")
        print(f"Error Rate: {metrics['error_rate']:.1%}")
        print(f"Throughput: {metrics['throughput']:.1f} msg/s")
        print(f"Avg Processing Time: {metrics['avg_processing_time']*1000:.1f} ms")
        
        print("\n" + "=" * self.terminal_width)