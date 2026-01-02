"""
Real-time Sentiment Stream Processing.

Provides high-performance stream processing for real-time sentiment analysis
with batch processing, windowing, and alert generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue
import time
import logging

from enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from multi_source_aggregator import MultiSourceAggregator, SentimentDataPoint
from sentiment_features import SentimentFeatureExtractor, SentimentFeatures

logger = logging.getLogger(__name__)


@dataclass
class StreamMetrics:
    """Metrics for stream processing performance."""
    processed_count: int = 0
    error_count: int = 0
    avg_processing_time: float = 0.0
    throughput: float = 0.0
    queue_depth: int = 0
    last_update: datetime = field(default_factory=datetime.now)


class SentimentStreamProcessor:
    """
    Real-time sentiment stream processor with windowing and alerting.
    
    Features:
    - Real-time processing pipeline
    - Sliding window aggregation
    - Batch processing for efficiency
    - Alert generation
    - Performance monitoring
    """
    
    def __init__(self,
                 analyzer: EnhancedSentimentAnalyzer,
                 aggregator: MultiSourceAggregator,
                 feature_extractor: SentimentFeatureExtractor,
                 batch_size: int = 32,
                 window_minutes: int = 5,
                 num_workers: int = 4):
        """
        Initialize stream processor.
        
        Args:
            analyzer: Enhanced sentiment analyzer
            aggregator: Multi-source aggregator
            feature_extractor: Feature extractor
            batch_size: Batch size for processing
            window_minutes: Sliding window size in minutes
            num_workers: Number of worker threads
        """
        self.analyzer = analyzer
        self.aggregator = aggregator
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.window_minutes = window_minutes
        self.num_workers = num_workers
        
        # Processing pipeline
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_queue = queue.Queue(maxsize=1000)
        self.alert_queue = queue.Queue()
        
        # Sliding windows
        self.windows = {
            '1m': deque(maxlen=60),
            '5m': deque(maxlen=300),
            '15m': deque(maxlen=900),
            '1h': deque(maxlen=3600)
        }
        
        # Stream state
        self.is_running = False
        self.workers = []
        self.metrics = StreamMetrics()
        self.processing_times = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = {
            'sentiment_spike': 0.7,
            'volume_spike': 100,
            'entity_sentiment': 0.8,
            'trend_reversal': 0.5
        }
        
        # Callbacks
        self.stream_callbacks = []
        self.alert_callbacks = []
        
        logger.info(f"Stream processor initialized with {num_workers} workers")
    
    def start_processing(self, sources: List[str]):
        """
        Start stream processing.
        
        Args:
            sources: List of sources to stream from
        """
        if self.is_running:
            logger.warning("Stream processing already running")
            return
        
        self.is_running = True
        
        # Start source streaming
        self.aggregator.start_streaming(sources)
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_process,
                name=f"SentimentWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start aggregation thread
        aggregator_thread = threading.Thread(
            target=self._aggregation_process,
            daemon=True
        )
        aggregator_thread.start()
        self.workers.append(aggregator_thread)
        
        # Start metrics thread
        metrics_thread = threading.Thread(
            target=self._metrics_process,
            daemon=True
        )
        metrics_thread.start()
        self.workers.append(metrics_thread)
        
        logger.info("Stream processing started")
    
    def stop_processing(self):
        """Stop stream processing."""
        self.is_running = False
        self.aggregator.stop_streaming()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        logger.info("Stream processing stopped")
    
    def _worker_process(self):
        """Worker process for sentiment analysis."""
        batch = []
        
        while self.is_running:
            try:
                # Collect batch
                timeout = 0.1
                while len(batch) < self.batch_size:
                    try:
                        data_point = self.input_queue.get(timeout=timeout)
                        batch.append(data_point)
                    except queue.Empty:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                start_time = time.time()
                results = self._process_batch(batch)
                processing_time = time.time() - start_time
                
                # Update metrics
                self.processing_times.append(processing_time)
                self.metrics.processed_count += len(batch)
                
                # Output results
                for result in results:
                    self.output_queue.put(result)
                    self._update_windows(result)
                    self._check_alerts(result)
                
                # Trigger callbacks
                for callback in self.stream_callbacks:
                    try:
                        callback(results)
                    except Exception as e:
                        logger.error(f"Stream callback error: {e}")
                
                # Clear batch
                batch = []
                
            except Exception as e:
                logger.error(f"Worker process error: {e}")
                self.metrics.error_count += 1
                batch = []
    
    def _process_batch(self, batch: List[SentimentDataPoint]) -> List[Dict[str, Any]]:
        """Process a batch of sentiment data points."""
        results = []
        
        for data_point in batch:
            try:
                # Extract features
                features = self.feature_extractor.extract_features(
                    data_point.text,
                    context=None  # Could add context window here
                )
                
                # Analyze sentiment with entities
                analysis = self.analyzer.analyze_with_entities(
                    data_point.text,
                    source=data_point.source,
                    author=data_point.author
                )
                
                # Combine results
                result = {
                    'timestamp': data_point.timestamp,
                    'source': data_point.source,
                    'platform': data_point.platform,
                    'author': data_point.author,
                    'text': data_point.text,
                    'sentiment': analysis['sentiment_score'],
                    'confidence': analysis['confidence'],
                    'entities': analysis.get('entities', {}),
                    'features': features,
                    'engagement': data_point.engagement,
                    'metadata': data_point.metadata
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing data point: {e}")
                self.metrics.error_count += 1
        
        return results
    
    def _aggregation_process(self):
        """Continuous aggregation from sources."""
        while self.is_running:
            try:
                # Get new data from aggregator
                new_data = self.aggregator.get_stream_data(timeout=0.5)
                
                # Add to input queue
                for data_point in new_data:
                    try:
                        self.input_queue.put(data_point, timeout=1)
                    except queue.Full:
                        logger.warning("Input queue full, dropping data point")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Aggregation process error: {e}")
    
    def _update_windows(self, result: Dict[str, Any]):
        """Update sliding windows with new result."""
        timestamp = result['timestamp']
        
        # Add to all windows
        for window in self.windows.values():
            window.append(result)
    
    def _check_alerts(self, result: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        # Sentiment spike alert
        if abs(result['sentiment']) > self.alert_thresholds['sentiment_spike']:
            alerts.append({
                'type': 'sentiment_spike',
                'timestamp': result['timestamp'],
                'sentiment': result['sentiment'],
                'source': result['source'],
                'text': result['text'][:200],
                'severity': 'high' if abs(result['sentiment']) > 0.9 else 'medium'
            })
        
        # Entity sentiment alerts
        for entity, entity_data in result.get('entities', {}).items():
            if abs(entity_data['sentiment']) > self.alert_thresholds['entity_sentiment']:
                alerts.append({
                    'type': 'entity_sentiment',
                    'timestamp': result['timestamp'],
                    'entity': entity,
                    'entity_type': entity_data['type'],
                    'sentiment': entity_data['sentiment'],
                    'mentions': entity_data['mentions'],
                    'severity': 'high'
                })
        
        # Volume spike detection (simplified)
        recent_volume = len(self.windows['5m'])
        if recent_volume > self.alert_thresholds['volume_spike']:
            alerts.append({
                'type': 'volume_spike',
                'timestamp': datetime.now(),
                'volume': recent_volume,
                'severity': 'medium'
            })
        
        # Send alerts
        for alert in alerts:
            self.alert_queue.put(alert)
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def _metrics_process(self):
        """Update processing metrics."""
        while self.is_running:
            try:
                # Calculate metrics
                if self.processing_times:
                    self.metrics.avg_processing_time = np.mean(self.processing_times)
                    
                    # Calculate throughput
                    total_time = sum(self.processing_times)
                    if total_time > 0:
                        self.metrics.throughput = self.metrics.processed_count / total_time
                
                self.metrics.queue_depth = self.input_queue.qsize()
                self.metrics.last_update = datetime.now()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics process error: {e}")
    
    def register_stream_callback(self, callback: Callable):
        """Register callback for processed stream data."""
        self.stream_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_window_statistics(self, window: str = '5m') -> Dict[str, Any]:
        """Get statistics for a time window."""
        if window not in self.windows:
            return {}
        
        window_data = list(self.windows[window])
        if not window_data:
            return {}
        
        # Calculate statistics
        sentiments = [d['sentiment'] for d in window_data]
        confidences = [d['confidence'] for d in window_data]
        
        # Entity statistics
        entity_counts = defaultdict(int)
        entity_sentiments = defaultdict(list)
        
        for data in window_data:
            for entity, entity_data in data.get('entities', {}).items():
                entity_counts[entity] += 1
                entity_sentiments[entity].append(entity_data['sentiment'])
        
        # Source distribution
        source_counts = defaultdict(int)
        for data in window_data:
            source_counts[data['source']] += 1
        
        return {
            'window': window,
            'data_points': len(window_data),
            'avg_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'avg_confidence': np.mean(confidences),
            'positive_ratio': sum(1 for s in sentiments if s > 0.3) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < -0.3) / len(sentiments),
            'top_entities': sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'entity_sentiments': {
                entity: np.mean(sents) 
                for entity, sents in entity_sentiments.items()
            },
            'source_distribution': dict(source_counts),
            'time_range': {
                'start': window_data[0]['timestamp'],
                'end': window_data[-1]['timestamp']
            }
        }
    
    def get_real_time_sentiment(self, entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get real-time sentiment analysis."""
        # Get latest data from multiple windows
        windows_data = {}
        for window_name, window_data in self.windows.items():
            if window_data:
                windows_data[window_name] = self.get_window_statistics(window_name)
        
        # Filter by entities if specified
        if entities:
            for window_name, stats in windows_data.items():
                if 'entity_sentiments' in stats:
                    stats['entity_sentiments'] = {
                        e: s for e, s in stats['entity_sentiments'].items()
                        if e in entities
                    }
        
        # Get current metrics
        metrics = {
            'processed_count': self.metrics.processed_count,
            'error_rate': self.metrics.error_count / max(self.metrics.processed_count, 1),
            'avg_processing_time': self.metrics.avg_processing_time,
            'throughput': self.metrics.throughput,
            'queue_depth': self.metrics.queue_depth,
            'last_update': self.metrics.last_update
        }
        
        return {
            'timestamp': datetime.now(),
            'windows': windows_data,
            'metrics': metrics,
            'alerts_pending': self.alert_queue.qsize()
        }
    
    def get_sentiment_velocity(self, entity: Optional[str] = None, 
                             window: str = '5m') -> Dict[str, float]:
        """Calculate rate of sentiment change."""
        if window not in self.windows:
            return {}
        
        window_data = list(self.windows[window])
        if len(window_data) < 10:
            return {}
        
        # Extract sentiment time series
        if entity:
            sentiments = []
            timestamps = []
            
            for data in window_data:
                if entity in data.get('entities', {}):
                    sentiments.append(data['entities'][entity]['sentiment'])
                    timestamps.append(data['timestamp'])
        else:
            sentiments = [d['sentiment'] for d in window_data]
            timestamps = [d['timestamp'] for d in window_data]
        
        if len(sentiments) < 10:
            return {}
        
        # Calculate velocity (rate of change)
        recent = np.mean(sentiments[-5:])
        older = np.mean(sentiments[-10:-5])
        velocity = recent - older
        
        # Calculate acceleration
        if len(sentiments) >= 15:
            oldest = np.mean(sentiments[-15:-10])
            acceleration = (recent - older) - (older - oldest)
        else:
            acceleration = 0.0
        
        return {
            'entity': entity,
            'window': window,
            'current_sentiment': sentiments[-1],
            'avg_sentiment': np.mean(sentiments),
            'velocity': velocity,
            'acceleration': acceleration,
            'trend': 'increasing' if velocity > 0.1 else 'decreasing' if velocity < -0.1 else 'stable'
        }