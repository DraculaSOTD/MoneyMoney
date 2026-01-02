import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter


class SimpleTokenizer:
    """
    Simple tokenizer for cryptocurrency sentiment analysis.
    
    Features:
    - Crypto-specific vocabulary
    - Subword tokenization
    - Special token handling
    - Emoji support
    """
    
    def __init__(self, vocab_size: int = 10000,
                 min_freq: int = 2,
                 max_length: int = 128):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for vocabulary inclusion
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.mask_token = '<MASK>'
        
        # Initialize vocabulary
        self.word_to_idx = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sos_token: 2,
            self.eos_token: 3,
            self.mask_token: 4
        }
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        # Crypto-specific vocabulary
        self.crypto_terms = self._load_crypto_vocabulary()
        
        # Emoji sentiment mapping
        self.emoji_sentiment = self._load_emoji_sentiment()
        
    def _load_crypto_vocabulary(self) -> List[str]:
        """Load cryptocurrency-specific terms."""
        return [
            # Price movements
            'moon', 'mooning', 'pump', 'dump', 'crash', 'surge', 'rally',
            'bearish', 'bullish', 'dip', 'correction', 'breakout', 'resistance',
            'support', 'consolidation', 'accumulation', 'distribution',
            
            # Trading terms
            'hodl', 'buy', 'sell', 'long', 'short', 'leverage', 'liquidation',
            'margin', 'stop-loss', 'take-profit', 'rekt', 'bag', 'whale',
            'fomo', 'fud', 'diamond hands', 'paper hands',
            
            # Technical terms
            'blockchain', 'defi', 'nft', 'smart contract', 'gas', 'fees',
            'halving', 'mining', 'staking', 'yield', 'apy', 'tvl',
            
            # Coins/tokens
            'btc', 'bitcoin', 'eth', 'ethereum', 'alt', 'altcoin',
            'stable', 'stablecoin', 'shitcoin', 'memecoin',
            
            # Market sentiment
            'bullrun', 'bear market', 'alt season', 'capitulation',
            'euphoria', 'fear', 'greed', 'panic', 'optimistic', 'pessimistic'
        ]
    
    def _load_emoji_sentiment(self) -> Dict[str, float]:
        """Load emoji to sentiment mapping."""
        return {
            # Positive
            '🚀': 0.9, '🌙': 0.8, '💎': 0.7, '🔥': 0.6, '💪': 0.6,
            '📈': 0.8, '💰': 0.7, '🤑': 0.8, '😍': 0.7, '🎉': 0.7,
            
            # Negative
            '📉': -0.8, '💩': -0.7, '😱': -0.6, '😭': -0.7, '🤮': -0.8,
            '⚠️': -0.5, '🆘': -0.7, '💔': -0.6, '😰': -0.6, '📊': -0.5,
            
            # Neutral
            '🤔': 0.0, '😐': 0.0, '📊': 0.0, '👀': 0.1, '🤷': 0.0
        }
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text samples
        """
        # Count word frequencies
        word_freq = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            word_freq.update(tokens)
            
        # Add crypto terms with high priority
        for term in self.crypto_terms:
            word_freq[term] = max(word_freq.get(term, 0), self.min_freq)
            
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        for word, freq in sorted_words:
            if freq >= self.min_freq and len(self.word_to_idx) < self.vocab_size:
                if word not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle URLs
        text = re.sub(r'http\S+', '<URL>', text)
        
        # Handle mentions
        text = re.sub(r'@\w+', '<MENTION>', text)
        
        # Handle numbers (prices, percentages)
        text = re.sub(r'\$?\d+\.?\d*[km]?%?', '<NUMBER>', text)
        
        # Extract emojis
        emojis = re.findall(r'[^\w\s,.\'-]', text)
        
        # Basic word tokenization
        words = re.findall(r'\b\w+\b|<\w+>', text)
        
        # Add emojis back
        tokens = words + emojis
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode text to token indices.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            Array of token indices
        """
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.sos_token] + tokens + [self.eos_token]
            
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx[self.unk_token])
                
        # Pad to max length
        while len(indices) < self.max_length:
            indices.append(self.word_to_idx[self.pad_token])
            
        return np.array(indices)
    
    def decode(self, indices: np.ndarray, skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text.
        
        Args:
            indices: Array of token indices
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for idx in indices:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                
                if skip_special_tokens and token in [self.pad_token, self.sos_token, 
                                                     self.eos_token, self.mask_token]:
                    continue
                    
                tokens.append(token)
                
        return ' '.join(tokens)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Batch of encoded sequences
        """
        batch = []
        
        for text in texts:
            encoded = self.encode(text)
            batch.append(encoded)
            
        return np.array(batch)
    
    def get_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment features
        """
        tokens = self.tokenize(text)
        
        features = {
            'emoji_sentiment': 0.0,
            'bullish_score': 0.0,
            'bearish_score': 0.0,
            'fomo_fud_score': 0.0,
            'technical_score': 0.0
        }
        
        # Emoji sentiment
        emoji_scores = []
        for token in tokens:
            if token in self.emoji_sentiment:
                emoji_scores.append(self.emoji_sentiment[token])
                
        if emoji_scores:
            features['emoji_sentiment'] = np.mean(emoji_scores)
            
        # Bullish/bearish keywords
        bullish_keywords = ['moon', 'pump', 'bullish', 'rally', 'breakout', 
                           'long', 'buy', 'hodl', 'bullrun']
        bearish_keywords = ['dump', 'crash', 'bearish', 'short', 'sell',
                           'correction', 'rekt', 'capitulation']
        
        features['bullish_score'] = sum(1 for t in tokens if t in bullish_keywords) / len(tokens)
        features['bearish_score'] = sum(1 for t in tokens if t in bearish_keywords) / len(tokens)
        
        # FOMO/FUD score
        fomo_keywords = ['fomo', 'moon', 'pump', 'missing out', 'last chance']
        fud_keywords = ['fud', 'fear', 'panic', 'crash', 'scam']
        
        fomo_count = sum(1 for t in tokens if t in fomo_keywords)
        fud_count = sum(1 for t in tokens if t in fud_keywords)
        features['fomo_fud_score'] = (fomo_count - fud_count) / max(1, len(tokens))
        
        # Technical analysis mentions
        technical_keywords = ['support', 'resistance', 'breakout', 'pattern',
                            'indicator', 'rsi', 'macd', 'volume']
        features['technical_score'] = sum(1 for t in tokens if t in technical_keywords) / len(tokens)
        
        return features
    
    def create_attention_mask(self, indices: np.ndarray) -> np.ndarray:
        """
        Create attention mask for padded sequences.
        
        Args:
            indices: Token indices
            
        Returns:
            Attention mask (1 for real tokens, 0 for padding)
        """
        return (indices != self.word_to_idx[self.pad_token]).astype(np.float32)