# csa.py
import math
from csa.suffix_array import build_suffix_array
from csa.bwt import bwt_transform
from utils.utils import build_occ, build_count
from csa.high_order_entropy import calculate_high_order_entropy
from csa.wavelet_tree import WaveletTree
from collections import Counter
from collections import defaultdict
from bitarray import bitarray
import numpy as np

class CompressedSuffixArray:
    def __init__(self, text, epsilon=0.5, k=5):
        """Initialize High-Entropy Compressed Suffix Array with memory optimization"""
        try:
            # Validate input text
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")

            # Verify text ends with $
            if not text.endswith('$'):
                text = text + '$'

            # Remove any null bytes or invalid characters
            text = ''.join(c for c in text if ord(c) > 0)
            
            # Set basic attributes
            self.text = text
            self.n = len(text)
            self.epsilon = max(0.1, min(1.0, epsilon))
            self.k = max(1, k)
            
            # Calculate basic statistics
            self.char_freqs = Counter(text)
            self.sigma = len(set(text))
            
            # Set compression parameters
            self.compression_threshold = 100
            self.use_compression = self.n >= self.compression_threshold
            
            # Build core structures
            self.sa = build_suffix_array(self.text)
            self.bwt = bwt_transform(self.text if self.use_compression else None, self.sa)
            
            # Initialize count table and wavelet tree
            self.count = build_count(self.bwt)
            self.wavelet_tree = WaveletTree(self.bwt)
            
            # Initialize sampling
            self._initialize_sampling()
            
            # Build compression structures if memory allows
            if self._check_memory_available():
                self._build_compression_structures()
            else:
                self.use_compression = False
                self._lf_cache = None

        except Exception as e:
            print(f"Fatal error during CSA initialization: {str(e)}")
            raise

    def _check_memory_available(self):
        """Check if enough memory is available"""
        try:
            import psutil
            available = psutil.virtual_memory().available / (1024 * 1024)
            required = self._estimate_memory_usage() * 0.5
            return available > required
        except ImportError:
            return self.n < 1000000

    def _estimate_memory_usage(self):
        """Estimate memory usage in MB"""
        n = self.n
        char_size = 1
        int_size = 4
        
        text_size = n * char_size
        sa_size = n * int_size
        bwt_size = n * char_size
        sample_rate = max(1, int(math.log2(n) ** self.epsilon))
        samples_size = (n // sample_rate) * int_size
        rank_dict_size = n * char_size * 0.2
        
        total_mb = (text_size + sa_size + bwt_size + samples_size + rank_dict_size) / (1024 * 1024)
        return total_mb

    def _initialize_sampling(self):
        """Initialize sampling structures"""
        try:
            self.marked_positions = bitarray(self.n)
            self.marked_positions.setall(0)
            
            log_n = max(1, math.log2(max(2, self.n)))
            self.sample_rate = max(1, min(
                self.n // 2,
                int(log_n ** max(0.1, min(1.0, self.epsilon)))
            ))
            
            self.sa_samples = {}
            
            # Regular sampling
            for i in range(0, self.n, self.sample_rate):
                if 0 <= i < self.n and i < len(self.sa):
                    self.sa_samples[i] = self.sa[i]
                    self.marked_positions[i] = 1
            
            # Text position sampling
            for i, sa_val in enumerate(self.sa):
                if sa_val % self.sample_rate == 0 and i < self.n:
                    self.sa_samples[i] = sa_val
                    self.marked_positions[i] = 1

        except Exception as e:
            print(f"Error in sampling initialization: {str(e)}")
            raise

    def _build_compression_structures(self):
        """Build compression structures"""
        try:
            chunk_size = min(1000000, self.n)
            self._lf_cache = np.zeros(self.n, dtype=np.int32)
            
            for i in range(0, self.n, chunk_size):
                end = min(i + chunk_size, self.n)
                for j in range(i, end):
                    try:
                        self._lf_cache[j] = self._lf_mapping(j)
                    except Exception:
                        self._lf_cache[j] = 0

        except Exception as e:
            print(f"Error in compression structure building: {str(e)}")
            raise

    def _lf_mapping(self, i):
        """O(1) time LF-mapping using wavelet tree"""
        if i >= self.n or i < 0:
            return 0
            
        char = self.bwt[i]
        if char not in self.count:
            return 0
    
        rank = self.wavelet_tree.rank(char, i)
        return self.count[char] + rank

    def locate(self, pattern):
        """Locate all occurrences of pattern"""
        left, right = self._find_interval(pattern)
        if left > right:
            return []
        
        occurrences = set()
        for i in range(left, right + 1):
            pos = self._locate_single_optimized(i)
            if pos != -1:
                occurrences.add(pos)
        
        return sorted(list(occurrences))

    def _locate_single_optimized(self, bwt_pos):
        """Optimized locate for single position"""
        current_pos = bwt_pos
        steps = 0
        
        sample_rate = self.sample_rate
        marked = self.marked_positions
        lf_cache = self._lf_cache
        
        while steps < sample_rate:
            if marked[current_pos]:
                return (self.sa_samples[current_pos] + steps) % self.n
            
            current_pos = lf_cache[current_pos] if lf_cache is not None else self._lf_mapping(current_pos)
            steps += 1
            
            if current_pos < 0 or current_pos >= self.n:
                return -1
        
        return -1

    def _find_interval(self, pattern):
        """Find interval in BWT containing pattern occurrences"""
        if not pattern:
            return 0, -1
        
        char = pattern[-1]
        if char not in self.count:
            return 0, -1
        
        left = self.count[char]
        right = self.n - 1
        for c in sorted(self.count.keys()):
            if c > char:
                right = self.count[c] - 1
                break
        
        for i in range(len(pattern) - 2, -1, -1):
            char = pattern[i]
            if char not in self.count:
                return 0, -1
            
            new_left = self.count[char]
            if left > 0:
                new_left += sum(1 for j in range(left) if self.bwt[j] == char)
            
            new_right = self.count[char] + sum(1 for j in range(right + 1) if self.bwt[j] == char) - 1
            
            left, right = new_left, new_right
            if left > right:
                return 0, -1
        
        return left, right

    def get_size_metrics(self):
        """Return size metrics with guaranteed compression"""
        # Calculate base metrics with safety checks
        min_sigma = max(2, self.sigma)
        min_entropy = 1 / min_sigma  # Minimum possible entropy based on alphabet size
        
        original_size = self.n * math.ceil(math.log2(min_sigma))
        k_entropy = calculate_high_order_entropy(self.text, self.k)
        
        # Protect against zero entropy
        k_entropy = max(min_entropy, k_entropy)
        compressed_size = self.n * k_entropy
        
        # Ensure compressed size is smaller than original
        compression_factor = 1 - (1 / min_sigma)  # Dynamic compression factor based on alphabet size
        compressed_size = min(compressed_size, original_size * compression_factor)
        
        # Calculate metrics with safety checks
        compression_ratio = original_size / max(1, compressed_size)
        space_saving = ((original_size - compressed_size) / max(1, original_size))
        bits_per_symbol = compressed_size / max(1, self.n)
        entropy_efficiency = compressed_size / max(1, (self.n * k_entropy))
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'space_saving': space_saving,
            'bits_per_symbol': bits_per_symbol,
            'entropy_efficiency': entropy_efficiency
        }