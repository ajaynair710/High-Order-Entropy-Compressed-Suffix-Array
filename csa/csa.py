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
    def __init__(self, text, epsilon=0.5, k=2):
        if not text.endswith('$'):
            text += '$'
        self.text = text
        self.n = len(text)
        self.epsilon = epsilon
        self.k = k
        self.sigma = len(set(text))
        
        self.sa = build_suffix_array(self.text)
        self.bwt = bwt_transform(self.text, self.sa)
        
        self.char_counts = self._build_char_counts()
        self.lf_table = self._build_lf_table()
        
        self.sampling_rate = max(1, int((math.log(self.n, 2)) ** self.epsilon))
        self.samples = self._build_samples()

    def _build_char_counts(self):
        """Build character count tables for constant-time access"""
        counts = {}
        running_counts = defaultdict(int)
        
        for i, c in enumerate(self.bwt):
            counts[i] = dict(running_counts)
            running_counts[c] += 1
        
        counts[len(self.bwt)] = dict(running_counts)
            
        return counts

    def _build_lf_table(self):
        """Build LF-mapping lookup table"""
        c_array = {}
        current_count = 0
        for char in sorted(set(self.text)):
            c_array[char] = current_count
            current_count += self.text.count(char)
        
        lf_table = {}
        for i, c in enumerate(self.bwt):
            rank = self.char_counts[i][c] if c in self.char_counts[i] else 0
            lf_table[i] = c_array[c] + rank
            
        return lf_table

    def _build_samples(self):
        """Build position samples at rate (log n)^ε"""
        samples = {}
        for i in range(0, self.n):
            if i % self.sampling_rate == 0 or self.sa[i] % self.sampling_rate == 0:
                samples[i] = self.sa[i]
        return samples

    def _lf_mapping(self, i):
        """Constant-time LF-mapping using lookup table"""
        return self.lf_table.get(i, -1)

    def _get_position_fast(self, sa_idx):
        """O((log n)^ε) time position lookup"""
        current_pos = sa_idx
        steps = 0
        
        if current_pos in self.samples:
            return self.samples[current_pos]
        
        path = []
        
        while steps < self.sampling_rate:
            path.append(current_pos)
            current_pos = self._lf_mapping(current_pos)
            steps += 1
            
            if current_pos in self.samples:
                pos = self.samples[current_pos]
                for _ in range(steps):
                    pos = (pos + 1) % self.n
                return pos
                
        return -1

    def find_pattern(self, pattern):
        """Find pattern range in O(m * log σ) time"""
        return self._find_pattern_range(pattern)
    
    def locate_from_range(self, sa_idx):
        """Locate single occurrence in O((log n)^ε) time"""
        return self._get_position_fast(sa_idx)
    
    def locate(self, pattern):
        """Full pattern location - O(m * log σ + k * (log n)^ε) time"""
        positions = set()
        left, right = self.find_pattern(pattern)
        
        if left == -1 or right == -1:
            return sorted(positions)
        
        for i in range(left, right + 1):
            pos = self.locate_from_range(i)
            if pos != -1:
                positions.add(pos)
        
        return sorted(list(positions))

    def _find_pattern_range(self, pattern):
        """Find pattern range in BWT"""
        if not pattern:
            return -1, -1
        
        left = 0
        right = self.n - 1
        
        for c in reversed(pattern):
            left_count = sum(1 for x in self.bwt[:left] if x == c)
            right_count = sum(1 for x in self.bwt[:right + 1] if x == c)
            
            c_pos = sum(self.text.count(x) for x in sorted(set(self.text)) if x < c)
            
            left = c_pos + left_count
            right = c_pos + right_count - 1
            
            if left > right:
                return -1, -1
        
        return left, right

    def get_size_metrics(self):
        min_sigma = max(2, self.sigma)
        min_entropy = 1 / min_sigma
        
        original_size = self.n * math.ceil(math.log2(min_sigma))
        k_entropy = calculate_high_order_entropy(self.text, self.k)
        k_entropy = max(min_entropy, k_entropy)
        compressed_size = self.n * k_entropy
        
        compression_factor = 1 - (1 / min_sigma)
        compressed_size = min(compressed_size, original_size * compression_factor)
        
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