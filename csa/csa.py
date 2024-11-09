# csa.py
import math
from csa.suffix_array import build_suffix_array
from csa.bwt import bwt_transform
from utils.utils import build_occ
from utils.utils import build_count
from csa.high_order_entropy import calculate_high_order_entropy
from csa.enhanced_fm_index import EnhancedFMIndex
from csa.wavelet_tree import WaveletTree
from collections import Counter
from collections import defaultdict
from bitarray import bitarray
import numpy as np

class CompressedSuffixArray:
    def __init__(self, text, epsilon=0.5, k=5):
        """
        Initialize High-Entropy Compressed Suffix Array
        Args:
            text: Input text
            epsilon: Sampling rate for Ψ values (controls space/time tradeoff)
            k: Order of entropy compression
        """
        # Add end marker if not present
        if text[-1] != '$':
            text = text + '$'
        
        self.text = text
        self.n = len(text)
        self.k = k
        self.epsilon = epsilon
        self.sigma = len(set(text))
        
        # Initialize basic structures needed regardless of compression
        # Use the correct suffix array construction
        self.sa = build_suffix_array(text)  # This should now give [6,5,3,1,0,4,2] for "banana$"
        self.bwt = bwt_transform(text, self.sa)
        self.occ = build_occ(self.bwt)
        
        # Don't compress if text is too small
        self.compression_threshold = 100
        self.use_compression = self.n >= self.compression_threshold
        
        if self.use_compression:
            # Build additional compressed structures only if text is long enough
            self.count = build_count(self.bwt)
            self.wavelet_tree = WaveletTree(self.bwt)
            self.psi_samples = self._sample_psi()
        
        # Initialize count table correctly
        self.count = {}
        # First count occurrences
        for char in text:
            self.count[char] = self.count.get(char, 0) + 1
        
        # Convert to cumulative positions
        cumsum = 0
        temp_count = {}
        for char in sorted(self.count.keys()):
            temp_count[char] = cumsum
            cumsum += self.count[char]
        self.count = temp_count
        
        # Add new sampling parameters
        self.sample_rate = max(1, int(math.log2(self.n) ** epsilon))
        self.sa_samples = self._sample_suffix_array()
        self.marked_positions = self._mark_sampled_positions()
        
        # Initialize rank dictionary
        self.build_rank_dictionary()
        
        # Add debug prints
        
        # Optimize parameters for better compression
        self.block_size = max(1, int(math.log2(self.n)))
        self.run_length_threshold = 4  # For run-length encoding
        self.context_order = min(k, int(0.5 * math.log(self.n, self.sigma)))
        
        # Enhanced compression structures
        if self.use_compression:
            self.run_lengths = self._compute_run_lengths()
            self.context_stats = self._build_context_statistics()
            self.compressed_bwt = self._compress_bwt()
        
    def _sample_psi(self):
        """Sample Ψ values at rate epsilon"""
        sample_interval = max(1, int(1/self.epsilon))
        samples = {}
        for i in range(0, self.n, sample_interval):
            samples[i] = self._compute_psi(i)
        return samples

    def _compute_psi(self, i):
        """Compute Ψ(i) using FM-index operations"""
        if i >= self.n:
            return 0
        char = self.bwt[i]
        # Count occurrences up to position i in the occ dictionary
        rank = sum(1 for j in range(i + 1) if self.bwt[j] == char)
        return self.count[char] + rank

    def lookup(self, i):
        """Lookup Ψ(i) using sampled values and wavelet tree"""
        if i in self.psi_samples:
            return self.psi_samples[i]
            
        # Find nearest previous sample
        prev_sample = (i // int(1/self.epsilon)) * int(1/self.epsilon)
        steps = i - prev_sample
        
        # Follow Ψ chain from sample
        pos = self.psi_samples[prev_sample]
        for _ in range(steps):
            pos = self._compute_psi(pos)
        return pos

    def calculate_high_order_entropy(self, text, k):
        """Calculate k-th order empirical entropy with improved context modeling"""
        if not text or k < 0:
            return 0
            
        n = len(text)
        sigma = len(set(text))
        
        # Adjust k based on text statistics
        max_k = min(k, int(math.log(n, sigma)))
        
        # Initialize context statistics
        contexts = defaultdict(Counter)
        total_entropy = 0
        
        # Process overlapping blocks for better context coverage
        for i in range(n - max_k):
            context = text[i:i+max_k]
            next_char = text[i+max_k]
            contexts[context][next_char] += 1
        
        # Calculate conditional entropy for each context
        total_chars = n - max_k
        for context, char_counts in contexts.items():
            context_total = sum(char_counts.values())
            if context_total > 0:
                # Calculate conditional entropy H(X|context)
                context_entropy = 0
                for count in char_counts.values():
                    p = count / context_total
                    context_entropy -= p * math.log2(p)
                # Weight by context frequency
                total_entropy += (context_total / total_chars) * context_entropy
        
        # Apply correction factor for shorter contexts
        if max_k > 0:
            correction = (max_k * math.log2(sigma)) / n
            total_entropy = max(total_entropy - correction, 0)
        
        return total_entropy

    def _calculate_compressed_size(self):
        """Calculate compressed size with enhanced compression"""
        if not self.use_compression:
            return super()._calculate_compressed_size()
        
        n = self.n
        sigma = self.sigma
        
        # Calculate effective k for entropy
        effective_k = min(self.k, int(0.5 * math.log(n, sigma)))
        
        # Base entropy calculation
        hk = self.calculate_high_order_entropy(self.text, effective_k)
        main_space = n * hk
        
        # Run-length compression savings
        run_savings = sum(length - 1 for _, length in self.run_lengths)
        run_overhead = len(self.run_lengths) * (math.log2(n) + math.log2(sigma))
        
        # Context model compression
        context_savings = 0
        context_overhead = 0
        for k, contexts in self.context_stats.items():
            for context, counts in contexts.items():
                total = sum(counts.values())
                if total > 0:
                    best_prob = max(count/total for count in counts.values())
                    if best_prob > 0.8:  # High probability threshold
                        context_savings += total * (1 - math.log2(1/best_prob))
                        context_overhead += len(context) * math.log2(sigma)
        
        # Calculate total size with all optimizations
        total_size = (
            main_space * (1 + 1/math.log2(n)) -  # Base size
            run_savings + run_overhead +          # Run-length adjustment
            context_savings + context_overhead    # Context model adjustment
        )
        
        return total_size

    def _estimate_average_run_length(self):
        """Estimate average run length in BWT"""
        if not self.bwt:
            return 1
            
        runs = 1
        for i in range(1, len(self.bwt)):
            if self.bwt[i] != self.bwt[i-1]:
                runs += 1
        
        return len(self.bwt) / max(1, runs)

    def _calculate_empirical_entropy(self):
        """Calculate the empirical entropy (H_0) of the text"""
        if self.n == 0:
            return 0
            
        # Count character frequencies
        freq = Counter(self.text)
        
        # Calculate entropy using Shannon's formula
        entropy = 0
        for count in freq.values():
            p = count / self.n
            entropy -= p * math.log2(p)
            
        return entropy

    def get_size_metrics(self):
        """Returns size metrics with tighter theoretical bounds"""
        original_size = self.n * math.ceil(math.log2(self.n))
        
        # Calculate theoretical minimum following paper
        k = min(self.k, int(0.5 * math.log(self.n, self.sigma)))
        entropy = self.calculate_high_order_entropy(self.text, k)
        theoretical_minimum = self.n * entropy
        
        # Calculate actual compressed size
        compressed_size = self._calculate_compressed_size()
        
        # Add minimal overhead as per paper
        minimal_overhead = (self.n * math.log2(self.sigma)) / (math.log2(self.n) * math.log2(math.log2(self.n)))
        theoretical_minimum += minimal_overhead
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / original_size if original_size > 0 else 0,
            'space_saving': 1 - (compressed_size / original_size) if original_size > 0 else 0,
            'entropy_efficiency': compressed_size / theoretical_minimum if theoretical_minimum > 0 else 0,
            'entropy_order': k,
            'bits_per_symbol': compressed_size / self.n if self.n > 0 else 0
        }

    def _sample_suffix_array(self):
        """Sample suffix array at rate (log n)^ε"""
        samples = {}
        for i in range(0, self.n, self.sample_rate):
            if i < len(self.sa):  # Add bounds check
                samples[i] = self.sa[i]
        return samples

    def _mark_sampled_positions(self):
        """Create bitvector marking sampled positions"""
        marked = [0] * self.n
        for pos in self.sa_samples.keys():
            marked[pos] = 1
        return marked

    def locate(self, pattern):
        """Locate all occurrences of pattern with (log n)^ε time per occurrence"""
        # First find the interval in BWT
        left, right = self._find_interval(pattern)
        if left > right:
            print("Pattern not found")
            return []
        
        # print(f"Found interval [{left}, {right}] in BWT")
        
        # Use sampled positions and LF-mapping for faster location
        occurrences = []
        for i in range(left, right + 1):
            pos = self._locate_single(i)
            if pos != -1:
                occurrences.append(pos)
        
        return sorted(occurrences)

    def _locate_single(self, bwt_pos):
        """Locate single occurrence using sampled positions and LF-mapping"""
        steps = 0
        current_pos = bwt_pos
        
        # Follow LF-mapping until we hit a sampled position or exceed sample rate
        while steps < self.sample_rate:
            if current_pos in self.sa_samples:
                # Found a sample, calculate original position
                return (self.sa_samples[current_pos] + steps) % self.n
                
            # Move to next position using LF-mapping
            current_pos = self._lf_mapping(current_pos)
            steps += 1
        
        # If we haven't found a sample, use sparser sampling
        while steps < self.n:  # Safety limit
            if current_pos in self.sa_samples:
                return (self.sa_samples[current_pos] + steps) % self.n
            current_pos = self._lf_mapping(current_pos)
            steps += 1
        
        return -1

    def _lf_mapping(self, i):
        """Compute LF-mapping using rank dictionary"""
        if i >= self.n:
            return 0
        char = self.bwt[i]
        rank = self.rank_dict[char][i + 1]
        return self.count[char] + rank - 1

    def _find_interval(self, pattern):
        """Find interval in BWT containing pattern occurrences"""
        if not pattern:
            return 0, -1
        
        # Start with last character of pattern
        char = pattern[-1]
        if char not in self.count:
            # print(f"Character '{char}' not found in count table!")
            return 0, -1
        
        # Initialize interval
        left = self.count[char]
        right = self.n - 1
        for c in sorted(self.count.keys()):
            if c > char:
                right = self.count[c] - 1
                break
        
        # print(f"Initial interval for '{char}': [{left}, {right}]")
        
        # Process pattern right to left
        for i in range(len(pattern) - 2, -1, -1):
            char = pattern[i]
            if char not in self.count:
                print(f"Character '{char}' not found in count table!")
                return 0, -1
            
            # Calculate new interval
            new_left = self.count[char]
            if left > 0:
                new_left += sum(1 for j in range(left) if self.bwt[j] == char)
            
            new_right = self.count[char] + sum(1 for j in range(right + 1) if self.bwt[j] == char) - 1
            
            # print(f"Updated interval for '{char}': [{new_left}, {new_right}]")
            
            left, right = new_left, new_right
            if left > right:
                # print(f"Invalid interval: [{left}, {right}]")
                return 0, -1
        
        # print(f"Final interval: [{left}, {right}]")
        return left, right

    def build_rank_dictionary(self):
        """Build rank dictionary for constant-time rank queries"""
        self.rank_dict = {}
        for char in set(self.bwt):
            self.rank_dict[char] = [0]
            count = 0
            for c in self.bwt:
                if c == char:
                    count += 1
                self.rank_dict[char].append(count)

    def _compute_run_lengths(self):
        """Compute run lengths in BWT for better compression"""
        runs = []
        current_run = 1
        
        for i in range(1, len(self.bwt)):
            if self.bwt[i] == self.bwt[i-1]:
                current_run += 1
            else:
                if current_run >= self.run_length_threshold:
                    runs.append((self.bwt[i-1], current_run))
                current_run = 1
                
        return runs
        
    def _build_context_statistics(self):
        """Build enhanced context statistics for higher-order compression"""
        contexts = defaultdict(lambda: defaultdict(Counter))
        n = len(self.text)
        
        # Process multiple context lengths simultaneously
        for k in range(1, self.context_order + 1):
            # Use sliding window for context gathering
            window_size = k + 1  # context + next char
            for i in range(n - window_size + 1):
                context = self.text[i:i+k]
                next_char = self.text[i+k]
                
                # Store both forward and reverse contexts
                contexts[k][context][next_char] += 1
                rev_context = context[::-1]
                contexts[k][rev_context][next_char] += 1
                
                # Add partial contexts for better modeling
                if k > 1:
                    for j in range(1, k):
                        partial = context[j:]
                        contexts[k-j][partial][next_char] += 1
        
        # Prune ineffective contexts
        pruned_contexts = defaultdict(lambda: defaultdict(Counter))
        min_context_freq = max(5, int(math.log2(n)))
        
        for k, k_contexts in contexts.items():
            for context, char_counts in k_contexts.items():
                total_count = sum(char_counts.values())
                if total_count >= min_context_freq:
                    # Calculate context effectiveness
                    max_prob = max(count/total_count for count in char_counts.values())
                    if max_prob > 0.5:  # Keep contexts with good prediction power
                        pruned_contexts[k][context] = char_counts
        
        return pruned_contexts
        
    def _compress_bwt(self):
        """Compress BWT using enhanced context modeling"""
        compressed = []
        i = 0
        
        while i < len(self.bwt):
            # Check for runs first
            run_length = 1
            while i + run_length < len(self.bwt) and self.bwt[i + run_length] == self.bwt[i]:
                run_length += 1
                
            if run_length >= self.run_length_threshold:
                compressed.append(('R', self.bwt[i], run_length))
                i += run_length
                continue
            
            # Try context-based compression
            best_prediction = None
            best_confidence = 0
            
            # Check all available context lengths
            for k in range(min(self.context_order, i), 0, -1):
                context = self.text[i-k:i]
                if context in self.context_stats[k]:
                    counts = self.context_stats[k][context]
                    total = sum(counts.values())
                    if total > 0:
                        for char, count in counts.items():
                            prob = count / total
                            if prob > best_confidence:
                                best_confidence = prob
                                best_prediction = (k, context, char)
            
            if best_prediction and best_confidence > 0.7:
                compressed.append(('C', best_prediction))
            else:
                compressed.append(('L', self.bwt[i]))
            i += 1
        
        return compressed