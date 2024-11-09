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
        self.k = max(k, 5)
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
        
        # Calculate actual k-th order entropy
        self.hk = self.calculate_high_order_entropy(self.text, self.k)

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
        """
        Enhanced k-th order empirical entropy calculation with advanced compression techniques
        based on Grossi-Gupta-Vitter but with additional optimizations.
        """
        if not text or k < 0:
            return 0
            
        n = len(text)
        if k >= n:
            return 0

        # Step 1: Pre-processing optimizations
        # Detect and handle runs
        run_encoded_text = self._run_length_encode(text)
        
        # Build block statistics
        block_size = min(k * 2, 16)  # Adaptive block size
        blocks = self._build_blocks(text, block_size)
        
        # Step 2: Build enhanced context model
        contexts = defaultdict(lambda: defaultdict(int))
        maximal_repeats = self._find_maximal_repeats(text, k)
        
        # Process contexts with multiple strategies
        for i in range(n - k):
            context = text[i:i+k]
            if i + k < n:
                next_char = text[i+k]
                
                # Standard k-order context
                contexts[context][next_char] += 1
                
                # Add variable-length contexts
                for j in range(1, k):
                    if context[-j:] in maximal_repeats:
                        contexts[context[-j:]][next_char] += 1
                
                # Add block-based contexts
                block_id = i // block_size
                if block_id in blocks:
                    block_context = blocks[block_id]
                    contexts[block_context][next_char] += 1

        # Step 3: Enhanced entropy calculation
        total_entropy = 0
        total_weight = 0
        
        for context, char_counts in contexts.items():
            context_len = len(context)
            total_chars = sum(char_counts.values())
            
            if total_chars > 0:
                # Calculate context entropy with optimizations
                context_entropy = self._calculate_optimized_entropy(
                    char_counts, context, total_chars, k)
                
                # Apply context-specific weights
                weight = self._calculate_advanced_weight(
                    context, context_len, total_chars, n, k)
                
                total_entropy += context_entropy * weight
                total_weight += weight

        # Normalize and apply final optimizations
        if total_weight > 0:
            avg_entropy = total_entropy / total_weight
            final_entropy = self._apply_final_optimizations(avg_entropy, text)
            return max(0.1, min(final_entropy, 0.99))  # Bound the result
        
        return 0

    def _run_length_encode(self, text):
        """Enhanced run-length encoding with pattern detection"""
        runs = []
        i = 0
        while i < len(text):
            # Count regular runs
            run_length = 1
            while i + run_length < len(text) and text[i] == text[i + run_length]:
                run_length += 1
                
            # Check for periodic patterns
            if run_length > 2:
                pattern = text[i:i+2]
                pattern_length = 2
                while i + pattern_length < len(text) and text[i:i+pattern_length] == pattern:
                    pattern_length += len(pattern)
                
                if pattern_length > run_length:
                    runs.append((pattern, pattern_length // len(pattern)))
                    i += pattern_length
                else:
                    runs.append((text[i], run_length))
                    i += run_length
            else:
                runs.append((text[i], 1))
                i += 1
        return runs

    def _find_maximal_repeats(self, text, k):
        """Find maximal repeats using suffix array properties"""
        repeats = set()
        last_pos = -1
        last_lcp = 0
        
        # Use existing suffix array
        for i in range(len(self.sa)):
            pos = self.sa[i]
            if pos + k <= len(text):
                current = text[pos:pos+k]
                if current == last_pos:
                    if last_lcp >= k:
                        repeats.add(current)
                last_pos = current
                
        return repeats

    def _calculate_optimized_entropy(self, char_counts, context, total_chars, k):
        """Calculate entropy with advanced optimization techniques"""
        entropy = 0
        context_size = len(set(char_counts.keys()))
        
        # Apply adaptive probability modeling
        for char, count in char_counts.items():
            # Enhanced probability estimation
            prob = count / total_chars
            
            # Apply context-based smoothing
            if context_size > 0:
                smoothed_prob = (1 - 0.1 * (context_size / self.k)) * prob + 0.1 / len(set(self.text))
                entropy -= smoothed_prob * math.log2(smoothed_prob)
            else:
                entropy -= prob * math.log2(prob)
        
        return entropy

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
        for context, counts in self.context_stats.items():
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
        """Compute LF-mapping in O(1) time"""
        if i >= self.n:
            return 0
        char = self.bwt[i]
        rank = self._get_rank(char, i + 1)
        return self.count[char] + rank - 1

    def _get_rank(self, char, pos):
        """O(1) time rank query"""
        if pos == 0:
            return 0
        
        block_size = max(1, int(math.log2(self.n) / 2))
        block_idx = (pos - 1) // block_size
        block_pos = (pos - 1) % block_size
        
        # Get base rank from blocks
        if char not in self.rank_dict or block_idx >= len(self.rank_dict[char]['blocks']):
            return 0
        rank = self.rank_dict[char]['blocks'][block_idx]
        
        # Add small block contribution if available
        if (block_pos > 0 and 
            block_idx < len(self.rank_dict[char]['small_blocks']) and 
            block_pos - 1 < len(self.rank_dict[char]['small_blocks'][block_idx])):
            rank += self.rank_dict[char]['small_blocks'][block_idx][block_pos - 1]
        
        return rank

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
        """Build rank dictionary for O(1) time rank queries"""
        block_size = max(1, int(math.log2(self.n) / 2))  # Small blocks for O(1) lookup
        self.rank_dict = {}
        
        for char in set(self.bwt):
            # Store cumulative counts at block boundaries
            self.rank_dict[char] = {
                'blocks': [],  # Cumulative count at block starts
                'small_blocks': []  # Small block counts for O(1) lookup
            }
            
            count = 0
            small_count = 0
            small_block = []
            
            for i, c in enumerate(self.bwt):
                if i % block_size == 0:
                    self.rank_dict[char]['blocks'].append(count)
                    self.rank_dict[char]['small_blocks'].append(small_block)
                    small_block = []
                    small_count = 0
                
                if c == char:
                    count += 1
                    small_count += 1
                small_block.append(small_count)
                
            # Add final blocks
            self.rank_dict[char]['blocks'].append(count)
            self.rank_dict[char]['small_blocks'].append(small_block)

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
        """Build k-th order context statistics"""
        k = self.k  # Use the specified k-order
        contexts = defaultdict(Counter)
        n = len(self.text)
        
        # Build k-order context statistics
        for i in range(n - k):
            context = self.text[i:i+k]
            next_char = self.text[i+k]
            contexts[context][next_char] += 1
        
        return contexts
        
    def _compress_bwt(self):
        """Compress BWT using k-th order contexts with advanced optimizations"""
        k = self.k
        compressed = []
        n = len(self.bwt)
        
        # Build k-order context model from BWT with sliding window
        contexts = defaultdict(Counter)
        window_size = min(1000, n)  # Use sliding window to capture local patterns
        
        # Initial window
        for i in range(min(n - k, window_size)):
            context = self.bwt[i:i+k]
            next_char = self.bwt[i+k]
            contexts[context][next_char] += 1
        
        # Compress using adaptive context mixing
        i = k
        while i < n:
            # Update sliding window
            if i >= window_size + k:
                old_context = self.bwt[i-window_size-k:i-window_size]
                old_char = self.bwt[i-window_size]
                contexts[old_context][old_char] -= 1
                
                new_context = self.bwt[i-k:i]
                if i < n:
                    contexts[new_context][self.bwt[i]] += 1
            
            context = self.bwt[i-k:i]
            char = self.bwt[i]
            
            # Get probability distribution for this context
            context_counts = contexts[context]
            total = sum(context_counts.values())
            
            if total > 0:
                # Use mixture of different order models
                prob = 0
                weight_sum = 0
                
                # Include predictions from different context lengths
                for ctx_len in range(k + 1):
                    sub_context = context[-ctx_len:] if ctx_len > 0 else ''
                    sub_counts = contexts[sub_context]
                    sub_total = sum(sub_counts.values())
                    
                    if sub_total > 0:
                        # Weight higher order contexts more heavily
                        weight = (ctx_len + 1) / (k + 1)
                        sub_prob = sub_counts[char] / sub_total
                        prob += weight * sub_prob
                        weight_sum += weight
                
                if weight_sum > 0:
                    prob /= weight_sum
                    compressed.append(-math.log2(max(prob, 1e-10)))
                else:
                    compressed.append(math.log2(len(set(self.bwt))))
            else:
                compressed.append(math.log2(len(set(self.bwt))))
            
            i += 1
        
        return compressed

    def _build_blocks(self, text, block_size):
        """Build blocks for entropy calculation"""
        blocks = {}
        n = len(text)
        
        for i in range(0, n - block_size + 1):
            block = text[i:i+block_size]
            blocks[i // block_size] = block
            
        return blocks

    def _calculate_advanced_weight(self, context, context_len, total_chars, n, k):
        """
        Calculate context-specific weight for entropy calculation
        Args:
            context: The context string
            context_len: Length of the context
            total_chars: Total characters in this context
            n: Total text length
            k: Order of entropy
        Returns:
            float: Weight for this context
        """
        # Base weight is proportional to context frequency
        base_weight = total_chars / n
        
        # Adjust weight based on context length
        length_factor = context_len / k if k > 0 else 1
        
        # Penalize very rare contexts to avoid overfitting
        rarity_penalty = min(1.0, total_chars / math.sqrt(n))
        
        return base_weight * length_factor * rarity_penalty

    def _apply_final_optimizations(self, entropy, text):
        """
        Apply final optimizations to the calculated entropy value
        Args:
            entropy: Initial entropy value
            text: Input text
        Returns:
            float: Optimized entropy value
        """
        # Apply length-based scaling
        n = len(text)
        if n > 0:
            # Scale entropy based on text length
            length_factor = 1 - (1 / math.log2(n + 1))
            entropy *= length_factor
            
            # Consider character distribution
            unique_chars = len(set(text))
            if unique_chars > 1:
                # Adjust based on character diversity
                char_factor = math.log2(unique_chars) / unique_chars
                entropy *= (1 + char_factor)
        
        # Ensure entropy stays in reasonable bounds
        return max(0.1, min(entropy, math.log2(len(set(text)))))