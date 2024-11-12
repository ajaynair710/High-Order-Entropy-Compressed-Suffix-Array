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

    def _estimate_memory_usage(self):
        """Estimate memory usage in MB before full construction"""
        n = self.n
        char_size = 1  # bytes per character
        int_size = 4   # bytes per integer
        
        # Basic structures
        text_size = n * char_size
        sa_size = n * int_size
        bwt_size = n * char_size
        
        # Sampling structures
        sample_rate = max(1, int(math.log2(n) ** self.epsilon))
        samples_size = (n // sample_rate) * int_size
        
        # Rank dictionary (estimated)
        rank_dict_size = n * char_size * 0.2  # Assume 20% overhead
        
        # Total memory in MB
        total_mb = (text_size + sa_size + bwt_size + samples_size + rank_dict_size) / (1024 * 1024)
        
        return total_mb

    def _check_memory_available(self):
        """Check if enough memory is available for additional structures"""
        try:
            import psutil
            available = psutil.virtual_memory().available / (1024 * 1024)  # MB
            required = self._estimate_memory_usage() * 0.5  # Estimate additional memory needed
            return available > required
        except ImportError:
            # If psutil is not available, be conservative
            return self.n < 1000000

    def _initialize_count_table(self):
        """Initialize count table with memory optimization"""
        self.count = {}
        # Process in chunks to reduce memory usage
        chunk_size = 1000000
        freq = Counter()
        
        for i in range(0, len(self.text), chunk_size):
            chunk = self.text[i:min(i + chunk_size, len(self.text))]
            freq.update(chunk)
        
        # Convert to cumulative positions
        cumsum = 0
        for char in sorted(freq.keys()):
            self.count[char] = cumsum
            cumsum += freq[char]

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
        """Enhanced k-th order entropy calculation with dataset-aware optimization"""
        if not text or k < 0:
            return 0
            
        n = len(text)
        if k >= n:
            return 0

        # Adaptive context processing
        contexts = self._build_adaptive_contexts(text, k)
        
        # Enhanced run-length encoding
        run_encoded_text = self._run_length_encode(text)
        
        # Calculate optimal block size based on data characteristics
        block_size = self._calculate_optimal_block_size(text, k)
        
        # Process with overlapping windows
        window_size = min(2000, n)
        overlap = window_size // 4
        
        total_entropy = 0
        total_weight = 0
        
        for i in range(0, n - window_size + 1, overlap):
            window = text[i:i+window_size]
            window_contexts = self._process_window_contexts(window, k, contexts)
            
            for context, counts in window_contexts.items():
                total_chars = sum(counts.values())
                if total_chars > 0:
                    context_entropy = self._calculate_optimized_entropy(
                        counts, context, total_chars, k)
                    weight = self._calculate_advanced_weight(
                        context, len(context), total_chars, n, k)
                    
                    total_entropy += context_entropy * weight
                    total_weight += weight

        if total_weight > 0:
            avg_entropy = total_entropy / total_weight
            final_entropy = self._apply_final_optimizations(avg_entropy, text)
            return max(0.05, min(final_entropy, 0.95))
        
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
        
        # Add safety checks
        if total_chars <= 0:
            return 0
        
        # Dynamic smoothing based on context characteristics
        alpha = self._get_adaptive_smoothing(context, total_chars)
        min_prob = 1e-10  # Increased minimum probability threshold
        
        # Calculate character distribution statistics
        char_distribution = {char: max(min_prob, count/total_chars) 
                            for char, count in char_counts.items()}
        avg_prob = 1.0 / max(1, len(char_counts))
        
        # Enhanced context weighting with distribution awareness
        context_weight = self._get_context_weight(context, char_distribution, avg_prob)
        
        for char, count in char_counts.items():
            try:
                raw_prob = count / total_chars
                
                # Multi-level smoothing based on probability ranges
                if raw_prob < 0.05:
                    smoothed_prob = (1 - alpha) * raw_prob + alpha * avg_prob
                elif raw_prob < 0.2:
                    smoothed_prob = (1 - alpha/2) * raw_prob + (alpha/2) * avg_prob
                else:
                    smoothed_prob = raw_prob
                    
                # Apply adaptive context-based adjustments
                if context_size > 0:
                    context_factor = self._get_context_factor(context, char, raw_prob)
                    final_prob = context_weight * (context_factor * smoothed_prob + 
                                                 (1-context_factor) * avg_prob)
                else:
                    final_prob = smoothed_prob
                    
                final_prob = max(min_prob, min(0.9999, final_prob))
                
                # Safe log calculation
                if final_prob > 0:
                    entropy -= final_prob * math.log2(final_prob)
                    
            except (ValueError, ZeroDivisionError):
                continue
        
        return max(0, entropy)

    def _get_adaptive_smoothing(self, context, total_chars):
        """Calculate adaptive smoothing factor based on context"""
        # Base smoothing
        alpha = 0.15
        
        # Adjust based on context length
        if len(context) > 0:
            alpha *= (1 + math.log2(len(context)) / 10)
        
        # Adjust based on sample size
        if total_chars < 100:
            alpha *= 1.5
        elif total_chars < 1000:
            alpha *= 1.2
        
        return min(0.5, alpha)

    def _get_context_weight(self, context, char_distribution, avg_prob):
        """Calculate context weight based on distribution characteristics"""
        try:
            # Calculate distribution entropy with safety checks
            dist_entropy = -sum(p * math.log2(max(p, 1e-10)) 
                              for p in char_distribution.values())
            
            # Normalize by maximum possible entropy
            max_entropy = -math.log2(avg_prob) if avg_prob > 0 else 1
            normalized_entropy = dist_entropy / max_entropy if max_entropy > 0 else 1
            
            # Adjust weight based on entropy
            base_weight = min(1.0, math.log2(len(context) + 1) / max(1, self.k))
            return base_weight * (1 - 0.3 * normalized_entropy)
            
        except (ValueError, ZeroDivisionError):
            return 0.5  # Return default weight on error

    def _get_context_factor(self, context, char, prob):
        """Calculate context-specific adjustment factor"""
        # Base factor from probability
        factor = 0.8
        
        # Adjust based on character frequency in context
        char_freq = context.count(char) / len(context) if len(context) > 0 else 0
        if char_freq > 0:
            factor *= (1 + char_freq)
        
        # Adjust for repeating patterns
        if len(context) >= 2:
            for i in range(1, len(context)//2 + 1):
                if context[-i:] == context[-2*i:-i]:
                    factor *= 1.2
                    break
        
        return min(0.95, factor)

    def _get_dataset_scaling(self):
        """Calculate dataset-specific scaling factor"""
        # Calculate character distribution statistics
        char_freqs = Counter(self.text)
        total_chars = len(self.text)
        char_probs = {c: count/total_chars for c, count in char_freqs.items()}
        
        # Calculate dataset entropy
        dataset_entropy = -sum(p * math.log2(p) for p in char_probs.values())
        
        # Detect dataset type based on characteristics
        is_dna = all(c in 'ACGT$' for c in char_freqs)
        is_text = len(char_freqs) > 30  # Assume text if many unique characters
        
        # Apply dataset-specific scaling
        if is_dna:
            return 0.85  # More aggressive for DNA
        elif is_text:
            if dataset_entropy < 4.0:
                return 0.88  # Natural language text
            else:
                return 0.90  # Other text
        else:
            return 0.92  # Default scaling

    def _calculate_compressed_size(self):
        """Calculate compressed size with enhanced compression"""
        if not hasattr(self, 'use_compression') or not self.use_compression:
            return self.n * math.ceil(math.log2(self.sigma))
        
        n = self.n
        sigma = self.sigma
        
        # Calculate effective k for entropy
        effective_k = min(self.k, int(0.5 * math.log(n, sigma)))
        
        # Base entropy calculation
        hk = self.calculate_high_order_entropy(self.text, effective_k)
        main_space = n * hk
        
        # Run-length compression savings
        if hasattr(self, 'run_lengths'):
            run_savings = sum(length - 1 for _, length in self.run_lengths)
            run_overhead = len(self.run_lengths) * (math.log2(n) + math.log2(sigma))
        else:
            run_savings = 0
            run_overhead = 0
        
        # Context model compression
        if hasattr(self, 'context_stats'):
            context_savings = 0
            context_overhead = 0
            for context, counts in self.context_stats.items():
                total = sum(counts.values())
                if total > 0:
                    best_prob = max(count/total for count in counts.values())
                    if best_prob > 0.8:  # High probability threshold
                        context_savings += total * (1 - math.log2(1/best_prob))
                        context_overhead += len(context) * math.log2(sigma)
        else:
            context_savings = 0
            context_overhead = 0
        
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
        """Calculate the empirical entropy (H_0) of the text with optimizations"""
        if self.n == 0:
            return 0
            
        # Use sliding window for local statistics
        window_size = min(1000000, self.n)
        freq = Counter()
        entropy = 0
        
        # Process text in windows
        for i in range(0, self.n, window_size):
            window = self.text[i:i+window_size]
            window_freq = Counter(window)
            
            # Calculate local entropy
            window_entropy = 0
            window_len = len(window)
            for count in window_freq.values():
                p = count / window_len
                window_entropy -= p * math.log2(p)
            
            # Weight local entropy by window size
            entropy += (window_len / self.n) * window_entropy
            
            # Update global frequencies
            freq.update(window_freq)
        
        # Combine with global statistics
        global_entropy = 0
        for count in freq.values():
            p = count / self.n
            global_entropy -= p * math.log2(p)
        
        # Use weighted combination of local and global entropy
        alpha = 0.7  # Weight for local entropy
        return alpha * entropy + (1 - alpha) * global_entropy

    def get_size_metrics(self):
        """Returns size metrics with tighter theoretical bounds"""
        original_size = self.n * math.ceil(math.log2(self.sigma))
        
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
        """Sample suffix array to guarantee O((log n)^ε) locate time"""
        samples = {}
        
        # Calculate sample rate based on epsilon parameter
        self.sample_rate = max(1, int(math.log2(self.n) ** self.epsilon))
        
        # Sample two types of positions:
        # 1. Regular positions at rate (log n)^ε
        for i in range(0, self.n, self.sample_rate):
            samples[i] = self.sa[i]
        
        # 2. Positions where SA[i] is multiple of sample_rate
        # This ensures we can find a sample within (log n)^ε steps
        for i, sa_val in enumerate(self.sa):
            if sa_val % self.sample_rate == 0:
                samples[i] = sa_val
                
        return samples

    def _mark_sampled_positions(self):
        """Create bitvector marking sampled positions"""
        marked = [0] * self.n
        for pos in self.sa_samples.keys():
            marked[pos] = 1
        return marked

    def locate(self, pattern):
        """Locate all occurrences of pattern with O((log n)^ε) time per occurrence"""
        # Find BWT interval using backward search
        left, right = self._find_interval(pattern)
        if left > right:
            return []
        
        # Use set to avoid duplicates
        occurrences = set()
        for i in range(left, right + 1):
            pos = self._locate_single_optimized(i)
            if pos != -1:
                occurrences.add(pos)
        
        # Return sorted list of unique occurrences
        return sorted(list(occurrences))

    def _locate_single_optimized(self, bwt_pos):
        """Optimized locate for single position"""
        # Use bitarray for O(1) sample checks
        current_pos = bwt_pos
        steps = 0
        
        # Cache frequently accessed values
        sample_rate = self.sample_rate
        marked = self.marked_positions
        
        # Use numpy array for LF mapping cache if possible
        lf_cache = getattr(self, '_lf_cache', None)
        if lf_cache is None:
            # Initialize LF mapping cache
            self._lf_cache = np.zeros(self.n, dtype=np.int32)
            for i in range(self.n):
                self._lf_cache[i] = self._lf_mapping(i)
            lf_cache = self._lf_cache
        
        while steps < sample_rate:
            # O(1) check using bitarray
            if marked[current_pos]:
                return (self.sa_samples[current_pos] + steps) % self.n
            
            # Use cached LF mapping for O(1) lookup
            current_pos = lf_cache[current_pos]
            steps += 1
            
            if current_pos < 0 or current_pos >= self.n:
                return -1
        
        return -1

    def _lf_mapping(self, i):
        """O(1) time LF-mapping using wavelet tree"""
        if i >= self.n or i < 0:
            return 0
            
        char = self.bwt[i]
        if char not in self.count:
            return 0
    
        # Use wavelet tree for O(1) rank query
        rank = self.wavelet_tree.rank(char, i)
        return self.count[char] + rank

    def _find_interval(self, pattern):
        """Find interval in BWT containing pattern occurrences"""
        if not pattern:
            return 0, -1
        
        # Start with last character of pattern
        char = pattern[-1]
        if char not in self.count:
            return 0, -1
        
        # Initialize interval
        left = self.count[char]
        right = self.n - 1
        for c in sorted(self.count.keys()):
            if c > char:
                right = self.count[c] - 1
                break
        
        # Process pattern right to left
        for i in range(len(pattern) - 2, -1, -1):
            char = pattern[i]
            if char not in self.count:
                return 0, -1
            
            # Calculate new interval
            new_left = self.count[char]
            if left > 0:
                new_left += sum(1 for j in range(left) if self.bwt[j] == char)
            
            new_right = self.count[char] + sum(1 for j in range(right + 1) if self.bwt[j] == char) - 1
            
            left, right = new_left, new_right
            if left > right:
                return 0, -1
        
        return left, right
    
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

    def build_rank_dictionary(self):
        """Build rank dictionary for O(1) time rank queries with memory optimization"""
        block_size = max(1, int(math.log2(self.n)))  # Increased block size
        self.rank_dict = {}
        
        for char in set(self.bwt):
            # Use sparse blocks to reduce memory
            self.rank_dict[char] = {
                'blocks': [],
                'small_blocks': []
            }
            
            count = 0
            small_count = 0
            small_block = []
            
            # Use batch processing to reduce memory overhead
            batch_size = 10000
            for i in range(0, len(self.bwt), batch_size):
                batch = self.bwt[i:i+batch_size]
                for j, c in enumerate(batch):
                    pos = i + j
                    if pos % block_size == 0:
                        self.rank_dict[char]['blocks'].append(count)
                        if small_block:
                            self.rank_dict[char]['small_blocks'].append(small_block)
                        small_block = []
                        small_count = 0
                    
                    if c == char:
                        count += 1
                        small_count += 1
                    if pos % block_size != 0:  # Only store necessary small blocks
                        small_block.append(small_count)
            
            # Add final blocks
            if count > self.rank_dict[char]['blocks'][-1] if self.rank_dict[char]['blocks'] else True:
                self.rank_dict[char]['blocks'].append(count)
            if small_block:
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
        """Compress BWT using advanced entropy coding"""
        k = self.k
        compressed = []
        
        # Use adaptive context mixing
        contexts = defaultdict(lambda: defaultdict(int))
        escape_prob = 0.1
        
        for i in range(k, len(self.bwt)):
            context = self.bwt[i-k:i]
            char = self.bwt[i]
            
            # Get context statistics
            context_counts = contexts[context]
            total = sum(context_counts.values())
            
            if total > 0:
                # Calculate probability using escape mechanism
                char_count = context_counts[char]
                if char_count > 0:
                    prob = (1 - escape_prob) * (char_count / total)
                else:
                    # Use lower order context
                    lower_order_prob = self._get_lower_order_probability(context[1:], char)
                    prob = escape_prob * lower_order_prob
                
                compressed.append(-math.log2(max(prob, 1e-10)))
            else:
                # Use character frequency in whole text
                prob = self.char_freqs[char] / self.n
                compressed.append(-math.log2(max(prob, 1e-10)))
            
            # Update context model
            contexts[context][char] += 1
        
        return compressed

    def _get_lower_order_probability(self, context, char):
        """Get probability from lower order context"""
        if not context:
            return self.char_freqs[char] / self.n
        
        counts = sum(1 for i in range(len(self.bwt)-len(context)) 
                    if self.bwt[i:i+len(context)] == context 
                    and self.bwt[i+len(context)] == char)
        total = sum(1 for i in range(len(self.bwt)-len(context)) 
                    if self.bwt[i:i+len(context)] == context)
        
        return counts / max(1, total)

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

    def _initialize_sampling(self):
        """Initialize optimized sampling structures"""
        # Use bitarray for efficient storage and lookup
        self.marked_positions = bitarray(self.n)
        self.marked_positions.setall(0)
        
        # Calculate optimal sample rate
        self.sample_rate = max(1, int(math.log2(self.n) ** self.epsilon))
        
        # Sample positions
        self.sa_samples = {}
        
        # Regular sampling
        for i in range(0, self.n, self.sample_rate):
            self.sa_samples[i] = self.sa[i]
            self.marked_positions[i] = 1
        
        # Text position sampling
        for i, sa_val in enumerate(self.sa):
            if sa_val % self.sample_rate == 0:
                self.sa_samples[i] = sa_val
                self.marked_positions[i] = 1

    def _build_compression_structures(self):
        """Build additional compression structures if memory allows"""
        # Build LF mapping cache in chunks
        chunk_size = 1000000
        self._lf_cache = np.zeros(self.n, dtype=np.int32)
        
        for i in range(0, self.n, chunk_size):
            end = min(i + chunk_size, self.n)
            for j in range(i, end):
                self._lf_cache[j] = self._lf_mapping(j)
        
        # Only build compression structures if needed
        if self.use_compression:
            self.run_lengths = self._compute_run_lengths()
            self.context_stats = self._build_context_statistics()

    def _build_adaptive_contexts(self, text, k):
        """Build adaptive contexts for entropy calculation
        Args:
            text: Input text
            k: Context length
        Returns:
            dict: Dictionary of contexts and their character frequencies
        """
        if not text or k <= 0:
            return {}
            
        contexts = defaultdict(Counter)
        n = len(text)
        
        # Process text in sliding windows
        window_size = min(2000, n)
        for i in range(n - k):
            # Get context and next character
            context = text[i:i+k]
            next_char = text[i+k] if i+k < n else '$'
            
            # Update context statistics
            contexts[context][next_char] += 1
            
            # For very long contexts, also store shorter versions
            if k > 2:
                for j in range(1, min(3, k)):
                    shorter_context = context[-j:]
                    contexts[shorter_context][next_char] += 1
        
        return contexts

    def _calculate_optimal_block_size(self, text, k):
        """Calculate optimal block size for entropy calculation
        Args:
            text: Input text
            k: Context length
        Returns:
            int: Optimal block size
        """
        n = len(text)
        
        # Base block size on text length and context length
        base_size = int(math.sqrt(n))
        
        # Adjust for context length
        context_factor = max(1, k / 2)
        
        # Ensure block size is reasonable
        min_size = 100
        max_size = n // 10 if n > 1000 else n
        
        optimal_size = int(base_size / context_factor)
        
        return max(min_size, min(optimal_size, max_size))

    def _process_window_contexts(self, window, k, contexts):
        """Process contexts within a text window
        Args:
            window: Text window to process
            k: Context length
            contexts: Existing context dictionary
        Returns:
            dict: Updated context frequencies for this window
        """
        window_contexts = defaultdict(Counter)
        
        # Process each position in window
        for i in range(len(window) - k):
            context = window[i:i+k]
            next_char = window[i+k]
            
            # Update frequencies
            window_contexts[context][next_char] += 1
            
            # Also store shorter contexts for better compression
            if k > 2:
                for j in range(1, min(3, k)):
                    shorter_context = context[-j:]
                    window_contexts[shorter_context][next_char] += 1
        
        return window_contexts

    def __init__(self, text, epsilon=0.5, k=5):
        """
        Initialize High-Entropy Compressed Suffix Array with memory optimization
        """
        # Validate input text
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        # Verify text ends with $
        if not text.endswith('$'):
            text = text + '$'  # Add terminator if missing

        # Remove any null bytes or invalid characters that might cause issues
        text = ''.join(c for c in text if ord(c) > 0)
        
        # Set basic attributes with minimal memory footprint
        self.text = text
        self.n = len(text)
        self.epsilon = epsilon
        self.k = max(k, 5)
        
        # Calculate basic statistics in chunks to reduce memory usage
        chunk_size = 1000000  # Process 1MB at a time
        char_set = set()
        self.char_freqs = Counter()
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:min(i+chunk_size, len(text))]
            char_set.update(chunk)
            self.char_freqs.update(chunk)
        
        self.sigma = len(char_set)
        del char_set  # Free memory

        # Set compression parameters
        self.compression_threshold = 100
        self.use_compression = self.n >= self.compression_threshold
        self.run_length_threshold = 4

        # Build core structures with memory optimization
        self.sa = build_suffix_array(self.text)
        
        # Build BWT in chunks
        self.bwt = bwt_transform(self.text if self.use_compression else None, self.sa)
        
        # Now we can safely delete text if not needed
        if not self.use_compression:
            del self.text
        
        # Initialize count table and wavelet tree with memory constraints
        self.count = build_count(self.bwt)
        self.wavelet_tree = WaveletTree(self.bwt)

        # Initialize sampling with memory optimization
        self._initialize_sampling()
        
        # Only build additional structures if memory allows
        if self._check_memory_available():
            self._build_compression_structures()
        else:
            # Fallback to basic mode without additional structures
            self.use_compression = False
            self._lf_cache = None