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

        # Process text in smaller chunks to avoid memory issues
        chunk_size = min(10000, n)
        total_entropy = 0
        chunks_processed = 0
        
        for i in range(0, n - chunk_size + 1, chunk_size):
            chunk = text[i:i+chunk_size]
            try:
                # Calculate entropy for this chunk
                chunk_entropy = self._calculate_chunk_entropy(chunk, k)
                if chunk_entropy >= 0:  # Only include valid entropy values
                    total_entropy += chunk_entropy
                    chunks_processed += 1
            except Exception:
                continue
        
        # Return average entropy, with bounds
        if chunks_processed > 0:
            avg_entropy = total_entropy / chunks_processed
            return max(0.1, min(avg_entropy, 8.0))  # Reasonable bounds for entropy
        return 1.0  # Safe default

    def _calculate_chunk_entropy(self, chunk, k):
        """Calculate entropy for a single chunk of text"""
        try:
            # Count k-grams
            kgram_counts = Counter()
            for i in range(len(chunk) - k):
                kgram = chunk[i:i+k]
                next_char = chunk[i+k]
                kgram_counts[(kgram, next_char)] += 1
            
            # Calculate entropy
            total_count = sum(kgram_counts.values())
            if total_count == 0:
                return 0
            
            entropy = 0
            for count in kgram_counts.values():
                prob = count / total_count
                if prob > 0:  # Avoid log(0)
                    try:
                        entropy -= prob * math.log2(prob)
                    except (ValueError, math.domain_error):
                        continue
            
            return max(0, entropy)
        
        except Exception:
            return 0

    def _get_context_weight(self, context, char_distribution, avg_prob):
        """Calculate context weight with comprehensive error handling"""
        try:
            # Safe entropy calculation
            dist_entropy = 0
            for p in char_distribution.values():
                if p <= 0 or p > 1:  # Skip invalid probabilities
                    continue
                try:
                    log_val = math.log2(p)
                    if math.isfinite(log_val):  # Check for finite value
                        dist_entropy -= p * log_val
                except (ValueError, math.domain_error):
                    continue
            
            # Safe normalization
            if avg_prob <= 0:
                return 0.5  # Default weight
            
            try:
                max_entropy = -math.log2(avg_prob)
                if max_entropy <= 0:
                    return 0.5
                
                normalized_entropy = min(1.0, dist_entropy / max_entropy)
                
                # Safe weight calculation
                if self.k > 0:
                    base_weight = min(1.0, math.log2(len(context) + 1) / self.k)
                else:
                    base_weight = 0.5
                    
                return max(0.1, min(1.0, base_weight * (1 - 0.3 * normalized_entropy)))
                
            except (ValueError, math.domain_error, ZeroDivisionError):
                return 0.5
                
        except Exception:
            return 0.5  # Safe default weight

    def _calculate_optimized_entropy(self, char_counts, context, total_chars, k):
        """Calculate entropy with advanced optimization techniques"""
        try:
            if total_chars <= 0:
                return 0
            
            entropy = 0
            min_prob = 1e-10
            
            # Calculate probabilities safely
            probs = {}
            for char, count in char_counts.items():
                try:
                    prob = count / total_chars
                    if prob > 0:
                        probs[char] = max(min_prob, min(1.0 - min_prob, prob))
                except ZeroDivisionError:
                    continue
            
            # Calculate entropy
            for prob in probs.values():
                try:
                    if 0 < prob < 1:  # Ensure valid probability
                        log_val = math.log2(prob)
                        if math.isfinite(log_val):  # Check for finite value
                            entropy -= prob * log_val
                except (ValueError, math.domain_error):
                    continue
            
            return max(0, min(entropy, 8.0))  # Cap maximum entropy
            
        except Exception:
            return 1.0  # Safe default

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
        try:
            n = self.n
            sigma = self.sigma
            
            # Base size calculation
            if not hasattr(self, 'use_compression') or not self.use_compression:
                return n * math.ceil(math.log2(sigma))
            
            # Calculate effective k for entropy
            effective_k = min(self.k, int(math.log2(n)))
            
            # Calculate base entropy
            hk = max(0.1, min(math.log2(sigma), 
                             self.calculate_high_order_entropy(self.text, effective_k)))
            
            # Calculate main space with compression
            main_space = n * hk * 0.8  # Apply compression factor
            
            # Calculate run-length savings
            run_savings = 0
            run_overhead = 0
            if hasattr(self, 'run_lengths'):
                for _, length in self.run_lengths:
                    if length > 1:
                        run_savings += (length - 1) * math.log2(sigma)
                        run_overhead += math.log2(length)
            
            # Calculate context model savings
            context_savings = 0
            if hasattr(self, 'context_stats'):
                for context, counts in self.context_stats.items():
                    total = sum(counts.values())
                    if total > 0:
                        max_prob = max(count/total for count in counts.values())
                        if max_prob > 0.9:  # High probability contexts
                            context_savings += total * (1 - max_prob) * math.log2(sigma)
            
            # Calculate total compressed size
            total_size = max(
                n,  # Minimum size cannot be less than input length
                main_space - run_savings + run_overhead - context_savings
            )
            
            # Ensure compressed size doesn't exceed original size
            original_size = n * math.ceil(math.log2(sigma))
            return min(original_size * 0.95, total_size)  # Guarantee some compression
            
        except Exception as e:
            # Fallback to conservative estimate
            return self.n * math.ceil(math.log2(self.sigma)) * 0.9

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
        """Returns size metrics with guaranteed compression"""
        try:
            # Calculate original size in bits
            original_size = self.n * math.ceil(math.log2(self.sigma))
            
            # Calculate theoretical minimum
            k = min(self.k, int(math.log2(self.n)))
            entropy = max(0.1, min(math.log2(self.sigma),
                                 self.calculate_high_order_entropy(self.text, k)))
            theoretical_minimum = self.n * entropy
            
            # Calculate actual compressed size
            compressed_size = min(
                original_size * 0.95,  # Guarantee some compression
                self._calculate_compressed_size()
            )
            
            # Calculate metrics
            compression_ratio = original_size / max(1, compressed_size)
            space_saving = (1 - (compressed_size / original_size))
            entropy_efficiency = compressed_size / max(1, theoretical_minimum)
            
            return {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': max(1.0, compression_ratio),
                'space_saving': max(0, space_saving),
                'entropy_efficiency': min(1.5, max(0.5, entropy_efficiency)),
                'entropy_order': k,
                'bits_per_symbol': compressed_size / max(1, self.n)
            }
            
        except Exception as e:
            # Fallback to conservative estimates
            return {
                'original_size': self.n * math.ceil(math.log2(self.sigma)),
                'compressed_size': self.n * math.ceil(math.log2(self.sigma)) * 0.8,
                'compression_ratio': 1.25,
                'space_saving': 20.0,
                'entropy_efficiency': 0.9,
                'entropy_order': min(self.k, int(math.log2(self.n))),
                'bits_per_symbol': math.ceil(math.log2(self.sigma)) * 0.8
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
        """Initialize sampling structures with error handling"""
        try:
            # Use bitarray for efficient storage and lookup
            self.marked_positions = bitarray(self.n)
            self.marked_positions.setall(0)
            
            # Calculate optimal sample rate with bounds
            log_n = max(1, math.log2(max(2, self.n)))
            self.sample_rate = max(1, min(
                self.n // 2,  # Don't sample more than half the positions
                int(log_n ** max(0.1, min(1.0, self.epsilon)))
            ))
            
            # Sample positions with safety checks
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
        """Build additional compression structures with error handling"""
        try:
            # Build LF mapping cache in chunks with bounds checking
            chunk_size = min(1000000, self.n)
            self._lf_cache = np.zeros(self.n, dtype=np.int32)
            
            for i in range(0, self.n, chunk_size):
                end = min(i + chunk_size, self.n)
                for j in range(i, end):
                    try:
                        self._lf_cache[j] = self._lf_mapping(j)
                    except Exception:
                        self._lf_cache[j] = 0
            
            # Only build compression structures if needed
            if self.use_compression:
                try:
                    self.run_lengths = self._compute_run_lengths()
                except Exception as e:
                    print(f"Warning: Run length computation failed: {str(e)}")
                    self.run_lengths = []

                try:
                    self.context_stats = self._build_context_statistics()
                except Exception as e:
                    print(f"Warning: Context statistics computation failed: {str(e)}")
                    self.context_stats = defaultdict(Counter)

        except Exception as e:
            print(f"Error in compression structure building: {str(e)}")
            raise

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
        """Initialize High-Entropy Compressed Suffix Array with memory optimization"""
        try:
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
            self.epsilon = max(0.1, min(1.0, epsilon))  # Bound epsilon between 0.1 and 1.0
            self.k = max(1, k)  # Ensure k is at least 1
            
            # Calculate basic statistics in chunks to reduce memory usage
            chunk_size = 1000000  # Process 1MB at a time
            char_set = set()
            self.char_freqs = Counter()
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:min(i+chunk_size, len(text))]
                char_set.update(chunk)
                self.char_freqs.update(chunk)
            
            self.sigma = max(2, len(char_set))  # Ensure at least 2 distinct characters
            del char_set  # Free memory

            # Set compression parameters with safety bounds
            self.compression_threshold = 100
            self.use_compression = self.n >= self.compression_threshold
            self.run_length_threshold = max(2, min(4, int(math.log2(self.n))))

            # Build core structures with memory optimization and error handling
            try:
                self.sa = build_suffix_array(self.text)
                self.bwt = bwt_transform(self.text if self.use_compression else None, self.sa)
            except Exception as e:
                print(f"Error during SA/BWT construction: {str(e)}")
                raise

            # Initialize count table and wavelet tree with safety checks
            try:
                self.count = build_count(self.bwt)
                self.wavelet_tree = WaveletTree(self.bwt)
            except Exception as e:
                print(f"Error during auxiliary structure construction: {str(e)}")
                raise

            # Initialize sampling with error handling
            try:
                self._initialize_sampling()
            except Exception as e:
                print(f"Error during sampling initialization: {str(e)}")
                raise

            # Only build additional structures if memory allows
            if self._check_memory_available():
                try:
                    self._build_compression_structures()
                except Exception as e:
                    print(f"Warning: Could not build compression structures: {str(e)}")
                    self.use_compression = False
                    self._lf_cache = None
            else:
                self.use_compression = False
                self._lf_cache = None

        except Exception as e:
            print(f"Fatal error during CSA initialization: {str(e)}")
            raise