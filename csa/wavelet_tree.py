# wavelet_tree.py
import math

class WaveletTree:
    def __init__(self, bwt):
        self.bwt = bwt
        self.alphabet = sorted(set(bwt))
        self.tree = self.build_tree(bwt, self.alphabet)
        # Add RRR-compressed bitvectors
        self.rrr_vectors = self._build_rrr_vectors(self.tree)

    def build_tree(self, bwt, alphabet):
        if len(alphabet) == 1:
            return [1] * len(bwt)
        
        mid = len(alphabet) // 2
        left, right = set(alphabet[:mid]), set(alphabet[mid:])
        bitvector = [1 if char in right else 0 for char in bwt]

        left_bwt = ''.join(char for char, bit in zip(bwt, bitvector) if bit == 0)
        right_bwt = ''.join(char for char, bit in zip(bwt, bitvector) if bit == 1)

        return {
            'bitvector': bitvector,
            'left': self.build_tree(left_bwt, alphabet[:mid]) if left_bwt else None,
            'right': self.build_tree(right_bwt, alphabet[mid:]) if right_bwt else None
        }

    def rank(self, char, index):
        """Count occurrences of char up to index (exclusive)"""
        if char not in self.alphabet:
            print(f"Character {char} not in wavelet tree alphabet")
            return 0
        if index <= 0:
            return 0
        
        node = self.tree
        pos = min(index, len(self.bwt))
        curr_alphabet = self.alphabet
                
        count = 0
        while isinstance(node, dict):
            mid = len(curr_alphabet) // 2
            left_chars = curr_alphabet[:mid]
            right_chars = curr_alphabet[mid:]
            is_right = char in right_chars
            
            bitvector = node['bitvector'][:pos]
            ones = bitvector.count(1)
            zeros = len(bitvector) - ones
            
            
            if is_right:
                count = ones
                node = node['right']
                curr_alphabet = right_chars
                pos = ones
            else:
                count = zeros
                node = node['left']
                curr_alphabet = left_chars
                pos = zeros
        
        return count

    def _build_rrr_vectors(self, node):
        if isinstance(node, list):
            return RRRBitvector(node)
        
        return {
            'bitvector': RRRBitvector(node['bitvector']),
            'left': self._build_rrr_vectors(node['left']) if node['left'] else None,
            'right': self._build_rrr_vectors(node['right']) if node['right'] else None
        }

    def size_in_bits(self):
        """Calculate size in bits using high-order entropy bound"""
        n = len(self.bwt)
        sigma = len(self.alphabet)
        
        # Space for wavelet tree structure
        struct_size = 2 * sigma * math.ceil(math.log2(n))  # Pointers
        
        # Space for RRR-compressed bitvectors
        bitvector_size = self._calculate_bitvector_size(self.tree)
        
        return struct_size + bitvector_size

    def _calculate_bitvector_size(self, node):
        if isinstance(node, list):
            return len(node)
        
        size = len(node['bitvector'])
        if node['left']:
            size += self._calculate_bitvector_size(node['left'])
        if node['right']:
            size += self._calculate_bitvector_size(node['right'])
        return size

class RRRBitvector:
    def __init__(self, bitvector):
        self.block_size = max(1, int(math.log2(len(bitvector)) / 2))
        self.superblock_size = self.block_size * self.block_size
        
        # Store block counts and class IDs
        self.blocks = []
        self.block_classes = []
        self.superblock_samples = []
        
        count = 0
        for i in range(0, len(bitvector), self.block_size):
            block = bitvector[i:i+self.block_size]
            if i % self.superblock_size == 0:
                self.superblock_samples.append(count)
            
            block_count = sum(block)
            self.blocks.append(self._encode_block(block))
            self.block_classes.append(block_count)
            count += block_count

    def _encode_block(self, block):
        """Encode block using combinatorial number system"""
        if not block:
            return 0
        n = len(block)
        ones = sum(block)
        if ones == 0 or ones == n:
            return 0
        
        # Calculate combinatorial number
        rank = 0
        remaining = ones
        for i, bit in enumerate(block):
            if bit and remaining > 0:
                rank += self._binom(n - i - 1, remaining - 1)
                remaining -= 1
        return rank

    def _binom(self, n, k):
        """Calculate binomial coefficient"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c

    def size_in_bits(self):
        """Calculate size in bits"""
        n = len(self.blocks) * self.block_size
        # Space for superblock samples
        superblock_bits = len(self.superblock_samples) * math.ceil(math.log2(n))
        # Space for block classes
        block_class_bits = len(self.block_classes) * math.ceil(math.log2(self.block_size))
        # Space for encoded blocks
        block_bits = sum(math.ceil(math.log2(self._binom(self.block_size, c) + 1)) 
                        for c in self.block_classes)
        
        return superblock_bits + block_class_bits + block_bits
