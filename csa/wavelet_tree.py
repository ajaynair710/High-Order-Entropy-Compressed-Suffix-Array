# wavelet_tree.py
import math
from bitarray import bitarray

class WaveletTree:
    def __init__(self, bwt):
        self.bwt = bwt
        self.alphabet = sorted(set(bwt))
        self.tree = self.build_tree(bwt, self.alphabet)

    def build_tree(self, bwt, alphabet):
        if len(alphabet) == 1:
            return [1] * len(bwt)
        
        mid = len(alphabet) // 2
        left, right = alphabet[:mid], alphabet[mid:]
        bitvector = bitarray(len(bwt))
        bitvector.setall(0)

        for i, char in enumerate(bwt):
            if char in right:
                bitvector[i] = 1

        left_bwt = ''.join(bwt[i] for i in range(len(bwt)) if bitvector[i] == 0)
        right_bwt = ''.join(bwt[i] for i in range(len(bwt)) if bitvector[i] == 1)

        return {
            'bitvector': bitvector,
            'left': self.build_tree(left_bwt, left) if left_bwt else None,
            'right': self.build_tree(right_bwt, right) if right_bwt else None
        }

    def rank(self, char, index):
        if char not in self.alphabet:
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

    def size_in_bits(self):
        n = len(self.bwt)
        sigma = len(self.alphabet)
        struct_size = 2 * sigma * math.ceil(math.log2(n))
        return struct_size + len(self.tree['bitvector']) 
