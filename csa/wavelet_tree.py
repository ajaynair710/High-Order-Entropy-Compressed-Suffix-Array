# wavelet_tree.py
import math
from bitarray import bitarray
from csa.bwt import bwt_transform, compress_with_lzma, compress_with_bz2
import lzma
import bz2

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

# Example usage
# if __name__ == "__main__":
#     text = "banana$"
#     suffix_array = [6, 5, 3, 1, 0, 4, 2]
#     bwt_result = bwt_transform(text, suffix_array)
    
#     # Compress the BWT result
#     lzma_compressed_data = compress_with_lzma(bwt_result)
#     bz2_compressed_data = compress_with_bz2(bwt_result)
    
#     # Decompress before using in Wavelet Tree
#     decompressed_bwt_lzma = lzma.decompress(lzma_compressed_data).decode('utf-8')
#     decompressed_bwt_bz2 = bz2.decompress(bz2_compressed_data).decode('utf-8')
    
#     # Build Wavelet Tree using decompressed data
#     wavelet_tree_lzma = WaveletTree(decompressed_bwt_lzma)
#     wavelet_tree_bz2 = WaveletTree(decompressed_bwt_bz2)
    
#     print("Wavelet Tree built from LZMA decompressed BWT:", wavelet_tree_lzma.tree)
#     print("Wavelet Tree built from BZip2 decompressed BWT:", wavelet_tree_bz2.tree)
