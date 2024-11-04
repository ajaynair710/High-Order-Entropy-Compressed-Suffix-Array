from .bwt import bwt_transform
from .suffix_array import build_suffix_array
from .bitvector import BitVector
from .sampling import SampledSuffixArray
from .wavelet_tree import WaveletTree

class CompressedSuffixArray:
    def __init__(self, text):
        self.text = text + "$"  # Append end-of-text character
        self.suffix_array = build_suffix_array(self.text)
        self.bwt = bwt_transform(self.text)
        self.bitvector = BitVector(len(self.bwt))
        self.wavelet_tree = WaveletTree(self.bwt)
        self.sampled_suffix_array = SampledSuffixArray(self.suffix_array)

        self.compress_to_high_order_entropy()

    def compress_to_high_order_entropy(self):
        # Encode BWT result using a wavelet tree
        self.wavelet_tree = WaveletTree(self.bwt)


    def locate(self, query):
        l, r = self.find_range(query)
        if l == -1 or r == -1:
            print(f"No occurrences found for query '{query}'")
            return []  # No occurrences found

        # Log the range boundaries found
        print(f"Range found for '{query}': l={l}, r={r}")
        
        # Gather positions from the suffix array
        positions = [self.suffix_array[i] for i in range(l, r + 1)]
        print(f"Positions in suffix array for '{query}': {positions}")

        return sorted(positions)




    def count(self, query):
        """
        Count the occurrences of `query` in the text.
        """
        l, r = self.find_range(query)
        return max(0, r - l + 1)
    
    

    def find_range(self, query):
        l, r = 0, len(self.suffix_array) - 1
        
        print(f"Initial l={l}, r={r}")
        
        for char in reversed(query):
            if l > 0:
                l = self.wavelet_tree.rank(char, l - 1) + 1
            else:
                l = self.wavelet_tree.rank(char, l)

            r = self.wavelet_tree.rank(char, r + 1) - 1
            
            print(f"Char={char}, updated l={l}, r={r}")
            
            if l > r:
                return -1, -1  # No occurrences found
                
        return l, r


    def locate(self, query):
        l, r = self.find_range(query)
        if l == -1 or r == -1:
            return []  # No occurrences found
            
        # Gather positions from the suffix array
        positions = [self.suffix_array[i] for i in range(l, r + 1)]
        
        print(f"Found positions for '{query}': {positions}")
        return sorted(positions)

