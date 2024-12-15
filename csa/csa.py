import numpy as np
from csa.bwt import bwt_transform
from csa.suffix_array import build_suffix_array, optimized_ksa
from csa.wavelet_tree import WaveletTree

class FMIndex:
    def __init__(self, text):
        self.text = text
        self.suffix_array = build_suffix_array(text)
        self.bwt = bwt_transform(text, self.suffix_array)
        self.rank = self.precompute_rank()

    def precompute_rank(self):
        rank = {}
        for i, char in enumerate(self.bwt):
            if char not in rank:
                rank[char] = []
            rank[char].append(i)
        return rank

    def backward_search(self, pattern):
        top = 0
        bottom = len(self.bwt) - 1
        
        while top <= bottom and pattern:
            symbol = pattern[-1]
            pattern = pattern[:-1]
            
            rank_top = self.rank.get(symbol, [])[top] if top > 0 and top < len(self.rank.get(symbol, [])) else 0
            rank_bottom = self.rank.get(symbol, [])[bottom] if bottom > 0 and bottom < len(self.rank.get(symbol, [])) else len(self.bwt)
            
            top = rank_top
            bottom = rank_bottom
            
        if top < 0: top = 0
        if bottom >= len(self.bwt): bottom = len(self.bwt) - 1
        
        return list(range(top, bottom + 1))

    def find_pattern(self, pattern):
        matches = self.backward_search(pattern)
        
        matches = [i for i in matches if i < len(self.suffix_array)]
        
        return [self.suffix_array[i] for i in matches]


def main():
    text = "this is an example text"
    
    fm_index = FMIndex(text)
    
    pattern = "example"
    print(f"Searching for the pattern: '{pattern}'")
    matches = fm_index.find_pattern(pattern)
    
    print(f"Pattern '{pattern}' found at indices: {matches}")
    
    wavelet_tree = WaveletTree(text)
    compressed = wavelet_tree.compress()
    print(f"Wavelet Tree Compression: {compressed}")

if __name__ == "__main__":
    main()
