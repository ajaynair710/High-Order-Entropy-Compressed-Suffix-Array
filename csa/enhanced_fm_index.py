# enhanced_fm_index.py
from csa.suffix_array import build_suffix_array
from csa.bwt import bwt_transform
from utils.utils import build_occ
from utils.utils import build_count

class EnhancedFMIndex:
    def __init__(self, text):
        self.text = text + "$"
        self.suffix_array = build_suffix_array(self.text)
        self.bwt = bwt_transform(self.text, self.suffix_array)
        self.occ = build_occ(self.bwt)
        self.count = build_count(self.text)

    def find(self, query):
        l, r = self.find_range(query)
        if l == -1 or r == -1:
            return []
        return [self.suffix_array[i] for i in range(l, r + 1)]

    def find_range(self, query):
        l, r = 0, len(self.bwt) - 1
        for char in reversed(query):
            new_l = self.rank(char, l) + self.count.get(char, 0)
            new_r = self.rank(char, r + 1) + self.count.get(char, 0) - 1

            if new_l > new_r:
                return -1, -1

            l, r = new_l, new_r

        return l, r

    def rank(self, character, index):
        if character not in self.occ:
            return 0
        if index >= len(self.occ[character]):
            index = len(self.occ[character]) - 1
        rank_in_bitvector = self.occ[character][index]
        return rank_in_bitvector
