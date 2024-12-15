import numpy as np
import math
from collections import defaultdict

class SuccinctRankSelect:
    def __init__(self, bitmap):
        self.n = len(bitmap)
        self.bit_vector = np.array(bitmap, dtype=np.uint8)
        self.rank_support = np.zeros(self.n + 1, dtype=np.uint32)
        
        for i in range(1, self.n + 1):
            self.rank_support[i] = self.rank_support[i - 1] + self.bit_vector[i - 1]

    def rank(self, i):
        return self.rank_support[i]

    def select(self, k):
        low, high = 0, self.n
        while low < high:
            mid = (low + high) // 2
            if self.rank(mid) < k:
                low = mid + 1
            else:
                high = mid
        return low

class GolombRiceEncoder:
    def __init__(self, bitmap):
        ones_count = sum(bitmap)
        total_len = len(bitmap)
        self.m = self.compute_dynamic_m(ones_count, total_len)

    def compute_dynamic_m(self, ones_count, total_len):
        if ones_count == 0:
            return 1
        ratio = ones_count / total_len
        m = max(1, int(math.log2(1 / ratio)))
        return m

    def encode(self, bitmap):
        encoded = []
        quotient, remainder = 0, 0
        for bit in bitmap:
            if bit == 1:
                quotient += 1
                remainder = quotient
            else:
                if quotient > 0:
                    encoded.extend(self._encode_golomb(quotient))
                quotient = 0
        if quotient > 0:
            encoded.extend(self._encode_golomb(quotient))
        return encoded

    def _encode_golomb(self, value):
        quotient = value // self.m
        remainder = value % self.m
        encoded_value = [0] * quotient + [1]
        encoded_value.extend(self._int_to_binary(remainder, self.m))
        return encoded_value

    def _int_to_binary(self, n, bits):
        return [int(b) for b in bin(n)[2:].zfill(bits)]

class WaveletTree:
    def __init__(self, text):
        self.text = text
        self.alphabet = sorted(set(text))
        self.m = None
        self.build_tree()

    def build_tree(self):
        current_text = self.text
        self.tree = []
        self.rank_structures = []

        while len(self.alphabet) > 1:
            mid = len(self.alphabet) // 2
            left_alphabet = self.alphabet[:mid]
            right_alphabet = self.alphabet[mid:]

            bitmap = [1 if c in right_alphabet else 0 for c in current_text]

            rle_encoded_bitmap = self.run_length_encode(bitmap)

            golomb_rice_encoder = GolombRiceEncoder(rle_encoded_bitmap)
            compressed_bitmap = golomb_rice_encoder.encode(rle_encoded_bitmap)

            if self.m is None:
                self.m = golomb_rice_encoder.m

            next_text = [c for c in current_text if c in left_alphabet]

            self.tree.append((compressed_bitmap, left_alphabet, right_alphabet, next_text))

            rank_select_structure = SuccinctRankSelect(bitmap)
            self.rank_structures.append(rank_select_structure)

            self.alphabet = left_alphabet
            current_text = next_text

    def run_length_encode(self, bitmap):
        encoded = []
        current_bit = bitmap[0]
        count = 0
        for bit in bitmap:
            if bit == current_bit:
                count += 1
            else:
                encoded.append((current_bit, count))
                current_bit = bit
                count = 1
        encoded.append((current_bit, count))
        rle_expanded = []
        for bit, length in encoded:
            rle_expanded.extend([bit] * length)
        return rle_expanded

    def level_ordered_encode(self, bitmap):
        encoded = []
        current_bit = bitmap[0]
        count = 0
        for bit in bitmap:
            if bit == current_bit:
                count += 1
            else:
                encoded.append((current_bit, count))
                current_bit = bit
                count = 1
        encoded.append((current_bit, count))
        return encoded

    def rank(self, c, i):
        rank_result = 0
        for level, (compressed_bitmap, left_alphabet, right_alphabet, next_text) in enumerate(self.tree):
            if c in left_alphabet:
                rank_result = self.rank_structures[level].rank(i + 1)
            else:
                rank_result = self.rank_structures[level].rank(i + 1)
        return rank_result

    def select(self, c, k):
        select_result = 0
        for level, (compressed_bitmap, left_alphabet, right_alphabet, next_text) in enumerate(self.tree):
            if c in left_alphabet:
                select_result = self.rank_structures[level].select(k)
            else:
                select_result = self.rank_structures[level].select(k)
        return select_result

    def compress(self):
        compressed = []
        for level in self.tree:
            compressed_bitmap, left_alphabet, right_alphabet, next_text = level
            compressed.append(compressed_bitmap)
        return compressed

    def decompress(self, compressed):
        current_text = [''] * len(self.text)
        for level, (compressed_bitmap, left_alphabet, right_alphabet, next_text) in reversed(list(enumerate(self.tree))):
            decoded_bitmap = self._decode_golomb_rice(compressed_bitmap)
            current_text = self._decompress_level(decoded_bitmap, left_alphabet, right_alphabet, current_text)
        return ''.join(current_text)

    def _decode_golomb_rice(self, compressed_bitmap):
        decoded = []
        i = 0
        while i < len(compressed_bitmap):
            quotient = 0
            while i < len(compressed_bitmap) and compressed_bitmap[i] == 0:
                quotient += 1
                i += 1
            if i + self.m <= len(compressed_bitmap):
                remainder = int(''.join(map(str, compressed_bitmap[i:i + self.m])), 2)
                i += self.m
                decoded.append(quotient * self.m + remainder)
            else:
                break
        return decoded

    def _decompress_level(self, bitmap, left_alphabet, right_alphabet, current_text):
        left_count = bitmap.count(0)
        right_count = bitmap.count(1)

        left_partition = [''] * left_count
        right_partition = [''] * right_count

        left_index = 0
        right_index = 0
        for i, b in enumerate(bitmap):
            if b == 0:
                if left_index < left_count:
                    left_partition[left_index] = current_text[i] if i < len(current_text) else ''
                    left_index += 1
            else:
                if right_index < right_count:
                    right_partition[right_index] = current_text[i] if i < len(current_text) else ''
                    right_index += 1

        return left_partition + right_partition

text = "this is an example text"
wavelet_tree = WaveletTree(text)
compressed_tree = wavelet_tree.compress()
decompressed_text = wavelet_tree.decompress(compressed_tree)

print(f"Original Text: {text}")
print(f"Decompressed Text: {decompressed_text}")
