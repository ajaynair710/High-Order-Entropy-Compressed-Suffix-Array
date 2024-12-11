import numpy as np
from collections import defaultdict
import math

class SuccinctRankSelect:
    def __init__(self, bitmap):
        self.n = len(bitmap)
        self.bit_vector = np.array(bitmap, dtype=np.uint8)
        self.rank_support = np.zeros(self.n + 1, dtype=np.uint32)
        
        # Precompute rank values
        for i in range(1, self.n + 1):
            self.rank_support[i] = self.rank_support[i - 1] + self.bit_vector[i - 1]

    def rank(self, i):
        """Returns the number of 1's in the bit vector up to index i."""
        return self.rank_support[i]

    def select(self, k):
        """Returns the index of the k-th 1 in the bit vector."""
        low, high = 0, self.n
        while low < high:
            mid = (low + high) // 2
            if self.rank(mid) < k:
                low = mid + 1
            else:
                high = mid
        return low

class WaveletTree:
    def __init__(self, text):
        self.text = text
        self.alphabet = sorted(set(text))  # Unique characters in the text
        self.build_tree()

    def build_tree(self):
        current_text = self.text
        self.tree = []  # Stores the compressed bitmaps at each level
        self.rank_structures = []  # List of Rank/Select structures

        while len(self.alphabet) > 1:
            mid = len(self.alphabet) // 2
            left_alphabet = self.alphabet[:mid]
            right_alphabet = self.alphabet[mid:]

            # Generate the bitmap for the partitioning
            bitmap = [1 if c in right_alphabet else 0 for c in current_text]

            # Use Level-Ordered Encoding (LOE) and Range Encoding
            level_ordered_encoded = self.level_ordered_encode(bitmap)
            range_encoded_bitmap = self.range_encode(level_ordered_encoded)

            # Create the new string for the next level of the tree
            next_text = [c for c in current_text if c in left_alphabet]

            # Store the compressed bitmap and new partitioned alphabet
            self.tree.append((range_encoded_bitmap, left_alphabet, right_alphabet, next_text))

            # Build Succinct Rank/Select structure for the bitmap
            rank_select_structure = SuccinctRankSelect(bitmap)
            self.rank_structures.append(rank_select_structure)

            # Move to the left partition for the next level
            self.alphabet = left_alphabet
            current_text = next_text

    def level_ordered_encode(self, bitmap):
        """Level-Ordered Encoding (LOE): Groups consecutive runs of 0s and 1s in blocks."""
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
        encoded.append((current_bit, count))  # Append last block
        return encoded

    def range_encode(self, level_ordered_encoded):
        """Range encoding: Compresses the sequence of run lengths using variable-length encoding."""
        encoded = []
        for bit_value, count in level_ordered_encoded:
            # Use a simple range encoding approach (you could replace this with an actual range encoding library)
            # Assign each bit value (0 or 1) a range proportional to its frequency
            # For simplicity, here we're just storing the run lengths, but you can refine this with real range encoding
            encoded.extend([bit_value] * count)
        return encoded

    def range_decode(self, encoded_bitmap):
        """Decodes the range-encoded bitmap back to the original bitmap."""
        return encoded_bitmap  # In this simplified example, we just return the encoded bitmap directly

    def rank(self, c, i):
        """Rank operation: Returns the number of occurrences of character `c` up to index `i`."""
        rank_result = 0
        for level, (compressed_bitmap, left_alphabet, right_alphabet, next_text) in enumerate(self.tree):
            if c in left_alphabet:
                # Rank in left partition (0 in the bitmap)
                rank_result = self.rank_structures[level].rank(i + 1)  # Rank in left partition
            else:
                # Rank in right partition (1 in the bitmap)
                rank_result = self.rank_structures[level].rank(i + 1)  # Rank in right partition
        return rank_result

    def select(self, c, k):
        """Select operation: Finds the k-th occurrence of character `c` in the compressed text."""
        select_result = 0
        for level, (compressed_bitmap, left_alphabet, right_alphabet, next_text) in enumerate(self.tree):
            if c in left_alphabet:
                # Select in left partition (0 in the bitmap)
                select_result = self.rank_structures[level].select(k)
            else:
                # Select in right partition (1 in the bitmap)
                select_result = self.rank_structures[level].select(k)
        return select_result

    def compress(self):
        """Compress the wavelet tree."""
        compressed = []
        for level in self.tree:
            compressed_bitmap, left_alphabet, right_alphabet, next_text = level
            compressed.append(compressed_bitmap)
        return compressed

    def decompress(self, compressed):
        """Decompress the wavelet tree back to the original text."""
        current_text = [''] * len(self.text)  # Create an empty list to hold the decompressed text
        for level, (compressed_bitmap, left_alphabet, right_alphabet, next_text) in reversed(list(enumerate(self.tree))):
            decoded_bitmap = self.range_decode(compressed_bitmap)  # Decode the range-encoded bitmap
            current_text = self._decompress_level(decoded_bitmap, left_alphabet, right_alphabet, current_text)
        return ''.join(current_text)

    def _decompress_level(self, bitmap, left_alphabet, right_alphabet, current_text):
        """Reconstruct the character sequence at this level from the encoded bitmap."""
        # Count occurrences of 0s and 1s in the bitmap
        left_count = bitmap.count(0)
        right_count = bitmap.count(1)

        # Allocate the correct number of spaces for left and right partitions
        left_partition = [''] * left_count
        right_partition = [''] * right_count

        # Fill partitions based on bitmap
        left_index = 0
        right_index = 0
        for i, b in enumerate(bitmap):
            if b == 0:
                if left_index < left_count:  # Check for index bounds
                    left_partition[left_index] = current_text[i] if i < len(current_text) else ''  # Ensure i is within bounds
                    left_index += 1
            else:
                if right_index < right_count:  # Check for index bounds
                    right_partition[right_index] = current_text[i] if i < len(current_text) else ''  # Ensure i is within bounds
                    right_index += 1

        # Ensure that left and right partition sizes are correct
        if left_index != left_count or right_index != right_count:
            print(f"Debug: Final Left index: {left_index}, Right index: {right_index}")
            raise ValueError(f"Mismatch in partition sizes during decompression. Left index: {left_index}, Right index: {right_index}")
        
        # Merge partitions and return the combined result
        return left_partition + right_partition

# Example Usage:
text = "this is an example text"
wavelet_tree = WaveletTree(text)