import numpy as np
import heapq
from collections import defaultdict, Counter
import lzma
import bz2

def bwt_transform(text, suffix_array):
    """
    Compute the Burrows-Wheeler Transform using a suffix array.
    For each suffix array position, we need the character that precedes it in the text.
    """
    n = len(text)
    bwt = [''] * n
    
    for i in range(n):
        pos = suffix_array[i] - 1
        if pos < 0:
            pos = n - 1
        bwt[i] = text[pos]
    
    return ''.join(bwt)

def mtf_transform(bwt_result):
    """
    Apply Move-to-Front transform to the result of BWT.
    """
    # Initialize the list of symbols (assuming ASCII characters)
    symbols = list(range(256))
    mtf_result = []

    for char in bwt_result:
        # Find the index of the character in the symbols list
        index = symbols.index(ord(char))
        mtf_result.append(index)
        
        # Move the character to the front of the list
        symbols.pop(index)
        symbols.insert(0, ord(char))
    
    return mtf_result

def rle_encode(mtf_result):
    """
    Apply Run-Length Encoding to the MTF result.
    """
    if not mtf_result:
        return []

    rle_result = []
    current_value = mtf_result[0]
    count = 1

    for value in mtf_result[1:]:
        if value == current_value:
            count += 1
        else:
            rle_result.append((current_value, count))
            current_value = value
            count = 1

    # Append the last run
    rle_result.append((current_value, count))

    return rle_result

class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(frequencies):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(rle_result):
    # Flatten the RLE result to a list of symbols
    flat_rle = [symbol for symbol, count in rle_result for _ in range(count)]
    
    # Calculate frequencies
    frequencies = Counter(flat_rle)
    
    # Build Huffman Tree
    huffman_tree = build_huffman_tree(frequencies)
    
    # Build Huffman Codes
    huffman_codes = build_huffman_codes(huffman_tree)
    
    # Encode the data
    encoded_data = ''.join(huffman_codes[symbol] for symbol in flat_rle)
    
    return encoded_data, huffman_codes

def compress_with_lzma(data):
    return lzma.compress(data.encode('utf-8'))

def compress_with_bz2(data):
    return bz2.compress(data.encode('utf-8'))

# Example usage
# if __name__ == "__main__":
#     text = "banana$"
#     suffix_array = [6, 5, 3, 1, 0, 4, 2]
#     bwt_result = bwt_transform(text, suffix_array)
#     mtf_result = mtf_transform(bwt_result)
#     rle_result = rle_encode(mtf_result)
#     encoded_data, huffman_codes = huffman_encode(rle_result)
    
#     # Compress the entire structure using LZMA
#     lzma_compressed_data = compress_with_lzma(encoded_data)
#     print("LZMA Compressed Data Size:", len(lzma_compressed_data))
    
#     # Compress the entire structure using BZip2
#     bz2_compressed_data = compress_with_bz2(encoded_data)
#     print("BZip2 Compressed Data Size:", len(bz2_compressed_data))
