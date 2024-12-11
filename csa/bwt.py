# bwt.py
import numpy as np

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
