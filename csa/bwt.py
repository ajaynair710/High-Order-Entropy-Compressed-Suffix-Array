import numpy as np

def bwt_transform(text, suffix_array):
    n = len(text)
    bwt = [''] * n
    
    for i in range(n):
        pos = suffix_array[i] - 1
        if pos < 0:
            pos = n - 1
        bwt[i] = text[pos]
    
    return ''.join(bwt)
