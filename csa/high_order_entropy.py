# high_order_entropy.py
import math
from collections import defaultdict

def calculate_high_order_entropy(text, k):
    """Calculate k-th order empirical entropy"""
    if not text or k < 0:
        return 0
        
    n = len(text)
    if k == 0:
        freq = defaultdict(int)
        for c in text:
            freq[c] += 1
        h0 = 0
        for count in freq.values():
            p = count / n
            h0 -= p * math.log2(p)
        return h0
        
    # For k > 0, calculate conditional entropy
    contexts = defaultdict(lambda: defaultdict(int))
    for i in range(n - k):
        context = text[i:i+k]
        next_char = text[i+k]
        contexts[context][next_char] += 1
        
    hk = 0
    for context_counts in contexts.values():
        context_total = sum(context_counts.values())
        context_h0 = 0
        for count in context_counts.values():
            p = count / context_total
            context_h0 -= p * math.log2(p)
        hk += (context_total / n) * context_h0
        
    return hk
