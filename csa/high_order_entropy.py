import math 
from collections import defaultdict

def calculate_high_order_entropy(text, k):
    """Calculate k-th order empirical entropy."""
    if not text or k < 0:
        return 0
    
    n = len(text)
    if k == 0:
        freq = defaultdict(int)
        for c in text:
            freq[c] += 1
        h0 = -sum((count / n) * math.log2(count / n) for count in freq.values())
        return h0
    
    if n <= k:
        return 0

    contexts = defaultdict(lambda: defaultdict(int))
    for i in range(n - k):
        context = text[i:i + k]
        next_char = text[i + k]
        contexts[context][next_char] += 1

    hk = 0
    for context, context_counts in contexts.items():
        context_total = sum(context_counts.values())
        context_h0 = -sum((count / context_total) * math.log2(count / context_total) for count in context_counts.values())
        hk += (context_total / n) * context_h0

    return hk
