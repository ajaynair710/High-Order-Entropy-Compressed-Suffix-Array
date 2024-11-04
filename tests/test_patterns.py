import random

def generate_random_patterns(text, lengths):
    patterns = []
    for length in lengths:
        start = random.randint(0, len(text) - length)
        patterns.append(text[start:start + length])
    return patterns
