import random

def generate_random_patterns(text, pattern_lengths):
    patterns = []
    for length in pattern_lengths:
        actual_length = min(length, len(text))  # Ensure length doesn't exceed text length
        start = random.randint(0, len(text) - actual_length)
        patterns.append(text[start:start + actual_length])
    return patterns
