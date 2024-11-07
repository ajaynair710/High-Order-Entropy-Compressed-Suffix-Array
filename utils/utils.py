from collections import Counter
import time

def time_function(func):
    """
    Decorator to measure the execution time of a function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

def build_count(text):
    alphabet = sorted(set(text))
    c = Counter(text)
    total = 0
    count = {}
    for char in alphabet:
        count[char] = total
        total += c[char]
    return count

def build_occ(bwt):
    alphabet = set(bwt)
    occ = {char: [0] for char in alphabet}
    for i, char in enumerate(bwt):
        for key in occ.keys():
            occ[key].append(occ[key][-1] + (char == key))
    return occ
