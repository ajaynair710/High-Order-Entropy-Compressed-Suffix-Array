import time
import tracemalloc

def measure_ram_usage():
    return tracemalloc.get_traced_memory()[1]

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
