import time
import psutil
import os
import gc
from memory_profiler import profile
from utils.utils import time_function
from tests.test_patterns import generate_random_patterns
from csa.csa import CompressedSuffixArray

def get_process_memory():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class BenchmarkResults:
    def __init__(self):
        self.construction_time = 0
        self.construction_memory = 0
        self.pattern_times = {}
        self.pattern_memory = {}
        self.total_time = 0
        self.peak_memory = 0

@profile
def benchmark_construction(text, epsilon=0.5):
    """Measure CSA construction time and memory"""
    gc.collect()  # Clear memory before start
    
    initial_memory = get_process_memory()
    start_time = time.time()
    
    csa = CompressedSuffixArray(text, epsilon=epsilon)
    
    construction_time = time.time() - start_time
    construction_memory = get_process_memory() - initial_memory
    
    return csa, construction_time, construction_memory

@profile
def benchmark_pattern_search(csa, pattern):
    """Measure single pattern search time and memory"""
    gc.collect()
    
    initial_memory = get_process_memory()
    start_time = time.time()
    
    locations = csa.locate(pattern)
    
    search_time = time.time() - start_time
    search_memory = get_process_memory() - initial_memory
    
    return locations, search_time, search_memory

def run_full_benchmark(text, pattern_lengths=[5, 10, 50, 100, 500, 1000], iterations=3):
    """Run complete benchmark suite with multiple iterations"""
    results = BenchmarkResults()
    
    # Benchmark CSA construction
    print("\nBenchmarking CSA Construction...")
    csa, results.construction_time, results.construction_memory = benchmark_construction(text)
    print(f"Construction Time: {results.construction_time:.4f} seconds")
    print(f"Construction Memory: {results.construction_memory:.2f} MB")
    
    # Generate test patterns
    patterns = generate_random_patterns(text, pattern_lengths)
    
    # Benchmark pattern searching
    print("\nBenchmarking Pattern Searches...")
    for pattern in patterns:
        pattern_results = []
        pattern_memory = []
        
        print(f"\nPattern length: {len(pattern)}")
        for i in range(iterations):
            locations, search_time, search_memory = benchmark_pattern_search(csa, pattern)
            pattern_results.append(search_time)
            pattern_memory.append(search_memory)
            print(f"Iteration {i+1}: Time={search_time:.4f}s, Memory={search_memory:.2f}MB")
            print(f"Found {len(locations)} occurrences")
        
        # Store average results
        results.pattern_times[len(pattern)] = sum(pattern_results) / iterations
        results.pattern_memory[len(pattern)] = sum(pattern_memory) / iterations
    
    # Calculate totals
    results.total_time = results.construction_time + sum(results.pattern_times.values())
    results.peak_memory = max([results.construction_memory] + list(results.pattern_memory.values()))
    
    return results

def print_benchmark_summary(results):
    """Print formatted benchmark results"""
    print("\n=== Benchmark Summary ===")
    print(f"\nConstruction:")
    print(f"Time: {results.construction_time:.4f} seconds")
    print(f"Memory: {results.construction_memory:.2f} MB")
    
    print("\nPattern Search (averages):")
    print("Pattern Length | Time (s) | Memory (MB)")
    print("-" * 40)
    for length in sorted(results.pattern_times.keys()):
        print(f"{length:>13} | {results.pattern_times[length]:>8.4f} | {results.pattern_memory[length]:>10.2f}")
    
    print("\nOverall:")
    print(f"Total Time: {results.total_time:.4f} seconds")
    print(f"Peak Memory: {results.peak_memory:.2f} MB")

if __name__ == "__main__":
    # Example usage
    test_text = "mississippi$" * 1000  # Create larger test text
    results = run_full_benchmark(test_text)
    print_benchmark_summary(results)
