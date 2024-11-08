import requests
import gzip
import os
from pathlib import Path
from tqdm import tqdm
import logging
from tests.benchmark import run_full_benchmark

# Standard test collections (50MB versions)
DATASETS_50MB = {
    'dna': 'http://pizzachili.dcc.uchile.cl/texts/dna/dna.50MB.gz',        # DNA sequences
    'english': 'http://pizzachili.dcc.uchile.cl/texts/nlang/english.50MB.gz', # English text
    'proteins': 'http://pizzachili.dcc.uchile.cl/texts/protein/proteins.50MB.gz', # Protein sequences
    'sources': 'http://pizzachili.dcc.uchile.cl/texts/code/sources.50MB.gz',   # Source code
    'xml': 'http://pizzachili.dcc.uchile.cl/texts/xml/dblp.xml.50MB.gz'    # XML data
}

def benchmark_dataset(dataset_path):
    """Run benchmark on a single dataset"""
    logging.info(f"Testing dataset: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Test metrics:
    # 1. Construction time & memory
    # 2. Pattern search time for different lengths
    # 3. Space usage
    results = run_full_benchmark(text)
    
    # Print results
    print(f"\nResults for {dataset_path}:")
    print(f"Construction Time: {results.construction_time:.4f} seconds")
    print(f"Memory Usage: {results.construction_memory:.2f} MB")
    print("\nPattern Search Times:")
    for length, time in results.pattern_times.items():
        print(f"Length {length}: {time:.4f} seconds")
    
    return results

def main():
    """Run benchmarks on all datasets"""
    for name, url in DATASETS_50MB.items():
        print(f"\nTesting {name} dataset...")
        
        # Download if not exists
        dataset_path = f"datasets/{name}.txt"
        if not os.path.exists(dataset_path):
            download_dataset(name, url)
        
        # Run benchmark
        benchmark_dataset(dataset_path)

if __name__ == "__main__":
    main() 