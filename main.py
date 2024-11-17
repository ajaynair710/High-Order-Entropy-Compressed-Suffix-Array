# main.py
from csa.enhanced_fm_index import EnhancedFMIndex
from csa.csa import CompressedSuffixArray
from tests.benchmark import print_benchmark_summary, run_full_benchmark

def main():
    eps = 0.5
    very_long_text = "thequickbrownfoxjumpsoverthelazydog$"
    very_long_text = very_long_text * 1000
    test_compression(very_long_text, eps)
    test_locate(very_long_text, "fox", eps)
    results = run_full_benchmark(very_long_text)

def test_compression(text, eps):
    csa = CompressedSuffixArray(text, epsilon=eps)
    metrics = csa.get_size_metrics()
    
    print(f"Original size (bits): {metrics['original_size']}")
    print(f"Compressed size (bits): {metrics['compressed_size']}")
    print(f"Compression ratio: {metrics['compression_ratio']:.2f}")
    print(f"Space saving: {metrics['space_saving']*100:.2f}%")
    print(f"Entropy efficiency: {metrics['entropy_efficiency']:.2f}")

def test_locate(text, pattern, eps):
    csa = CompressedSuffixArray(text, epsilon=eps)
    actual_count = text.count(pattern)
    # print(f"\nExpected occurrences of '{pattern}': {actual_count}")
    # print(f"Locating pattern '{pattern}':")
    occurrences = csa.locate(pattern)
    


if __name__ == "__main__":
    main()

