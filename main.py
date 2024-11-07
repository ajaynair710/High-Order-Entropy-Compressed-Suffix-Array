# main.py
from csa.enhanced_fm_index import EnhancedFMIndex
from csa.csa import CompressedSuffixArray
from tests.benchmark import print_benchmark_summary, run_full_benchmark

def main():
    # Original compression tests
    # text = "banana$"
    eps = 0.5
    # print(f"\nTesting with short text '{text}':")
    # test_compression(text, eps)
    # test_locate(text, "ana", eps)
    # results = run_full_benchmark(text)
    # print_benchmark_summary(results)

   # Test with longer strings
    # long_text = "mississippi$"
    # long_text = long_text * 100
    # print(f"\nTesting with longer text (length {len(long_text)}):")
    # test_compression(long_text, eps)
    # test_locate(long_text, "ssi", eps)
    # results = run_full_benchmark(long_text)
    # print_benchmark_summary(results)

    very_long_text = "thequickbrownfoxjumpsoverthelazydog$"
    very_long_text = very_long_text * 1000
    print(f"\nTesting with very long text (length {len(very_long_text)}):")
    test_compression(very_long_text, eps)
    test_locate(very_long_text, "fox", eps)
    results = run_full_benchmark(very_long_text)
    print_benchmark_summary(results)

def test_compression(text, eps):
    csa = CompressedSuffixArray(text, epsilon=eps)
    metrics = csa.get_size_metrics()
    
    print(f"Original size (bits): {metrics['original_size']}")
    print(f"Compressed size (bits): {metrics['compressed_size']}")
    print(f"Compression ratio: {metrics['compression_ratio']:.2f}")
    print(f"Space saving: {metrics['space_saving']*100:.2f}%")
    print(f"Entropy efficiency: {metrics['entropy_efficiency']:.2f}")

def test_locate(text, pattern, eps):
    """Test pattern location in text"""
    csa = CompressedSuffixArray(text, epsilon=eps)
    
    print(f"\nLocating pattern '{pattern}':")
    occurrences = csa.locate(pattern)
    
    if not occurrences:
        print("Pattern not found")
    else:
        print(f"Found {len(occurrences)} occurrences at positions: {occurrences}")
        
        # Verify each occurrence
        # print("\nVerifying occurrences:")
        # for pos in occurrences:
        #     start = max(0, pos - 5)  # Show 5 chars before
        #     end = min(len(text), pos + len(pattern) + 5)  # Show 5 chars after
        #     context = text[start:end]
        #     marker = " " * (pos - start) + "^" * len(pattern)
        #     print(f"Position {pos}:")
        #     print(context)
        #     print(marker)

if __name__ == "__main__":
    main()

