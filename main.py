# main.py
from csa.enhanced_fm_index import EnhancedFMIndex
from csa.csa import CompressedSuffixArray
from csa.high_order_entropy import calculate_high_order_entropy
from tests.benchmark import print_benchmark_summary, run_full_benchmark

def main():
    eps = 0.5
    very_long_text = "thequickbrownfoxjumpsoverthelazydog$"
    chunk_size = 1000  # Define a chunk size
    total_chunks = 10000  # Total number of chunks to process

    for i in range(total_chunks):
        chunk = very_long_text * chunk_size  # Create a chunk
        h0 = calculate_high_order_entropy(chunk, 0)  # Zeroth-order entropy
        h5 = calculate_high_order_entropy(chunk, 5)  # Fifth-order entropy

        print(f"Chunk {i + 1}: H_0: {h0:.4f}, H_5: {h5:.4f}")

        test_compression(chunk, eps)
        test_locate(chunk, "fox", eps)

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
    print(f"Expected occurrences of '{pattern}': {actual_count}")
    occurrences = csa.locate(pattern)
    print(f"Occurrences of '{pattern}': {len(occurrences)} at positions {occurrences}")

if __name__ == "__main__":
    main()

