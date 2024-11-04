# from csa.csa import CompressedSuffixArray
# from utils.data_loader import load_text
# from tests.benchmark import run_benchmarks

# def main():
#     # Load text from Pizza&Chili corpus for testing
#     text = load_text(r"dataset\english.50MB.gz", size_limit=100000)
    
#     # Build CSA
#     csa = CompressedSuffixArray(text)
#     csa.compress_to_high_order_entropy()

#     # Run benchmarks
#     run_benchmarks(csa)

# if __name__ == "__main__":
#     main()


from csa.csa import CompressedSuffixArray
import sys
import time

def main():
    text = "banana"
    csa = CompressedSuffixArray(text)

    # Print the Suffix Array and BWT
    print(f"Suffix Array: {csa.suffix_array}")
    print(f"BWT: {csa.bwt}")

    # Assess Compression
    original_size = sys.getsizeof(text.encode('utf-8'))
    compressed_size = sys.getsizeof(csa.suffix_array) + sys.getsizeof(csa.bwt)
    compression_ratio = original_size / compressed_size
    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Test the count method
    query = "ana"
    count_result = csa.count(query)
    print(f"Count of '{query}': {count_result}")

    # Test the locate method
    start_time = time.time()
    locate_results = csa.locate(query)
    end_time = time.time()
    print(f"Positions of '{query}': {locate_results}")
    print(f"Time taken to locate: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()


