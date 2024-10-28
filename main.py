from csa.csa import CompressedSuffixArray
from utils.data_loader import load_text
from tests.benchmark import run_benchmarks

def main():
    # Load text from Pizza&Chili corpus for testing
    text = load_text("path/to/dataset.txt")
    
    # Build CSA
    csa = CompressedSuffixArray(text)
    csa.compress_to_high_order_entropy()

    # Run benchmarks
    run_benchmarks(csa)

if __name__ == "__main__":
    main()
