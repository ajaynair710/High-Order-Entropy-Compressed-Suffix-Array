from utils.utils import time_function
from tests.test_patterns import generate_random_patterns


@time_function
def run_benchmarks(csa):
    patterns = generate_random_patterns(csa.text, [5, 10, 50, 100, 500, 1000])
    for pattern in patterns:
        print(f"Pattern: {pattern}, Count: {csa.count(pattern)}, Locations: {csa.locate(pattern)}")
