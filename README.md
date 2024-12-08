# High-Order Entropy Compressed Suffix Array (H_k-CSA)

## Overview
This repository contains an implementation of the **High-Order Entropy-Compressed Suffix Array (H_k-CSA)**, based on the paper ["High-Order Entropy-Compressed Text Indexes"](https://www.cs.ucsd.edu/~rgr/website/hk_csa.pdf) by Roberto Grossi, Ankur Gupta, and Jeffrey Scott Vitter. The goal of this project is to develop a compressed suffix array (CSA) that supports fast locate queries in O((logn)^ϵ) time per occurrence and achieve high-order entropy compression for large datasets.

The implementation is done in both **Python**, focusing on modularity, scalability, and performance optimization without relying on third-party libraries like **SDSL**. The Python code compares performance against SDSL's FM-index/CSA using the **Pizza&Chili corpus**.

## Features
- **Efficient Suffix Array Construction:** Implements suffix array construction using the **Burrows-Wheeler Transform (BWT)** and **Wavelet Trees (WT)**.
- **High-Order Entropy Compression:** Compresses the CSA to high-order entropy for space efficiency.
- **Fast Locate Queries:** Optimizes locate queries to O((logn)^ϵ) time per occurrence.
- **Cross-Platform:** The project is implemented in Python for performance comparison.
- **Experimental Analysis:** Performance comparison on real datasets with metrics like **running time** and **RAM usage**.

## Prerequisites

### Python:
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `time`
  - `psutil` (for memory usage analysis)
  - Any other required libraries will be listed in `requirements.txt`.

