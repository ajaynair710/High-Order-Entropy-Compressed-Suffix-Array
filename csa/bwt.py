import numpy as np

def bwt_transform(text, suffix_array):
    """
    Compute the Burrows-Wheeler Transform using a suffix array.
    For each suffix array position, we need the character that precedes it in the text.
    """
    n = len(text)
    bwt = [''] * n
    
    for i in range(n):
        pos = suffix_array[i] - 1
        if pos < 0:
            pos = n - 1
        bwt[i] = text[pos]
    
    return ''.join(bwt)

# def test_bwt_transform():
#     # Test case 1: banana$
#     text1 = "banana$"
#     sa1 = [6, 5, 3, 1, 0, 4, 2]
#     bwt1 = bwt_transform(text1, sa1)
#     expected_bwt1 = "annb$aa"
    
#     # Test case 2: aaa$
#     text2 = "aaa$"
#     sa2 = [3, 2, 1, 0]
#     bwt2 = bwt_transform(text2, sa2)
#     expected_bwt2 = "aaa$"
    
#     # Test case 3: abcd$
#     text3 = "abcd$"
#     sa3 = [4, 0, 1, 2, 3]
#     bwt3 = bwt_transform(text3, sa3)
#     expected_bwt3 = "d$abc"
    
#     # Print results
#     print("Test case 1 (banana$):")
#     print(f"Input text: {text1}")
#     print(f"Suffix Array: {sa1}")
#     print(f"Expected BWT: {expected_bwt1}")
#     print(f"Got BWT:      {bwt1}")
#     print(f"Correct?: {bwt1 == expected_bwt1}\n")
    
#     print("Test case 2 (aaa$):")
#     print(f"Input text: {text2}")
#     print(f"Suffix Array: {sa2}")
#     print(f"Expected BWT: {expected_bwt2}")
#     print(f"Got BWT:      {bwt2}")
#     print(f"Correct?: {bwt2 == expected_bwt2}\n")
    
#     print("Test case 3 (abcd$):")
#     print(f"Input text: {text3}")
#     print(f"Suffix Array: {sa3}")
#     print(f"Expected BWT: {expected_bwt3}")
#     print(f"Got BWT:      {bwt3}")
#     print(f"Correct?: {bwt3 == expected_bwt3}\n")
    
#     # Show detailed transformation for banana$
#     print("Detailed BWT transformation for banana$:")
#     print("Suffix Array Position | Suffix      | Previous char")
#     print("--------------------- | ----------- | -------------")
#     for i, pos in enumerate(sa1):
#         suffix = text1[pos:]
#         prev_char = text1[(pos - 1) % len(text1)]
#         print(f"{pos:>20} | {suffix:<11} | {prev_char}")

# if __name__ == "__main__":
#     test_bwt_transform()
