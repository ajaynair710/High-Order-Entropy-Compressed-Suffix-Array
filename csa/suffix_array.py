class Triple(object):
    def __init__(self, T, idx, length):
        t_i = lambda i: T[i] if i < length else 0
        self._triple = (t_i(idx), t_i(idx + 1), t_i(idx + 2))
        self._index = idx
        self._rank = None
        self._rpos = None

    @property
    def triple(self):
        return self._triple

    @property
    def index(self):
        return self._index

    @property
    def rpos(self):
        return self._rpos

    @rpos.setter
    def rpos(self, pos):
        self._rpos = pos

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, pos):
        self._rank = pos

    def __repr__(self):
        return f"Triple({self.triple}, {self.index}, {self.rank})"

class NonsamplePair(object):
    def __init__(self, T, idx, S_i_ranks):
        self.index = idx
        self.pair = None
        max_index = len(T)
        if idx < max_index:
            self.pair = (T[idx], S_i_ranks.get(idx + 1, 0))
        else:
            self.pair = (0, 0)

def ksa(T):
    length = len(T)
    B_0 = range(0, length, 3)
    B_1 = range(1, length, 3)
    B_2 = range(2, length, 3)

    # Sort sample suffixes
    R_1 = [Triple(T, idx, length) for idx in B_1]
    R_2 = [Triple(T, idx, length) for idx in B_2]
    R = R_1 + R_2

    for i, r_char in enumerate(R):
        r_char.rpos = i

    sorted_suffixes_R = sorted(R, key=lambda x: x.triple)

    # Assign ranks
    rank = 0
    prev_triple = None
    for suffix in sorted_suffixes_R:
        if suffix.triple != prev_triple:
            rank += 1
            prev_triple = suffix.triple
        suffix.rank = rank

    R_prime = [suffix.rank for suffix in R]

    # Recursive case
    if rank < len(R):
        R_prime_suffix_array = ksa(R_prime)
        sorted_suffixes_R = [R[i] for i in R_prime_suffix_array[1:]]
    else:
        # Base case - direct sort
        sorted_suffixes_R = sorted(R, key=lambda x: x.triple)

    # Build rank lookup table
    rank_Si = {}
    for i, suffix in enumerate(sorted_suffixes_R):
        rank_Si[suffix.index] = i + 1

    # Sort non-sample suffixes
    nonsample_pairs = [NonsamplePair(T, idx, rank_Si) for idx in B_0]
    sorted_nonsample_pairs = sorted(nonsample_pairs, key=lambda x: x.pair)

    # Merge step
    result = []
    i = 0  # Index for sorted sample suffixes
    j = 0  # Index for sorted non-sample suffixes

    def compare_suffixes(sample_idx, nonsample_idx):
        if sample_idx % 3 == 1:
            sample_tuple = (
                T[sample_idx] if sample_idx < length else 0,
                rank_Si.get(sample_idx + 1, 0)
            )
            nonsample_tuple = (
                T[nonsample_idx] if nonsample_idx < length else 0,
                rank_Si.get(nonsample_idx + 1, 0)
            )
            return sample_tuple <= nonsample_tuple
        else:  # sample_idx % 3 == 2
            sample_tuple = (
                T[sample_idx] if sample_idx < length else 0,
                T[sample_idx + 1] if sample_idx + 1 < length else 0,
                rank_Si.get(sample_idx + 2, 0)
            )
            nonsample_tuple = (
                T[nonsample_idx] if nonsample_idx < length else 0,
                T[nonsample_idx + 1] if nonsample_idx + 1 < length else 0,
                rank_Si.get(nonsample_idx + 2, 0)
            )
            return sample_tuple <= nonsample_tuple

    while i < len(sorted_suffixes_R) and j < len(sorted_nonsample_pairs):
        sample_suffix = sorted_suffixes_R[i]
        nonsample_suffix = sorted_nonsample_pairs[j]
        
        if compare_suffixes(sample_suffix.index, nonsample_suffix.index):
            result.append(sample_suffix.index)
            i += 1
        else:
            result.append(nonsample_suffix.index)
            j += 1

    # Append remaining suffixes
    while i < len(sorted_suffixes_R):
        result.append(sorted_suffixes_R[i].index)
        i += 1
    while j < len(sorted_nonsample_pairs):
        result.append(sorted_nonsample_pairs[j].index)
        j += 1

    return result

def build_suffix_array(text):
    T = [ord(c) for c in text]
    return ksa(T)

# def test_suffix_array():
#     # Test case 1: Simple string "banana$"
#     text1 = "banana$"
#     sa1 = build_suffix_array(text1)
#     expected_sa1 = [6, 5, 3, 1, 0, 4, 2]  # Correct suffix array for "banana$"
    
#     # Test case 2: String with repeating characters "aaa$"
#     text2 = "aaa$"
#     sa2 = build_suffix_array(text2)
#     expected_sa2 = [3, 2, 1, 0]  # Correct suffix array for "aaa$"
    
#     # Test case 3: String with all different characters "abcd$"
#     text3 = "abcd$"
#     sa3 = build_suffix_array(text3)
#     expected_sa3 = [4, 0, 1, 2, 3]  # Correct suffix array for "abcd$"
    
#     # Verify results
#     def verify_suffix_array(text, sa):
#         # Check if length is correct
#         if len(sa) != len(text):
#             return False
            
#         # Check if all positions are present
#         if sorted(sa) != list(range(len(text))):
#             return False
            
#         # Check if suffixes are in lexicographical order
#         suffixes = [(text[pos:], pos) for pos in sa]
#         return suffixes == sorted(suffixes)
    
#     # Print results
#     print("Test case 1 (banana$):")
#     print(f"Expected: {expected_sa1}")
#     print(f"Got:      {sa1}")
#     print(f"Correct?: {sa1 == expected_sa1}")
#     print(f"Valid?:   {verify_suffix_array(text1, sa1)}\n")
    
#     print("Test case 2 (aaa$):")
#     print(f"Expected: {expected_sa2}")
#     print(f"Got:      {sa2}")
#     print(f"Correct?: {sa2 == expected_sa2}")
#     print(f"Valid?:   {verify_suffix_array(text2, sa2)}\n")
    
#     print("Test case 3 (abcd$):")
#     print(f"Expected: {expected_sa3}")
#     print(f"Got:      {sa3}")
#     print(f"Correct?: {sa3 == expected_sa3}")
#     print(f"Valid?:   {verify_suffix_array(text3, sa3)}\n")
    
#     # Print all suffixes in order for debugging
#     print("Detailed suffix list for banana$:")
#     sa = sa1
#     for i, pos in enumerate(sa):
#         print(f"SA[{i}] = {pos}: {text1[pos:]}")

# if __name__ == "__main__":
#     test_suffix_array()
