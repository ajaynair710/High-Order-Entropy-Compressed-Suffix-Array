class Triple(object):
    def __init__(self, T, idx, length):
        t_i = lambda i: T[i] if i < length else ''
        self._triple = (t_i(idx), t_i(idx + 1), t_i(idx + 2))  # The current triple of chars
        self._index = idx  # Original index of the suffix
        self._rank = None  # Rank to be assigned
        self._rpos = None  # Rank position

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
            self.pair = (T[idx], S_i_ranks.get(idx + 1, 0))  # Pair with rank of the next character
        else:
            self.pair = (0, 0)  # Default for out of bounds suffixes

def ksa(T):
    length = len(T)
    B_0 = range(0, length, 3)
    B_1 = range(1, length, 3)
    B_2 = range(2, length, 3)

    R_1 = [Triple(T, idx, length) for idx in B_1]
    R_2 = [Triple(T, idx, length) for idx in B_2]
    R = R_1 + R_2

    # Assign rank positions to the sample suffixes
    for i, r_char in enumerate(R):
        r_char.rpos = i

    # Sort suffixes R by the triple (first, second, third characters)
    sorted_suffixes_R = sorted(R, key=lambda x: x.triple)

    # Assign ranks based on the sorted order of the triples
    rank = 0
    prev_triple = None
    for suffix in sorted_suffixes_R:
        if suffix.triple != prev_triple:
            rank += 1
            prev_triple = suffix.triple
        suffix.rank = rank

    # Now, for further suffix sorting, we apply KSA recursively
    R_prime = [suffix.rank for suffix in R]
    if rank < len(R):
        R_prime_suffix_array = ksa(R_prime)
        sorted_suffixes_R = [R[i] for i in R_prime_suffix_array[1:]]
    else:
        sorted_suffixes_R = sorted(R, key=lambda x: x.triple)

    rank_Si = {}
    for i, suffix in enumerate(sorted_suffixes_R):
        rank_Si[suffix.index] = i + 1

    # Create Nonsample pairs for the unsampled suffixes
    nonsample_pairs = [NonsamplePair(T, idx, rank_Si) for idx in B_0]
    sorted_nonsample_pairs = sorted(nonsample_pairs, key=lambda x: x.pair)

    result = []
    i = 0
    j = 0

    # Compare the sampled and nonsampled suffixes
    def compare_suffixes(sample_idx, nonsample_idx):
        if sample_idx % 3 == 1:
            sample_tuple = (
                ord(T[sample_idx]) if sample_idx < length else 0,
                rank_Si.get(sample_idx + 1, 0)
            )
            nonsample_tuple = (
                ord(T[nonsample_idx]) if nonsample_idx < length else 0,
                rank_Si.get(nonsample_idx + 1, 0)
            )
            return sample_tuple <= nonsample_tuple
        else:
            sample_tuple = (
                ord(T[sample_idx]) if sample_idx < length else 0,
                ord(T[sample_idx + 1]) if sample_idx + 1 < length else 0,
                rank_Si.get(sample_idx + 2, 0)
            )
            nonsample_tuple = (
                ord(T[nonsample_idx]) if nonsample_idx < length else 0,
                ord(T[nonsample_idx + 1]) if nonsample_idx + 1 < length else 0,
                rank_Si.get(nonsample_idx + 2, 0)
            )
            return sample_tuple <= nonsample_tuple

    # Merge the results of sampled and nonsampled suffixes
    while i < len(sorted_suffixes_R) and j < len(sorted_nonsample_pairs):
        sample_suffix = sorted_suffixes_R[i]
        nonsample_suffix = sorted_nonsample_pairs[j]

        if compare_suffixes(sample_suffix.index, nonsample_suffix.index):
            result.append(sample_suffix.index)
            i += 1
        else:
            result.append(nonsample_suffix.index)
            j += 1

    while i < len(sorted_suffixes_R):
        result.append(sorted_suffixes_R[i].index)
        i += 1
    while j < len(sorted_nonsample_pairs):
        result.append(sorted_nonsample_pairs[j].index)
        j += 1

    return result

def build_suffix_array(text):
    """Naive approach to build the suffix array."""
    suffixes = [(text[i:], i) for i in range(len(text))]
    suffixes.sort()  # Sort the suffixes lexicographically
    return [index for (_, index) in suffixes]

def optimized_ksa(T, k):
    """Construct the suffix array using a fixed sampling strategy."""
    length = len(T)
    sampled_indices = fixed_sampling(T, k)

    # Create triples for sampled suffixes
    R = [Triple(T, idx, length) for idx in sampled_indices]

    # Sort sampled suffixes
    sorted_sampled_suffixes = sorted(R, key=lambda x: x.triple)

    # Assign ranks to sampled suffixes
    rank = 0
    prev_triple = None
    for suffix in sorted_sampled_suffixes:
        if suffix.triple != prev_triple:
            rank += 1
            prev_triple = suffix.triple
        suffix.rank = rank

    # Create the suffix array from sampled suffixes
    suffix_array = [suffix.index for suffix in sorted_sampled_suffixes]

    # Handle nonsampled suffixes
    nonsampled_indices = [i for i in range(length) if i not in sampled_indices]
    for idx in nonsampled_indices:
        # Create a Triple for each nonsampled suffix
        suffix = Triple(T, idx, length)
        suffix.rank = rank  # Assign the highest rank to nonsampled suffix
        suffix_array.append(suffix.index)

    return suffix_array

def fixed_sampling(T, k):
    """Select every k-th suffix from the text T."""
    return [i for i in range(len(T)) if i % k == 0]

# Example Usage:
text = "banana"
suffix_array = ksa(text)
print(f"Suffix Array using KSA: {suffix_array}")

optimized_suffix_array = optimized_ksa(text, 3)
print(f"Optimized Suffix Array (with sampling): {optimized_suffix_array}")
