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

    R_1 = [Triple(T, idx, length) for idx in B_1]
    R_2 = [Triple(T, idx, length) for idx in B_2]
    R = R_1 + R_2

    for i, r_char in enumerate(R):
        r_char.rpos = i

    sorted_suffixes_R = sorted(R, key=lambda x: x.triple)

    rank = 0
    prev_triple = None
    for suffix in sorted_suffixes_R:
        if suffix.triple != prev_triple:
            rank += 1
            prev_triple = suffix.triple
        suffix.rank = rank

    R_prime = [suffix.rank for suffix in R]

    if rank < len(R):
        R_prime_suffix_array = ksa(R_prime)
        sorted_suffixes_R = [R[i] for i in R_prime_suffix_array[1:]]
    else:
        sorted_suffixes_R = sorted(R, key=lambda x: x.triple)

    rank_Si = {}
    for i, suffix in enumerate(sorted_suffixes_R):
        rank_Si[suffix.index] = i + 1

    nonsample_pairs = [NonsamplePair(T, idx, rank_Si) for idx in B_0]
    sorted_nonsample_pairs = sorted(nonsample_pairs, key=lambda x: x.pair)

    result = []
    i = 0
    j = 0

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
        else:
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

    while i < len(sorted_suffixes_R):
        result.append(sorted_suffixes_R[i].index)
        i += 1
    while j < len(sorted_nonsample_pairs):
        result.append(sorted_nonsample_pairs[j].index)
        j += 1

    return result

def build_suffix_array(text):
    """suffix array construction"""
    n = len(text)
    suffixes = sorted((text[i:], i) for i in range(n))
    return [index for (_, index) in suffixes]
