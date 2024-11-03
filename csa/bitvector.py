class BitVector:
    def __init__(self, size):
        self.size = size
        self.bits = [0] * size
        self.rank_support = [0] * (size + 1)

    def set_bit(self, i, value):
        self.bits[i] = value
        self.build_rank_support()

    def build_rank_support(self):
        for i in range(1, self.size + 1):
            self.rank_support[i] = self.rank_support[i - 1] + self.bits[i - 1]

    def rank(self, i):
        return self.rank_support[i + 1]

    def select(self, k):
        for i, count in enumerate(self.rank_support):
            if count >= k:
                return i - 1
        return -1
