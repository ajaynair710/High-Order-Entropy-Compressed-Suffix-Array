import math

class SampledSuffixArray:
    def __init__(self, suffix_array):
        self.suffix_array = suffix_array
        self.sample_rate = int(math.log2(len(suffix_array)) ** 2)
        self.samples = {i: pos for i, pos in enumerate(suffix_array) if i % self.sample_rate == 0}

    def locate(self, index):
        if index in self.samples:
            return self.samples[index]
        
        step = 0
        while (index - step) not in self.samples:
            step += 1
        sample_pos = self.samples[index - step]
        return sample_pos + step
