class EliasFanoEncoder:
    def __init__(self, sorted_list):
        self.sorted_list = sorted_list
        self.build_encoding()

    def build_encoding(self):
        self.upper_bits = []
        self.lower_bits = []
        max_val = max(self.sorted_list)
        u_bits = len(bin(max_val)) - 1
        l_bits = u_bits - int(len(self.sorted_list).bit_length())
        
        for value in self.sorted_list:
            self.upper_bits.append(value >> l_bits)
            self.lower_bits.append(value & ((1 << l_bits) - 1))

    def rank(self, value):
        upper = value >> len(self.lower_bits[0].bit_length())
        count = self.upper_bits.count(upper)
        return count

    def select(self, index):
        if 0 <= index < len(self.sorted_list):
            return self.sorted_list[index]
        return None
