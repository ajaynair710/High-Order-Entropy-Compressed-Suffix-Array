class WaveletTree:
    def __init__(self, text):
        self.text = text
        self.tree = {}
        self.left_tree = {}
        self.right_tree = {}
        self.build_tree(text, sorted(set(text)))

    def build_tree(self, text, alphabet):
        # Base case: If alphabet has only one character, it's a leaf node
        if len(alphabet) == 1:
            self.tree[text] = [1] * len(text)  # Leaf node with all 1's
            return
        
        # Divide alphabet into left and right halves
        mid = len(alphabet) // 2
        left, right = set(alphabet[:mid]), set(alphabet[mid:])

        # Build current level of the tree as a bitvector for left and right partitions
        self.tree[text] = [1 if char in right else 0 for char in text]
        
        # Partition the text for left and right nodes
        left_text = ''.join([char for char in text if char in left])
        right_text = ''.join([char for char in text if char in right])
        
        # Recursively build the left and right subtrees
        if left_text:
            self.left_tree[text] = WaveletTree(left_text)
        if right_text:
            self.right_tree[text] = WaveletTree(right_text)
    


    def rank(self, character, index):
        if character not in self.tree:
            return 0
        current_text = self.text
        rank_count = 0
        
        while True:
            if index < 0 or index >= len(current_text):
                return 0
            
            bitvector = self.tree[current_text]
            char_bit = 1 if character in self.right_tree else 0

            # Count occurrences of bits up to `index`
            rank_in_bitvector = sum(bit for bit in bitvector[:index + 1] if bit == char_bit)

            # Update index based on whether we go left or right in the tree
            if char_bit == 0:
                current_text = ''.join([c for i, c in enumerate(current_text) if bitvector[i] == 0])
            else:
                current_text = ''.join([c for i, c in enumerate(current_text) if bitvector[i] == 1])

            index = rank_in_bitvector - 1
            
            # If we've reached a leaf node
            if len(current_text) == 1:
                return rank_count + (1 if current_text[0] == character else 0)

            # Update rank_count for the next level
            rank_count += rank_in_bitvector

    def select(self, character, k):
        """
        Find the position of the k-th occurrence of `character`.
        """
        if k < 1 or character not in self.tree:
            return -1

        current_text = self.text
        path = []
        
        # Traverse the wavelet tree to find the k-th occurrence
        while True:
            if len(current_text) == 1:
                break

            # Determine whether to go left or right based on `character`
            if character in self.left_tree:
                path.append(0)  # Go left
                current_text = ''.join([c for i, c in enumerate(current_text) if self.tree[current_text][i] == 0])
            else:
                path.append(1)  # Go right
                current_text = ''.join([c for i, c in enumerate(current_text) if self.tree[current_text][i] == 1])

        # Traverse back up the tree using the path to determine the k-th occurrence
        pos = 0  # Start position
        for direction in reversed(path):
            bitvector = self.tree[current_text]
            if direction == 0:
                pos = sum(1 for bit in bitvector[:pos + k] if bit == 0)  # Count zeros
            else:
                pos = sum(1 for bit in bitvector[:pos + k] if bit == 1)  # Count ones

        return pos if pos < len(self.text) else -1
