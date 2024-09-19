class Permutation:
    def __init__(self):
        self.ids = {}
        self.permutation = None

    def add(self, k, v):
        """
        example of k is rank (int)
        example of v is did (str)
        """
        if k <= 0:
            print("invalid rank %d" % k)
            return False
        if k in self.ids:
            print("duplicate item at position %d" % k)
            return False
        self.ids[k] = v
        self.permutation = None
        return True

    def value(self):
        if self.permutation == None:
            permutation = [None] * len(self.ids)
            for k, v in self.ids.items():
                rank = k - 1  # base zero rank
                if not (rank < len(permutation)):
                    print("incomplete permutation")
                    return None
                permutation[rank] = v
            if None in permutation:
                print("incomplete permutation")
                return None
            self.permutation = permutation
        return self.permutation
