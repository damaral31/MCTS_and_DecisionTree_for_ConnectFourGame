if __name__ == "__main__":
    raise Exception("This script is not meant to be run directly")

class Node:
    def __init__(self, attribute, threshold=None, gain=None, n_samples=None):
        self.attribute = attribute
        self.threshold = threshold
        self.gain = gain
        self.n_samples = n_samples

