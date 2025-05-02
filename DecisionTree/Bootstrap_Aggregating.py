from collections import defaultdict
from .Ruleset import Ruleset
import pickle

class Bagging:
    def __init__(self, attributes, data, default, type_map):
        self.classifiers = []
        self.attributes = attributes
        self.data = data
        self.default = default
        self.type_map = type_map

    def train(self):
        self.classifiers = [Ruleset(self.attributes, self.data, self.default, self.type_map) for _ in range(10)]
        for i, clf in enumerate(self.classifiers):
            print(f"Training classifier #{i + 1}")
            clf.train()

    def predict(self, test):
        votes = defaultdict(float)
        for clf in self.classifiers:
            pred, acc = clf.predict(test)
            if pred:
                votes[pred] += acc
        if not votes:
            return self.default, 0.0
        winner = max(votes.items(), key=lambda x: x[1])
        return winner[0], winner[1] / len(self.classifiers)
    
    def save_model(self, file_path):
        """Save the trained model to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        """Load a pre-trained model from a file."""
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model