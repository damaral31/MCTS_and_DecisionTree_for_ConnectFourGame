import random
from .ID3Tree import ID3Tree
import pickle

class Ruleset:
    def __init__(self, attributes, data, default, type_map):
        self.attributes = attributes
        self.default = default
        self.type_map = type_map
        self.rules = []
        random.shuffle(data)
        split = int(len(data) * 0.67)
        self.train_data = data[:split]
        self.prune_data = data[split:]

    def train(self):
        tree = ID3Tree(self.attributes, self.train_data, self.default, self.type_map)
        tree.train()
        self.rules = tree.build_rules()
        for rule in self.rules:
            rule.accuracy(self.train_data)
        self.prune()

    def prune(self):
        for rule in self.rules:
            for _ in range(len(rule.premises)):
                acc_before = rule.accuracy(self.prune_data)
                removed = rule.premises.pop()
                if acc_before > rule.get_accuracy(self.prune_data):
                    rule.premises.append(removed)
                    break
        self.rules.sort(key=lambda r: -r.accuracy(self.prune_data))

    def predict(self, test):
        for rule in self.rules:
            prediction = rule.predict(test)
            if prediction is not None:
                return prediction, rule.accuracy()
        return self.default, 0.0
    
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
        print(f"Model loaded from {file_path}")
        return model