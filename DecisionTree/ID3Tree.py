import math
from collections import Counter
from .Rule import Rule
from .Node import Node
import pickle

class ID3Tree:
    def __init__(self, attributes, data, default, type_map):
        self.attributes = attributes
        self.data = data
        self.default = default
        self.type_map = type_map
        self.tree = None

    def entropy(self, labels):
        total = len(labels)
        counter = Counter(labels)
        return -sum((count / total) * math.log2(count / total) for count in counter.values())

    def train(self):
        data = [list(d[:-1]) + [Counter([d[-1]]).most_common(1)[0][0]] for d in self.data]
        self.tree = self.id3_train(data, self.attributes)

    def fitness_for(self, attribute):
        if self.type_map[attribute] == 'discrete':
            return self.id3_discrete
        return self.id3_continuous

    def id3_train(self, data, attributes):
        if not data:
            return self.default
        if len(set(row[-1] for row in data)) == 1:
            return data[0][-1]

        scores = [(self.fitness_for(attr)(data, attr), attr) for attr in attributes]
        best_gain, best_attr = max(scores)

        if self.type_map[best_attr] == 'continuous':
            threshold = best_gain[1]
            node = Node(best_attr, threshold, best_gain[0])
            above = [row for row in data if row[self.attributes.index(best_attr)] >= threshold]
            below = [row for row in data if row[self.attributes.index(best_attr)] < threshold]
            return {node: {
                '>=': self.id3_train(above, attributes),
                '<': self.id3_train(below, attributes)
            }}
        else:
            index = self.attributes.index(best_attr)
            values = set(row[index] for row in data)
            node = Node(best_attr, None, best_gain[0])
            return {node: {
                val: self.id3_train([row for row in data if row[index] == val], [a for a in attributes if a != best_attr])
                for val in values
            }}

    def id3_continuous(self, data, attribute):
        idx = self.attributes.index(attribute)
        values = sorted(set(row[idx] for row in data))
        if len(values) == 1:
            return -1, None

        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        base_entropy = self.entropy([row[-1] for row in data])

        best_gain, best_thresh = -1, None
        for t in thresholds:
            above = [row for row in data if row[idx] >= t]
            below = [row for row in data if row[idx] < t]
            p, n = len(above) / len(data), len(below) / len(data)
            gain = base_entropy - p * self.entropy([r[-1] for r in above]) - n * self.entropy([r[-1] for r in below])
            if gain > best_gain:
                best_gain, best_thresh = gain, t
        return best_gain, best_thresh

    def id3_discrete(self, data, attribute):
        idx = self.attributes.index(attribute)
        base_entropy = self.entropy([row[-1] for row in data])
        values = set(row[idx] for row in data)

        remainder = 0
        for val in values:
            subset = [row for row in data if row[idx] == val]
            remainder += (len(subset) / len(data)) * self.entropy([row[-1] for row in subset])

        return base_entropy - remainder, None

    def build_rules(self, tree=None, premises=None):
        tree = self.tree if tree is None else tree
        premises = premises or []
        rules = []

        for node, branches in tree.items():
            for value, subtree in branches.items():
                new_premise = premises + [(node.attribute, value, node.threshold) if node.threshold is not None else (node.attribute, '=', value)]
                if isinstance(subtree, dict):
                    rules.extend(self.build_rules(subtree, new_premise))
                else:
                    rules.append(Rule(self.attributes, new_premise, subtree))
        return rules
    
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
