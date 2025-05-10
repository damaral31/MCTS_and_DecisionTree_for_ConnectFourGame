if __name__ == "__main__":
    raise Exception("This script is not meant to be run directly")

import math
from collections import Counter
from .Rule import Rule  
from .Node import Node  
import pickle  

class ID3Tree:
    def __init__(self, attributes, data, default, type_map):
        """
        Initialize the ID3Tree with attributes, training data, a default value, and a type map.
        - attributes: List of attribute names.
        - data: Training data (list of lists).
        - default: Default value to return when classification is not possible.
        - type_map: Dictionary mapping attributes to their types ('discrete' or 'continuous').
        """
        self.attributes = attributes
        self.data = data
        self.default = default
        self.type_map = type_map
        self.tree = None  # Placeholder for the trained decision tree

    def entropy(self, labels):
        """
        Calculate the entropy of a set of labels.
        - labels: List of class labels.
        """
        total = len(labels)
        counter = Counter(labels)  # Count occurrences of each label
        return -sum((count / total) * math.log2(count / total) for count in counter.values())

    def train(self):
        """
        Train the decision tree using the ID3 algorithm.
        """
                
        self.tree = self.id3_train(self.data, self.attributes)
        

    def fitness_for(self, attribute):
        """
        Determine the appropriate fitness function based on the attribute type.
        - attribute: The attribute to evaluate.
        """
        if self.type_map[attribute] == 'discrete':
            return self.id3_discrete
        return self.id3_continuous

    def id3_train(self, data, attributes):
        """
        Recursively build the decision tree using the ID3 algorithm.
        - data: Training data.
        - attributes: List of attributes to consider.
        """
        if not data:
            return self.default  # Return default if no data is available
        if len(set(row[-1] for row in data)) == 1:
            return data[0][-1]  # Return the label if all data has the same label

        # Calculate fitness scores for all attributes
        scores = [(self.fitness_for(attr)(data, attr), attr) for attr in attributes]
        best_gain, best_attr = max(scores)  # Select the attribute with the highest gain

        if self.type_map[best_attr] == 'continuous':
            # Handle continuous attributes
            threshold = best_gain[1]
            node = Node(best_attr, threshold, best_gain[0])  # Create a node with a threshold
            above = [row for row in data if row[self.attributes.index(best_attr)] >= threshold]
            below = [row for row in data if row[self.attributes.index(best_attr)] < threshold]
            return {node: {
                '>=': self.id3_train(above, attributes),
                '<': self.id3_train(below, attributes)
            }}
        else:
            # Handle discrete attributes
            index = self.attributes.index(best_attr)
            values = set(row[index] for row in data)
            node = Node(best_attr, None, best_gain[0])  # Create a node without a threshold
            return {node: {
                val: self.id3_train([row for row in data if row[index] == val], [a for a in attributes if a != best_attr])
                for val in values
            }}

    def id3_continuous(self, data, attribute):
        """
        Calculate the information gain for a continuous attribute.
        - data: Training data.
        - attribute: The attribute to evaluate.
        """
        idx = self.attributes.index(attribute)
        values = sorted(set(row[idx] for row in data))  # Unique sorted values of the attribute
        if len(values) == 1:
            return -1, None  # No split possible if only one unique value

        # Calculate potential thresholds
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        base_entropy = self.entropy([row[-1] for row in data])  # Entropy of the entire dataset

        best_gain, best_thresh = -1, None
        for t in thresholds:
            # Split data into above and below threshold
            above = [row for row in data if row[idx] >= t]
            below = [row for row in data if row[idx] < t]
            p, n = len(above) / len(data), len(below) / len(data)
            # Calculate information gain
            gain = base_entropy - p * self.entropy([r[-1] for r in above]) - n * self.entropy([r[-1] for r in below])
            if gain > best_gain:
                best_gain, best_thresh = gain, t
        return best_gain, best_thresh

    def id3_discrete(self, data, attribute):
        """
        Calculate the information gain for a discrete attribute.
        - data: Training data.
        - attribute: The attribute to evaluate.
        """
        idx = self.attributes.index(attribute)
        base_entropy = self.entropy([row[-1] for row in data])  # Entropy of the entire dataset
        values = set(row[idx] for row in data)  # Unique values of the attribute

        remainder = 0
        for val in values:
            # Subset of data where the attribute equals the current value
            subset = [row for row in data if row[idx] == val]
            remainder += (len(subset) / len(data)) * self.entropy([row[-1] for row in subset])

        return base_entropy - remainder, None

    def build_rules(self, tree=None, premises=None):
        """
        Build a list of rules from the decision tree.
        - tree: The decision tree (default is the trained tree).
        - premises: List of premises leading to the current node.
        """
        tree = self.tree if tree is None else tree
        premises = premises or []
        rules = []

        for node, branches in tree.items():
            for value, subtree in branches.items():
                # Add the current condition to the premises
                new_premise = premises + [(node.attribute, value, node.threshold) if node.threshold is not None else (node.attribute, '=', value)]
                if isinstance(subtree, dict):
                    # Recursively build rules for subtrees
                    rules.extend(self.build_rules(subtree, new_premise))
                else:
                    # Create a rule for a leaf node
                    rules.append(Rule(self.attributes, new_premise, subtree))
        return rules

    def save_model(self, file_path):
        """
        Save the trained model to a file.
        - file_path: Path to the file where the model will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        """
        Load a pre-trained model from a file.
        - file_path: Path to the file containing the saved model.
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model