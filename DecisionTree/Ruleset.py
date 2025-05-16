if __name__ == "__main__":
    raise Exception("This script is not meant to be run directly")

import random
from .ID3Tree import ID3Tree
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Ruleset:
    def __init__(self, attributes, data, default, type_map):
        """
        Initialize the Ruleset with attributes, data, default value, and type mapping.
        Splits the data into training and pruning sets.
        """
        self.attributes = attributes
        self.default = default
        self.type_map = type_map
        self.rules = []
        random.shuffle(data)  # Shuffle data for randomness
        split = int(len(data) * 0.67)  # Use 67% for training, rest for pruning
        self.train_data = data[:split]
        self.prune_data = data[split:]

    def train(self):
        """
        Train the ruleset using the ID3 decision tree algorithm.
        Converts the trained tree into a set of rules and calculates their accuracy.
        Then prunes the rules to improve generalization.
        """
        tree = ID3Tree(self.attributes, self.train_data, self.default, self.type_map)
        tree.train()
        self.rules = tree.build_rules()  # Extract rules from the trained tree
        for rule in self.rules:
            rule.accuracy(self.train_data)  # Calculate accuracy on training data
        self.prune()  # Prune rules using the pruning set

    def prune(self):
        """
        Prune each rule by removing premises if it does not decrease accuracy on the pruning set.
        Sorts the rules by their accuracy after pruning.
        """
        for rule in self.rules:
            for _ in range(len(rule.premises)):
                acc_before = rule.accuracy(self.prune_data)
                removed = rule.premises.pop()  # Try removing the last premise
                if acc_before > rule.get_accuracy(self.prune_data):
                    rule.premises.append(removed)  # Restore if accuracy drops
                    break
        self.rules.sort(key=lambda r: -r.accuracy(self.prune_data))  # Sort by accuracy descending

    def predict(self, test):
        """
        Predict the outcome for a given test instance using the rules.
        Returns the prediction and the rule's accuracy, or the default if no rule matches.
        """
        for rule in self.rules:
            prediction = rule.predict(test)
            if prediction is not None:
                return prediction, rule.accuracy()
        return self.default, 0.0  # Return default if no rule matches
    
    def feature_importance(self, normalize=True):
        """
        Calcula a importância das features com base nas premissas das regras geradas.
        A importância é ponderada pela acurácia da regra.
        """
        if not self.rules:
            raise ValueError("O modelo ainda não foi treinado.")

        importance = {attr: 0.0 for attr in self.attributes}

        for rule in self.rules:
            acc = rule.accuracy(self.train_data)
            used_attrs = [premise[0] for premise in rule.premises]
            for attr in used_attrs:
                importance[attr] += acc  # Pondera pela acurácia da regra

        if normalize:
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

        return importance

    def get_train_metrics(self):
        """
        Calculate and return the training metrics for the ruleset.
        Returns a dictionary with accuracy, precision, recall, and F1 score.
        """
        y_true = [row[-1] for row in self.train_data]
        y_pred = [self.predict(row)[0] for row in self.train_data]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save_model(self, file_path):
        """
        Save the trained ruleset model to a file using pickle serialization.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(file_path):
        """
        Load a previously saved ruleset model from a file.
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model