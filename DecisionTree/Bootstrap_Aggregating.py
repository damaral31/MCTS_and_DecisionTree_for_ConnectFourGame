if __name__ == "__main__":
    raise Exception("This script is not meant to be run directly")

from collections import defaultdict
from .Ruleset import Ruleset
import pickle

class Bagging:
    def __init__(self, attributes, data, default, type_map):
        """
        Initializes the Bagging class with the necessary attributes.
        
        :param attributes: List of attributes used by the classifiers.
        :param data: Training dataset.
        :param default: Default value for predictions when there is no consensus.
        :param type_map: Mapping of attribute types (e.g., categorical or numerical).
        """
        self.classifiers = []  # List to store individual classifiers
        self.attributes = attributes  # Attributes used by the classifiers
        self.data = data  # Training data
        self.default = default  # Default value for predictions
        self.type_map = type_map  # Mapping of attribute types

    def train(self):
        """
        Trains 10 independent classifiers using the Ruleset class.
        """
        # Create 10 instances of Ruleset and store them in the classifiers list.
        self.classifiers = [Ruleset(self.attributes, self.data, self.default, self.type_map) for _ in range(10)]
        for i, clf in enumerate(self.classifiers):
            print(f"Training classifier #{i + 1}")  
            clf.train()  # Train the current classifier.

    def predict(self, test):
        """
        Makes a prediction for a test instance by combining the results of the classifiers.
        
        :param test: Test instance for which the prediction will be made.
        :return: Final prediction and average confidence.
        """
        votes = defaultdict(float)  # Dictionary to accumulate votes from classifiers.
        for clf in self.classifiers:
            pred, acc = clf.predict(test)  # Get the prediction and confidence from the classifier.
            if pred:  
                votes[pred] += acc  # Add the confidence to the corresponding vote.
        if not votes:  
            return self.default, 0.0  # Return the default value with confidence 0.
        
        # Determine the prediction with the highest vote sum.
        winner = max(votes.items(), key=lambda x: x[1])
        return winner[0], winner[1] / len(self.classifiers)  # Return the prediction and average confidence.

    def save_model(self, file_path):
        """
        Saves the trained model to a file.
        
        :param file_path: Path to the file where the model will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)  # Serialize the model and save it to the file.
        print(f"Model saved to {file_path}")  # Display a success message.

    @staticmethod
    def load_model(file_path):
        """
        Loads a previously trained model from a file.
        
        :param file_path: Path to the file from which the model will be loaded.
        :return: Instance of the loaded model.
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)  # Deserialize the model from the file.
        return model  # Return the loaded model.