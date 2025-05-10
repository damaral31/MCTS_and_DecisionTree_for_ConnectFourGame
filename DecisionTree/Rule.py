if __name__ == "__main__":
    raise Exception("This script is not meant to be run directly")

class Rule:
    def __init__(self, attributes, premises=None, conclusion=None):
        self.attributes = attributes # List of attribute names (features) used in the rule
        self.premises = premises or [] # List of premises (conditions) for the rule, each as (attribute, operator, value)
        self.conclusion = conclusion # The conclusion (predicted class or value) if all premises are satisfied
        self._accuracy = None # Cached accuracy value

    def predict(self, test):
        # Check if the test instance satisfies all premises
        for attr, op, value in self.premises:
            idx = self.attributes.index(attr)  # Find the index of the attribute
            # Evaluate the condition based on the operator
            if op == '>=' and not test[idx] >= value:
                return None  # Premise not satisfied
            elif op == '<' and not test[idx] < value:
                return None  # Premise not satisfied
            elif op == '=' and not test[idx] == value:
                return None  # Premise not satisfied
        # All premises satisfied, return the conclusion
        return self.conclusion

    def get_accuracy(self, data):
        # Calculate the accuracy of the rule on the given dataset
        correct, total = 0, 0
        for row in data:
            pred = self.predict(row)  # Predict for each row
            if pred is not None:
                total += 1  # Rule applies to this row
                if pred == row[-1]:  # Check if prediction matches actual value
                    correct += 1
        # Return Laplace-corrected accuracy to avoid zero division
        return (correct + 1) / (total + 2)

    def accuracy(self, data=None):
        # Return cached accuracy if data is None, otherwise compute and cache it
        if data is None:
            return self._accuracy
        self._accuracy = self.get_accuracy(data)
        return self._accuracy