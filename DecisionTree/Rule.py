class Rule:
    def __init__(self, attributes, premises=None, conclusion=None):
        self.attributes = attributes
        self.premises = premises or []
        self.conclusion = conclusion
        self._accuracy = None

    def predict(self, test):
        for attr, op, value in self.premises:
            idx = self.attributes.index(attr)
            if op == '>=' and not test[idx] >= value:
                return None
            elif op == '<' and not test[idx] < value:
                return None
            elif op == '=' and not test[idx] == value:
                return None
        return self.conclusion

    def get_accuracy(self, data):
        correct, total = 0, 0
        for row in data:
            pred = self.predict(row)
            if pred is not None:
                total += 1
                if pred == row[-1]:
                    correct += 1
        return (correct + 1) / (total + 2)

    def accuracy(self, data=None):
        if data is None:
            return self._accuracy
        self._accuracy = self.get_accuracy(data)
        return self._accuracy