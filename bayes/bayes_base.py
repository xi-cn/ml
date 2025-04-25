import numpy as np

class BayesBase:
    def __init__(
        self,
    ):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, test_X):
        pass