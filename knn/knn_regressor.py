import numpy as np
from .knn_base import KnnBase

class KnnRegressor(KnnBase):

    def __init__(self, k=5, p=2):
        super().__init__(k, p)

    # 根据邻居选取value
    def _choose_value(self, neighbor:np.ndarray):
        values = self.y[neighbor]
        value = np.mean(values)
        return value



    def predict(self, test_X):
        # 计算每个x的k近邻
        neighbors = np.apply_along_axis(self._k_neighbors, 1, test_X)
        # 计算label
        pred_y = np.apply_along_axis(self._choose_value, 1, neighbors)
        
        return pred_y
