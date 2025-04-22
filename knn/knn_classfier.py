import numpy as np
from .knn_base import KnnBase

class KnnClassfier(KnnBase):

    def __init__(self, k=5, p=2):
        super().__init__(k, p)

    # 根据邻居选取label
    def _choose_label(self, neighbor:np.ndarray):
        label_num = {key:0 for key in self.label}
        labels = self.y[neighbor]
        for label in labels:
            label_num[label] += 1
        # 对字典按照值排序
        sorted_keys = sorted(label_num, key=lambda x: label_num[x], reverse=True)
        return sorted_keys[0]



    def predict(self, test_X):
        # 计算每个x的k近邻
        neighbors = np.apply_along_axis(self._k_neighbors, 1, test_X)
        # 计算label
        pred_y = np.apply_along_axis(self._choose_label, 1, neighbors)
        
        return pred_y
