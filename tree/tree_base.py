import numpy as np

class TreeBase:
    def __init__(
        self,
        criterion = "ID3",
        max_depth = None,
        min_samples_split = 2,
        min_impurity_split = 1e-7,
    ):
        """
        criterion: 特征选择标准 ["ID3", "C4.5"]
        max_depth: 最大树深度
        min_samples_split: 内部节点再划分所需的最小样本数
        min_impurity_split: 节点划分最小不纯度
        """  
        self.criterion = criterion  
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split

    # fit函数
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.label = np.unique(y)
    
    # predict函数
    def predict(self, test_X):
        pass