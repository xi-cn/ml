import numpy as np

class KnnBase:

    def __init__(
        self,
        k = 5,
        p = 2
    ):
        """
        k: 选择邻居数量(建议奇数)\n
        p: 闵氏距离的指数
        """
        self.k = k
        self.p = p
    
    # 计算距离
    def _dist(self, p1:np.ndarray, p2:np.ndarray):
        dist_p = np.abs(np.power(p1 - p2, self.p))
        dist_sum = np.sum(dist_p, axis=0)
        dist = np.power(dist_sum, 1/self.p)
        return dist
    
    # 计算最近的k个元素
    def _k_neighbors(self, test_x:np.array):
        # 计算距离
        dist = np.apply_along_axis(self._dist, 1, self.X, test_x)
        # 距离最小的元素
        min_index = np.argsort(dist)
        # 选取k个结果
        k_index = min_index[:self.k]

        return k_index
    
    # fit函数
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.label = np.unique(y)
    
    # predict函数
    def predict(self, test_X):
        pass


if __name__ == "__main__":
    knn = KnnBase(p=1)
    a1 = np.array([1, 2])   
    a2 = np.array([3, 3])

