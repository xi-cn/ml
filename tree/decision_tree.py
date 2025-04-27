from .tree_base import TreeBase
import numpy as np

class DecisionTree(TreeBase):
    def __init__(self, criterion="ID3", max_depth=None, min_samples_split=2, min_impurity_split=1e-7):
        super().__init__(criterion, max_depth, min_samples_split, min_impurity_split)

        if criterion == "ID3":
            self.criterion_func = self._gain
        elif criterion == "C4.5":
            self.criterion_func = self._gain_ratio
        else:
            raise Exception('criterion must be in ["ID3", "C4.5"]')

    # 计算熵
    def _entropy(self, D:np.ndarray):
        labels = np.unique(D)
        entropy = 0
        for label in labels:
            label_num = np.sum(D == label)
            label_rate = label_num / len(D)
            entropy += label_rate * np.log2(label_rate)
        entropy *= -1
        return entropy
    
    # 计算信息增益
    def _gain(self, X:np.ndarray, y:np.ndarray, f):
        # Y的熵
        HY = self._entropy(y)
        # 计算信息增益
        x = X[:, f]
        labels = np.unique(x)
        # 互信息
        HY_A = 0
        for label in labels:
            label_rate = np.sum(x == label) / len(x)
            label_entropy = self._entropy(y[x == label])
            HY_A += label_rate * label_entropy
        gY_A = HY - HY_A

        return gY_A
    
    # 计算信息增益比
    def _gain_ratio(self, X:np.ndarray, y:np.ndarray, f):
        gY_A = self._gain(X, y, f)
        # 计算特征A的熵
        A = X[:, f]
        HA = max(self._entropy(A), 1e-7)
        return gY_A / HA


    # 创建叶子节点
    def _create_leaf(self, y:np.ndarray):
        labels, counts = np.unique(y, return_counts=True)
        most_index = np.argmax(counts)
        return labels[most_index]

    # 创建树
    def _create_tree(self, X:np.ndarray, y:np.ndarray, features:list, depth):
        # 深度到达限制
        if self.max_depth != None and depth > self.max_depth:
            return self._create_leaf(y)
        # 样本数量到达限制
        if len(X) < self.min_samples_split:
            return self._create_leaf(y)
        # 特征数量不够
        if len(features) == 0:
            return self._create_leaf(y)

        # 创建数据副本
        X = X.copy()
        y = y.copy()
        features = features.copy()

        split_f = features[0]
        max_criterion = 0
        # 计算评判纯度
        for f in features:
            criterion = self.criterion_func(X, y, f)
            if criterion > max_criterion:
                max_criterion = criterion
                split_f = f

        # 纯度小于阈值
        if max_criterion <= self.min_impurity_split:
            return self._create_leaf(y)
        
        # 创建子树
        root = {
            'feature': split_f,
            'tree': {}
        }
        labels = np.unique(X[:, split_f])
        features.remove(split_f)
        for label in labels:
            sample_index = X[:, split_f] == label
            root['tree'][label] = self._create_tree(X[sample_index], y[sample_index], features, depth+1)

        return root

    # 递归选择类别
    def _select_label(self, tree, test_X):
        if type(tree) != dict:
            return tree
        tree_feature = tree['feature']
        x_feature = test_X[tree_feature]
        label = self._select_label(tree['tree'][x_feature], test_X)
        return label
    
    # 预测一个实例
    def _predict(self, test_X):
        label = self._select_label(self.tree, test_X)
        return label

    # 拟合建树
    def fit(self, X, y):
        features = list(range(X.shape[1]))
        self.tree = self._create_tree(X, y, features, 1)

    # predict函数
    def predict(self, test_X):
        labels = np.apply_along_axis(self._predict, 1, test_X)
        return labels
