from .bayes_base import BayesBase
import numpy as np

class NavieBayes(BayesBase):
    def __init__(
        self,
        alpha = 1
    ):
        """
        alpha: 平滑参数
        """
        super().__init__()
        self.alpha = alpha

    # 计算先验概率
    def _pre_probablity(self):
        self.labels = np.unique(self.y)
        self.pre_prob = {label : 0 for label in self.labels}
        # 统计数量
        for label in self.y:
            self.pre_prob[label] += 1
        # 计算概率
        for label in self.labels:
            self.pre_prob[label] = (self.pre_prob[label] + self.alpha) / (len(self.y) + len(self.labels) * self.alpha)

    # 计算条件概率
    def _condition_probablity(self):
        # 所有特征
        self.features = []
        for col in range(self.X.shape[1]):
            self.features.append(np.unique(self.X[:, col]))
        
        self.condition_prob = {label:[] for label in self.labels}
        # 计算每个特征每个数值的概率
        for label in self.labels:
            # 满足label的数量 和 索引
            label_num = len(self.y[self.y == label])
            label_indice = self.y == label

            for i, feature in enumerate(self.features):
                probs = {}
                for f in feature:
                    # 满足label 和特征样本数量
                    sample_num = sum((self.X[:, i] == f) & label_indice)
                    # 条件概率
                    prob = (sample_num + self.alpha) / (label_num + len(feature) * self.alpha)
                    probs[f] = prob
                self.condition_prob[label].append(probs)

    # 计算概率
    def _predict(self, test_X):
        pred_y = None
        pred_rate = 0
        # 计算每个label的条件概率
        for label in self.labels:
            rate = self.pre_prob[label]
            for col in range(self.X.shape[1]):
                cur_feature = test_X[col]
                # 特征从未在训练集中出现过
                if not cur_feature in self.features[col]:
                    continue
                else:
                    rate = rate * self.condition_prob[label][col][cur_feature]
            if rate > pred_rate:
                pred_y = label
                pred_rate = rate
        return pred_y


    def fit(self, X, y):
        super().fit(X, y)
        # 计算先验概率和条件概率
        self._pre_probablity()
        self._condition_probablity()

    def predict(self, test_X):
        pred_y = np.apply_along_axis(self._predict, 1, test_X)
        return pred_y
