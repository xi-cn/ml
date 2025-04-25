from .bayes_base import BayesBase
import numpy as np
from math import pow, exp, pi

class GaussianBayes(BayesBase):
    def __init__(self):
        super().__init__()
    
    # 计算先验概率
    def _pre_probablity(self):
        self.labels = np.unique(self.y)
        self.pre_prob = {label : 0 for label in self.labels}
        # 统计数量
        for label in self.y:
            self.pre_prob[label] += 1
        # 计算概率
        for label in self.labels:
            self.pre_prob[label] = self.pre_prob[label] / len(self.y)

    # 条件概率的均值和方差
    def _condition_parameters(self):
        
        self.condition_param = {}
        # 计算每个特征每个数值的均值和方差
        for label in self.labels:
            label_indice = self.y == label
            param = []
            for col in range(self.X.shape[1]):
                sample = self.X[label_indice, col]
                mean = np.mean(sample)
                std = np.std(sample)
                param.append((mean, std))
            self.condition_param[label] = param

    # 计算概率
    def _predict(self, test_X):
        pred_y = None
        pred_rate = 0
        # 计算每个label的条件概率
        for label in self.labels:
            rate = self.pre_prob[label]
            for col in range(self.X.shape[1]):
                mean, std = self.condition_param[label][col]
                cur_feature = test_X[col]
                rate = rate * 1/pow(2*pi*std*std, 1/2) * exp(-pow(cur_feature-mean, 2) / (2 * std * std))
                
            if rate > pred_rate:
                pred_y = label
                pred_rate = rate
        return pred_y

    def fit(self, X, y):
        super().fit(X, y)
        # 计算先验概率和条件概率的均值方差
        self._pre_probablity()
        self._condition_parameters()

    def predict(self, test_X):
        pred_y = np.apply_along_axis(self._predict, 1, test_X)
        return pred_y