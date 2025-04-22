
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_iris, load_diabetes
import numpy as np

from knn import KnnClassfier, KnnRegressor

# 归一化
def normalize(X:np.ndarray):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X = (X - mins) / (maxs - mins + 1e-5)
    return X
    

# 分类问题
def classfication_method(MODEL):
    X, y = load_iris(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=11)

    model = MODEL()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

    metric = confusion_matrix(test_y, pred_y)
    acc = accuracy_score(test_y, pred_y)

    print("Accurate: {:.3f}".format(acc))
    print("Confusion Metric: \n{}".format(metric))

# 回归问题
def regression_method(MODEL):
    X, y = load_diabetes(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=11)

    model = MODEL()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    
    mse = mean_squared_error(test_y, pred_y)
    mae = mean_absolute_error(test_y, pred_y)

    print("mse: {:.3f}".format(mse))
    print("mae: {:.3f}".format(mae))


if __name__ == "__main__":

    classfication_method(KnnClassfier)
    print()
    regression_method(KnnRegressor)

    pass