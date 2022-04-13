# sample weighting
# Inference: <Stable Gene Selection from Microarray Data via Sample Weighting>
import numpy as np
from joblib import Parallel, delayed


def spaceTrans(i, j, data, label):
    # sample i and its jth feature
    # Eq.2
    sum = 0
    for _ in range(len(data[:, j])):
        if label[_] != label[i]:
            sum += abs(data[i, j] - data[_, j])
        else:
            sum -= abs(data[i, j] - data[_, j])
    return sum


def dist(x, X):
    n = len(X)
    result = 0
    for i in range(len(X)):
        result += np.sqrt(np.sum(x - X[i]) ** 2)
    return result / (n - 1)


def calSampleWeight(marginSpace):
    # calculate weight of sample ith
    # Eq.3
    weights = np.zeros(len(marginSpace))
    for i in range(len(weights)):
        weights[i] = dist(marginSpace[i], marginSpace)
    return weights


def parallelSpaceTrans(i, j, data, label):
    return spaceTrans(i, j, data, label)


def marginSampleWeight(data, label):
    # feature space transformation
    marginSpace = np.zeros(np.shape(data))
    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         marginSpace[i,j]=spaceTrans(i,j,data,label)
    marginSpace = Parallel(n_jobs=-1)(
        [delayed(spaceTrans)(i, j, data, label) for i in range(len(data)) for j in range(len(data[i]))])
    marginSpace = np.reshape(marginSpace, data.shape)
    # sample weighting
    weightsY = calSampleWeight(marginSpace)
    weightsY = normalize(weightsY)
    weightsY = splitInCan(weightsY)
    prob_all = 0
    clf = 0
    print("!!!", weightsY)
    return prob_all, clf, weightsY


def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def splitInCan(weightsY, canNum=10):
    from math import ceil
    _can = 1 / canNum
    for i in range(len(weightsY)):
        weightsY[i] = ceil(weightsY[i] / _can) * _can
    return weightsY
