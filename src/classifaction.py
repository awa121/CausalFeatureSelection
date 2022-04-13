# !usr/bin/env python
# -*- coding: utf-8 -*-

import time
from sklearn import metrics
from src.awapyHSICLasso.api import HSICLasso
import heapq
from sklearn.metrics import classification_report
from utils import *
import pandas as pd


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model

# split data into train and test data by ratio. ratio=dom(train data)/dom(data)
def read_data(data, label, randomseed=0, ratio=0.8):
    print("The randomseed is", randomseed)
    np.random.seed(randomseed)
    _data = np.concatenate((label, data), axis=0).T
    n_total = len(_data)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], data
    # if shuffle:
    np.random.shuffle(_data)
    train_x, train_y = _data[:offset, 1:], _data[:offset, 0]
    test_x, test_y = _data[offset:, 1:], _data[offset:, 0]
    while len(np.unique(train_y)) != len(np.unique(test_y)):
        randomseed += 10
        print("random_seed needs to be changed as ", randomseed)
        np.random.shuffle(_data)
        train_x, train_y = _data[:offset, 1:], _data[:offset, 0]
        test_x, test_y = _data[offset:, 1:], _data[offset:, 0]

    return train_x, train_y, test_x, test_y


def selected_feature(feature, featname):
    # trans feature number to feature name(gene name)
    result = ""
    for i in range(len(feature)):
        result = result + featname[feature[i]] + "/"
    return result

def feature_selection(method, args, setting):
    # data_file: *.csv
    # the zero line is ‘class,feature1,feature2,feature3.....’
    # the one~last line is 'label,num1,num2,num3.....'
    data = pd.read_csv(setting["exDataFile"], sep=",",encoding="gbk")

    X = np.array(data[0:])[:, 1:]
    y = np.array(data["class"])

    featname = data.columns.tolist()[1:]
    with open(setting["resultFile"], "a+") as f:
        f.write(str(len(X.T)) + "," + str(args.featureNum) + ",")

    if "LAND" in method:
        hsic_lasso = HSICLasso()
        hsic_lasso.input(setting["exDataFile"])
        hsic_lasso.classification(method, args, setting)

        feature_selected = hsic_lasso.dump()
        print("Selected feature:",feature_selected)
        print("Score:",hsic_lasso.get_index_score())
        # hsic_lasso.save_param()
        with open(setting["resultFile"], "a+") as f:
            f.write(feature_selected + ",")

        raw_data = hsic_lasso.X_in
        label = hsic_lasso.Y_in
        raw_feature = hsic_lasso.featname
        feature = hsic_lasso.get_index()

        return raw_data, label, raw_feature, feature
    # comparinng methods according Hsiclasso

    elif method == "ReliefF":
        from skfeature.function.similarity_based import reliefF
        score = reliefF.reliefF(X, y)
        idx = reliefF.feature_ranking(score)
        feature = idx[:args.featureNum]
        with open(setting["resultFile"], "a+") as f:
            f.write(selected_feature(feature, featname) + ",")
        return X.T, y.reshape(1, len(y)), featname, feature
    elif method == "MRMR":
        from skfeature.function.information_theoretical_based import MRMR
        score = MRMR.mrmr(X, y, n_selected_features=args.featureNum)
        feature = score[0]
        with open(setting["resultFile"], "a+") as f:
            f.write(selected_feature(feature, featname) + ",")
        return X.T, y.reshape(1, len(y)), featname, feature
    elif method == "MIFS":
        from skfeature.function.information_theoretical_based import MIFS
        score = MIFS.mifs(X, y, n_selected_features=args.featureNum)
        feature = score[0]
        with open(setting["resultFile"], "a+") as f:
            f.write(selected_feature(feature, featname) + ",")
        return X.T, y.reshape(1, len(y)), featname, feature
    elif method == "RFS":
        from skfeature.function.sparse_learning_based import RFS
        _y = y.reshape((len(y), 1))
        score = RFS.rfs(X, _y)
        from skfeature.function.similarity_based.reliefF import feature_ranking
        feature = feature_ranking(score)[:args.featureNum]
        feature = feature.reshape((args.featureNum))
        with open(setting["resultFile"], "a+") as f:
            f.write(selected_feature(feature, featname) + ",")
        return X.T, y.reshape(1, len(y)), featname, feature




def classifaction(method, args, setting):
    target_names = [str(i) for i in range(args.labelNum)]
    with open(setting["resultFile"], "a+") as f:
        f.write(method + "," + args.cancer + "," + str(args.labelNum) + ",")

    print(method)
    print(setting["resultFile"])


    data = pd.read_csv(setting["exDataFile"], sep=",", encoding="gbk")
    raw_X = np.array(data[0:])[:, 1:]
    raw_Y = np.array(data["class"])
    featuresName = data.columns.tolist()[1:]

    imblance_ratio =str(np.sum( raw_Y== 1)) + "_" + str(np.sum(raw_Y == 0))
    print(imblance_ratio)
    with open(setting["resultFile"], "a+") as f:
        f.write(imblance_ratio + ",")

    randomSeed = args.randomSeed
    featuresList=[]
    measure4=np.zeros((len(setting["classifiers"]),4))
    for _ in range(setting["repectNum"]):
        print("The",_,"repect.")
        train_x, train_y, test_x, test_y,randomSeed = read_data(raw_X, raw_Y, randomseed=randomSeed,ratio=setting["trainTestRatio"])
        randomSeed+=1
        logTrainX, logTrainY, raw_feature, features = feature_selection(train_x, train_y, featuresName, method, args, setting)
        featuresList.append(features)

        _result=classifiers(features,trainX=train_x,trainY=train_y,testX=test_x,testY=test_y,setting=setting)
        measure4=measure4+_result

    measure4=measure4/setting["repectNum"]
    averageAUCPRC = round(statistics.mean(measure4[:,0]),3)
    averageF1 = round(statistics.mean(measure4[:,1]),3)
    averageGM = round(statistics.mean(measure4[:,2]),3)
    averageMCC = round(statistics.mean(measure4[:,3]),3)
    print('Average AUCPRC: %.2f%%' % (100 * averageAUCPRC))
    print('Average F1: %.2f%%' % (100 * averageF1))
    print('Average GM: %.2f%%' % (100 * averageGM))
    print('Average MCC: %.2f%%' % (100 * averageMCC))

    with open(setting["resultFile"], "a+") as f:
        f.write(str(len(raw_X.T)) + "," + str(args.featureNum) + ",")

    finalFeature = selected_feature(featuresList, featuresName,args.featureNum)
    with open(setting["resultFile"], "a+") as f:
        f.write(finalFeature + ",")

    classifiersResult = ""
    for _cl in measure4:
        for _matrix in _cl:
            classifiersResult+= str(round(_matrix,3))+"_"
        classifiersResult=classifiersResult[:-1]+","
    with open(setting["resultFile"], "a+") as f:
         f.write(classifiersResult[:-1] + "\n")


    return averageAUCPRC, averageF1, averageGM, averageMCC