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

    raw_data, label, raw_feature, feature = feature_selection(method, args, setting)

    data = [raw_data[i, :] for i in feature]
    data = np.array(data)

    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'GBDT': gradient_boosting_classifier,
                   }

    print('reading training and testing data...')

    train_x, train_y, test_x, test_y = read_data(data, label, ratio=setting["trainTestRatio"])
    imblance_ratio = ""
    for i in np.unique(label):
        imblance_ratio = imblance_ratio + str(np.sum(label == i)) + "_" + str(np.sum(test_y == i)) + "/"
    with open(setting["resultFile"], "a+") as f:
        f.write(imblance_ratio + ",")
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

    averageClassifierAUCPRC=0
    averageClassifierF1=0
    averageClassifierGM=0
    averageClassifierMCC=0
    for classifier in setting["classifiers"]:
        nor_AUCPRC = []
        nor_F1 = []
        nor_GM = []
        nor_MCC = []

        nor_accuracy = 0
        nor_F1_score = 0
        nor_balanced_acc = 0
        randomSeed = args.randomSeed
        for repeat in range(setting["repectNum"]):

            print('******************* %s ********************' % classifier)
            if classifier == 'GBDT-feature-selection':
                train_x, train_y, test_x, test_y = read_data(raw_data, label, randomseed=randomSeed, ratio=0.8)
                num_train, num_feat = train_x.shape
                num_test, num_feat = test_x.shape
                is_binary_class = (len(np.unique(train_y)) == 2)
                print('******************** Data Info *********************')
                print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
                start_time = time.time()
                model = classifiers[classifier](train_x, train_y, raw_feature, args.featureNum)
            else:
                train_x, train_y, test_x, test_y = read_data(data, label, randomseed=randomSeed, ratio=0.8)

                start_time = time.time()
                model = classifiers[classifier](train_x, train_y)
            print(classifier)
            print('training took %.2fs!' % (time.time() - start_time))
            predict = model.predict(test_x)

            if (args.labelNum == 2):
                AUCPRC = auc_prc(test_y, predict)
                F1 = f1_optim(test_y, predict)
                GM = gm_optim(test_y, predict)
                MCC = mcc_optim(test_y, predict)

                nor_AUCPRC.append(AUCPRC)
                nor_F1.append(F1)
                nor_GM.append(GM)
                nor_MCC.append(MCC)
                print('AUCPRC: %.2f%%' % (100 * AUCPRC))
                print('F1: %.2f%%' % (100 * F1))
                print('GM: %.2f%%' % (100 * GM))
                print('MCC: %.2f%%' % (100 * MCC))
            # for muli class
            else:
                accuracy = metrics.accuracy_score(test_y, predict)
                F1_score = metrics.f1_score(test_y, predict, average="macro")
                balanced_acc = metrics.balanced_accuracy_score(test_y, predict)

                nor_accuracy += accuracy
                nor_F1_score += F1_score
                nor_balanced_acc += balanced_acc

            randomSeed += 1
        if (args.labelNum == 2):
            nor_AUCPRC, nor_AUCPRC_std = np.mean(nor_AUCPRC), np.std(nor_AUCPRC)
            nor_F1, nor_F1_std = np.mean(nor_F1), np.std(nor_F1)
            nor_GM, nor_GM_std = np.mean(nor_GM), np.std(nor_GM)
            nor_MCC, nor_MCC_std = np.mean(nor_MCC), np.std(nor_MCC)

            print('AUCPRC: %.2f%%' % (100 * nor_AUCPRC))
            print('F1: %.2f%%' % (100 * nor_F1))
            print('GM: %.2f%%' % (100 * nor_GM))
            print('MCC: %.2f%%' % (100 * nor_MCC))

            averageClassifierAUCPRC += nor_AUCPRC
            averageClassifierF1 += nor_F1
            averageClassifierGM += nor_GM
            averageClassifierMCC += nor_MCC

            nor_AUCPRC, nor_AUCPRC_std, nor_F1, nor_F1_std, nor_GM, nor_GM_std, nor_MCC, nor_MCC_std = map(
                lambda x: str(round(x, 3)),
                [nor_AUCPRC, nor_AUCPRC_std, nor_F1, nor_F1_std, nor_GM, nor_GM_std, nor_MCC, nor_MCC_std])
            kkk = nor_AUCPRC + "^" + nor_AUCPRC_std + "_" + nor_F1 + "^" + nor_F1_std + "_" + nor_GM + "^" + nor_GM_std + "_" + nor_MCC + "^" + nor_MCC_std

        else:
            nor_accuracy = nor_accuracy / setting["repectNum"]
            nor_F1_score = nor_F1_score / setting["repectNum"]
            nor_balanced_acc = nor_balanced_acc / setting["repectNum"]
            print('accuracy: %.2f%%' % (100 * nor_accuracy))
            print('F1_score_macro accuracy: %.2f%%' % (100 * nor_F1_score))
            print('balanced accuracy: %.2f%%' % (100 * nor_balanced_acc))
            kkk = str(round(nor_accuracy, 3)) + "_" + str(round(nor_F1_score, 3)) + "_" + str(
                round(nor_balanced_acc, 3)) + "/"
        if setting["repectNum"] == 1:
            t = classification_report(test_y, predict, target_names=target_names)
            # t=classification_report(test_y,predict,target_names=["0","1","2","3","4"])
            print(t)
        with open(setting["resultFile"], "a+") as f:
            f.write(kkk + ",")
    with open(setting["resultFile"], "a+") as f:
        f.write("\n")

    averageClassifierAUCPRC /=len(setting["classifiers"])
    averageClassifierF1  /=len(setting["classifiers"])
    averageClassifierGM /=len(setting["classifiers"])
    averageClassifierMCC /=len(setting["classifiers"])


    return averageClassifierAUCPRC,averageClassifierF1,averageClassifierGM,averageClassifierMCC
