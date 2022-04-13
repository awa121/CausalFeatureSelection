import sys
import os
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src.classifaction import classifaction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection by HSIC lasso")
    parser.add_argument('--cancer', type=str, default="BRCA")
    parser.add_argument('--methodList', type=str, default="CLAND")
    parser.add_argument('--labelNum', type=int, default=2,
                        help="The number of class in the datasets")
    parser.add_argument('--randomSeed', type=int, default=50)
    parser.add_argument('--featureNum', type=int, default=10)
    parser.add_argument('--savePath', type=str,
                        default="/home/liuyijun/awa_project/hayaku/data/results",
                        help="The path of result saved in .")

    args = parser.parse_args()
    print(args)

    classifiersList = ["NB", "KNN", "LR", "RF", "GBDT"]
    exDataPath = "/home/liuyijun/awa_project/hayaku/data/TCGA/"

    methods=args.methodList.split("_")

    test_classifiers = ["NB", "KNN", "LR", "RF", "GBDT"]
    setting = {"exDataFile":exDataPath + args.cancer + "/allfile_with_class_normal_tumor.csv",
               "trainTestRatio":0.8,"classifiers":test_classifiers,"B":10,"M":3,"shuffle":False,
               "umbalance":False,"nlars":"nlarsOrg","repectNum":10,
               "resultFile":args.savePath+"/"+args.cancer+"Seed"+str(args.randomSeed)+"Feature"+str(args.featureNum)+".csv"}

    with open(setting["resultFile"], "w") as f:
        f.write("parameters,dataset,label,Feature number,select Feature number,selected Feature,unbalance ratio,NB,KNN,LR,RF,SVM,GDBT\n============================\n")

    result=[]
    for method in methods:
        AUCPRC,F1,GM,MCC=classifaction(method,args,setting)
        _str=method+","+str(AUCPRC)+","+str(F1)+","+str(GM)+","+str(MCC)
        print(_str)
        result.append(_str)
    f=open(setting["resultFile"], "a+")
    for line in result:
        f.write(line+"\n")
    f.close()
