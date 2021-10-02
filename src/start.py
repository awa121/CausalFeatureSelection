import os
import time


def run(games):
    # curenv = os.environ.copy()
    cmds = []
    filePath = "/home/liuyijun/awa_project/hayaku/src/"
    methodStr = "_".join(games["methodList"])
    for _cancer in games["cancerList"]:
        for _seed in games["randomSeed"]:
            for _feature in games["featureNum"]:
                log = games["savePath"] + "/" + _cancer + methodStr + "Seed"+str(_seed) + "Feature"+str(_feature)+".log"
                cmd = ["nohup", "/home/liuyijun/anaconda3/bin/python",
                   filePath + games["fileName"],
                   "--cancer=" + _cancer,
                   "--methodList=" + methodStr,
                   "--featureNum="+str(_feature),
                   "--randomSeed=" + str(_seed),
                   "--savePath=" + games["savePath"],
                   ">", log, "2>&1 &"
                   ]
                cmds.append(cmd)
    while cmds:
        cmd = cmds.pop(0)
        print(cmd, 'start')
        os.system(" ".join(cmd))
        time.sleep(120)


if __name__ == "__main__":

    ls_date = time.strftime("%Y%m%d%H%M", time.localtime())
    savePath = "/home/liuyijun/awa_project/hayaku/data/results/" + ls_date
    print(savePath)
    os.mkdir(savePath)

    cancerList = ["KICH", "COAD", "THCA", "HNSC","ESCA", "BLCA"]
    methodList = ["ReliefF","MRMR", "MIFS", "RFS", "LAND", "CLAND-ReweiSample3"]


    featureNum=[5,10,15,20,25,30,35,40]
    randomSeed = [50]
    fileName = "CLAND.py"

    games = {"cancerList": cancerList, "methodList": methodList,
             "featureNum": featureNum, "randomSeed": randomSeed, "savePath": savePath, "fileName": fileName}
    run(games=games)
