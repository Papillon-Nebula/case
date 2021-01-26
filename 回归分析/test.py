from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


dataMat, labelMat = loadDataSet("回归分析-python3\\ex0.txt")
# standRegres(dataMat, labelMat)
# xMat = mat(dataMat)
# yMat = mat(labelMat)
# x = xMat[:,1].flatten().A[0]
# y = yMat.T[:,0].flatten().A[0]
# plt.scatter(x,y)
# plt.show()

ws = standRegres(dataMat,labelMat)
xMat = mat(dataMat)
yMat = mat(labelMat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()






