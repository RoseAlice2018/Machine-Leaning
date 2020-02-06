""""
KNN实现
通过KNN来对数据进行分类预测
样本集：
1，1，A
1，2,A
1,5,1,5,A
3,4,B
4,4,B
预测数据：
2,2
算法实现思路：
1.计算预测样本和数据集中样本的距离
2.将所有的距离从大到小排序
3.计算前K个最小距离的类别的个数
4.返回前k个最小距离中个数最多的分类
"""
# import numpy as np
# import operator
#
#
# def data_deal(dat):
#     """
#     :param dat: 输入数据集
#     :return: 输出x，y
#     """
#     x = dat[:, :-1].astype(np.float)
#     y = dat[:, -1]
#     return x, y
#
#
# def knn_classifier(k, dataset, input):
#     """
#     :param k: k个邻居
#     :param dataset: 所有数据集
#     :param input: 预测数据
#     :return: 预测分类
#     """
#     x, y = data_deal(dat)
#     distance = (np.sum((input-x)**2, axis=1))**0.5
#     dist = np.argsort(distance)
#     count = 0
#     for i in range(k):
#         label = y[dist[i]]
#         count[label] = count.get(label, 0)+1
#         s = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
#         return s[0][0]
#
#
# if __name__ == "__main__":
#     dat = np.loadtxt("dat.txt", dtype=np.str, delimiter=",")
#
#     # 预测数据
#     predict = [2, 2]
#     print(knn_classifier(3, dat, predict))
import numpy as np
import operator


def handle_data(dataset):
    """
    :param dataset: 样本集
    :return: 输出 x和y
    """
    x = dataset[:, :-1].astype(np.float)
    y = dataset[:, -1]
    return x, y


def knn_classifier(k, dataset, input):
    """
    :param k:  k个邻居
    :param dataset: 所有数据集
    :param input: 输出的预测数据
    :return: 输出预测的分类
    """
    x, y = handle_data(dataset)
    distance = np.sum((input-x)**2, axis=1)**0.5
    sortedDist = np.argsort(distance)
    countLabel={}
    for i in range(k):
        label = y[sortedDist[i]]
        countLabel[label] = countLabel.get(label, 0)+1
    print(countLabel)
    sortedLabel = sorted(countLabel.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabel[0][0]


if __name__ == "__main__":
    dat = np.loadtxt("dat.txt", dtype=np.str, delimiter=",")
    predict_data =[4, 2]
    print(knn_classifier(4, dat, predict_data))
