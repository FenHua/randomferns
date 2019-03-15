# coding:utf-8
import numpy as np
from functools import reduce
from weakLearner import WeakLearner, AxisAligned


# 厥类
class Fern(object):
    def __init__( self, depth=10, test_class=AxisAligned(), regression=False ):
        self.depth = depth  # 深度
        self.test_class = test_class  # 弱分类器
        self.regression = regression  # 是否回归标记

    def apply_tests(self, points):
        '''
        python中的reduce内建函数是一个二元操作函数
        它用来将一个数据集合（链表，元组等）中的所有数据进行下列操作：
        用传给reduce中的函数 func()（必须是一个二元操作函数）先对集合中的第1，2个数据进行操作，得到的结果再
        与第三个数据用func()函数运算，最后得到一个结果。
        '''
        return reduce(lambda x, y: 2*x+y, self.test_class.run(points, self.tests))

    def fit(self, points, responses):
        # 训练函数
        self.tests = np.array(self.test_class.generate_all(points, self.depth))  # 生成单个蕨对应的特征
        if self.regression:
            # 回归操作
            self.target_dim = responses.shape[1]  # 获取label的数量
            self.data = np.zeros((2**self.depth, self.target_dim), dtype='float64')  # 开辟一个data空间大小为2**depth × target_dim
            bins = self.apply_tests(points)  # 获取bins的量
            bincount = np.bincount(bins, minlength=self.data.shape[0])  # 统计bins元素的个数，返回list的长度大于等于minlength
            for dim in range(self.target_dim):
                # 更新data 根据weights的值统计得到
                self.data[:,dim] += np.bincount(bins, weights=responses[:,dim], minlength=self.data.shape[0])
            self.data[bincount>0] /= bincount[bincount>0][...,np.newaxis]  # np.newaxis 在使用和功能上等价于 None
        else:
            # 分类操作
            self.n_classes = responses.max() + 1  # 获取类别数
            self.data = np.ones((2**self.depth, int(self.n_classes)), dtype='float64')  # 开辟一个data空间大小为2**depth × target_dim
            self.data[self.apply_tests(points), responses.astype('int32')] += 1  # 更新data
            self.data /= points.shape[0] + self.n_classes
            # maximising the product is the same as maximising the sum of logarithms
            self.data = np.log(self.data)
        
    def _predict(self, points):
        return self.data[self.apply_tests(points)]  # 返回回归预测值

    def predict(self, points):
        # 预测
        if self.regression:
            return self._predict(points)
        else:
            return np.argmax(self._predict(points), axis=1)  # 分类使用

# 随机厥
class RandomFerns(object):
    def __init__(self, depth=10, n_estimators=50, bootstrap=0.7, test_class=AxisAligned(), regression=False):
        self.depth = depth  # 深度
        self.test_class = test_class  # 弱分类器
        self.n_estimators = n_estimators  # 弱分类器的个数
        self.bootstrap = bootstrap  # 自举系数
        self.regression = regression  # 是否回归标识

    def fit(self, points, responses):
        # 训练
        self.ferns = []  # 厥
        for i in range(self.n_estimators):
            subset = np.random.randint(0, points.shape[0], int(self.bootstrap*points.shape[0]))  # 自举操作
            self.ferns.append(Fern(self.depth, self.test_class, regression=self.regression))
            self.ferns[-1].fit(points[subset], responses[subset])

    def _predict(self, points):
        return np.sum(list(map(lambda x: x._predict(points), self.ferns)), axis=0)  # 每个厥预测结果的和

    def predict(self, points):
        if self.regression:
            return np.array(list(map(lambda x:x._predict(points), self.ferns)))  # 回归
        else:
            return np.argmax(self._predict(points), axis=1)  # 分类

    def predict_proba(self, points):
        # 针对分类结果，返回各类别的概率的大小
        if self.regression:
            raise NotImplemented("predict_proba is not implemented for regression")
        proba = np.exp(self._predict(points))  # 指数化
        return proba / proba.sum(axis=1)[...,np.newaxis]  # 转换为概率
        
