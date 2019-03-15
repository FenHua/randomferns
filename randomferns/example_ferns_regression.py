#coding:utf-8
#!/usr/bin/python

import numpy as np
import cv2
from randomferns import *
import weakLearner
import itertools


def img_test(tree, points, colors, filename, size=512, radius=3):
    img = np.zeros((size,size,3), dtype='float')  # 创建一张图片
    v_min = points.min()  # 获取当前点集的最小值
    v_max = points.max()  # 获取当前点集的最大值
    step = float(v_max - v_min)/img.shape[0]  # 单位长
    grid = np.arange(v_min, v_max, step)  # 生成等差向量
    xy = np.array(list(itertools.product(grid, grid)))  # 笛卡尔积生成二维数组
    predictions = xy + tree.predict(xy)  # 预测值
    if len(predictions.shape)==2:
        predictions = predictions[np.newaxis,...]  # 数据不变，多添加一维
    predictions = predictions.reshape((predictions.shape[0]*predictions.shape[1], predictions.shape[2]))
    predictions = np.round((predictions-v_min)/step).astype('int32')   # 转换到像素位置
    flat_indices = np.ravel_multi_index(np.transpose(predictions), img.shape[:2], mode='clip')
    bins = np.bincount(flat_indices, minlength=np.prod(img.shape[:2]))
    img += bins.reshape(img.shape[:2])[...,np.newaxis]
    # 人工切除
    img[0] = 0
    img[-1] = 0
    img[:,0] = 0
    img[:,-1] = 0
    img *= 255/img.max()
    points = ((points - v_min)/step).astype('int')
    # 可视化
    for p,r in zip(points,responses):
        cv2.circle(img, tuple(p), radius+1, (0,0,0), thickness=-1)
        cv2.circle(img, tuple(p), radius, (0,255,0), thickness=-1)
    cv2.imwrite(filename,img.astype('uint8'))

        
t = np.linspace(0,2*np.pi,num=50)
radius = [30,60]
colors = np.array([[255,0,0],[0,255,0],[0,0,255]], dtype='float32')
points = np.zeros((len(t)*len(radius),2))
for r in range(len(radius)):
    points[r*len(t):(r+1)*len(t),0] = radius[r]*np.cos(t)  # 第一维
    points[r*len(t):(r+1)*len(t),1] = radius[r]*np.sin(t)  # 第二维
center = points.mean(axis=0) + 45*np.ones((2))/np.sqrt(2)
responses = center[np.newaxis,...] - points

for learner in weakLearner.__all__:
    print(learner)
    fern = Fern(depth=10, test_class=getattr(weakLearner, learner)(), regression=True)
    fern.fit(points,responses)
    img_test(fern, points, colors, 'img/fern_'+str(learner)+'_regression.png')
    randomferns = RandomFerns(depth=10, n_estimators=50,test_class=getattr(weakLearner, learner)(), regression=True)
    randomferns.fit(points, responses)
    img_test(randomferns, points, colors, 'img/randomferns_'+str(learner)+'_regression.png')


            
