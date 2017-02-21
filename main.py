#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import Kalman  as KF

#状態空間モデルを定義
class dlm:
    def __init__(self):
        pass

    def h(self, x):
        return x*x + x*x*x

    def f(self, x):
        return x

#状態空間モデルを作成
model = dlm()

#カルマンフィルタを作成
kalman = KF.Kalman(
        sx      = np.array([0, 1]),
        sP      = np.diag([0.1, 0.1]),
        sv      = 4,
        sw      = 4,
        model   = model,
        nx      = 2,
        ny      = 2,
        lamb    = 3,
        rhoQ    = 0.99,
        rhoR    = 0.99,
        Q       = np.diag([0.1, 0.1]),
        R       = np.diag([0.1, 0.1])
    )

#推定値
estimate = []

#フレーム数
T = 100

#正解値を作成
#x1は徐々に分散が大きくなる
x1 = np.sin(np.linspace(0, 10*np.pi, T))# + np.array([np.random.normal(0, 0.1*i) for i in range(T)])
x2 = 2*np.sin(np.linspace(0, 10*np.pi, T) + (np.pi/3))# + np.array([np.random.normal(0, 0.01) for i in range(T)])
answer = np.array([x1, x2]).T

#観測値を作成
w = np.array([[np.random.normal(0, 0.01) for i in range(2)] for j in range(T)])
observe = model.h(answer) #+ w

for t in range(T):
    #最初の2フレームは観測値zを計算できないので適当に決める
    if t<2:
        z = [0, 0]
    else:
        z = estimate[t-1] - estimate[t-2]

    #通常のUnscentedカルマンフィルタで推定
    #x, P, e = kalman.ukf(observe[t])

    #変分ベイズUnscentedカルマンフィルタで推定
    x, P, e = kalman.vbukf(observe[t], z)
    estimate.append(x)

plt.figure()
plt.plot(answer.T[0], 'r--')
plt.plot(np.array(estimate).T[0], 'g')
plt.plot(observe.T[0], 'b--')

plt.figure()
plt.plot(answer.T[1], 'r--')
plt.plot(np.array(estimate).T[1], 'g')

plt.show()
