#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

import numpy as np

class Kalman:
    def __init__(self, sx, sP, sv, sw, model, nx, ny, lamb, rhoQ, rhoR, Q=[], R=[]):
        '''
        カルマンフィルタの初期値、パラメータを設定する

        Inputs
        ======
        - sx    : 推定値の初期値
        - sP    : 推定値の分散の初期値
        - sv    : 分散Rの初期値
        - sw    : 分散Qの初期値
        - model : 遷移関数fと観測関数hが定義された状態空間モデル
        - nx    : 推定値の次元
        - ny    : 観測値の次元
        - lamb  : Unscented変換のパラメータ
        - rhoQ  : 変分ベイズのパラメータ
        - rhoR  : 変分ベイズのパラメータ
        - Q     : 遷移分布の分散(UKFのときに必要)
        - R     : 観測分布の分散(UKFのときに必要)
        '''

        self.x = sx
        self.P = sP
        self.v = sv
        self.V = np.identity(ny)
        self.w = sw
        self.W = np.identity(nx)

        self.e = 0

        self.f = model.f
        self.h = model.h

        self.nx = self.x.size
        self.ny = ny

        #Unscented変換で使う係数
        self.lamb = lamb
        self.Ws0 = self.lamb/(self.nx+self.lamb)
        self.Wc0 = self.lamb/(self.nx+self.lamb)
        self.Ws = 1/(2*(self.nx+self.lamb))
        self.Wc = self.Ws

        self.rhoQ = rhoQ
        self.rhoR = rhoR
        self.BQ = np.sqrt(self.rhoQ)*np.identity(self.nx)
        self.BR = np.sqrt(self.rhoR)*np.identity(self.ny)

        self.Q = Q
        self.R = R

    def ut(self, f, x, P):
        '''
        シグマ点を計算する

        Inputs
        ======
        - f : 非線形関数
        - x : 変換前の平均
        - P : 変換前の分散

        Returns
        =======
        - sigma : シグマ点
        - ita   : シグマ点に関数fを適用したもの
        '''

        sqrtP = np.linalg.cholesky(P)
        sigma = np.empty((2*self.nx+1, self.nx))
        sigma[0] = x
        for i in range(1, self.nx+1):#1~nxまで
            sigma[i] = x + np.sqrt(self.nx + self.lamb)*sqrtP[:,i-1]
        for i in range(self.nx + 1, 2*self.nx + 1):#nx+1から2nxまで
            sigma[i] = x - np.sqrt(self.nx + self.lamb)*sqrtP[:,i-self.nx-1]

        ita = np.array([f(sigma[i]) for i in range(2*self.nx+1)])

        return sigma, ita

    def sigmaMean(self, sigma):
        '''
        シグマ点の平均を計算する

        Inputs
        ======
        - sigma : シグマ点

        Returns
        =======
        - x_ : シグマ点の平均
        '''

        ws = [self.Ws for i in range(2*self.nx+1)]
        ws[0] = self.Ws0
        ws = np.array(ws)

        x_ = ws.dot(sigma)
        return x_

    def sigmaVar(self, x, x_, y, y_):
        '''
        シグマ点xとシグマ点yの重み付き共分散を計算する

        Inputs
        ======
        - x  : シグマ点
        - x_ : xの平均
        - y  : シグマ点
        - y_ : yの平均

        Returns
        =======
        - P_ : シグマ点の共分散
        '''

        P_ = self.Wc0*np.outer(x[0]-x_, y[0]-y_)
        for i in range(1, 2*self.nx+1):
            P_ += self.Wc*np.outer(x[i]-x_, y[i]-y_)

        return P_

    def sigmaVar2(self, y, ita):
        '''
        変換後のシグマ点の重み付き分散を計算する
        変分ベイズの計算に必要

        Inputs
        ======
        - y   : 変換後のシグマ点の平均
        - ita : 変換後のシグマ点

        Returns
        =======
        - P_ : シグマ点の分散
        '''

        P_ = self.Wc0*np.outer(y-ita[0], y-ita[0])
        for i in range(1, 2*self.nx+1):
            P_ += self.Wc*np.outer(y-ita[i], y-ita[i])
        return P_

    def ukf(self, y):
        '''
        Unscentedカルマンフィルタで推定値を計算
        - 状態方程式
            x ~ N(f(x), Q)
        - 観測方程式
            y ~ N(h(x), R)

        Inputs
        ======
        - y : 観測値

        Returns
        =======
        - x_ : 推定値
        - P_ : 推定値の分散
        - e  : 推定誤差の平均値
        '''

        x = self.x
        P = self.P

        if len(self.Q)==0 or len(self.R)==0:
            print('分散Q、分散Rを設定してください')

        # 予測ステップ
        sigma, ita = self.ut(self.f, x, P)

        x_ = self.sigmaMean(ita)
        P_ = self.sigmaVar(ita, x_, ita, x_) + self.Q

        # 更新ステップ
        sigma, ita = self.ut(self.h, x_, P_)
        x_ = self.sigmaMean(sigma)
        y_ = self.sigmaMean(ita)
        V = self.sigmaVar(ita, y_, ita, y_) + self.R
        U = self.sigmaVar(sigma, x_, ita, y_)
        e = y - y_

        K = U.dot(np.linalg.inv(V))
        x_ = x_ + K.dot(e)
        P_ = P_ - K.dot(U.T)

        self.x = x_
        self.P = P_
        self.e = e

        return x_, P_, np.average(np.abs(e))

    def vbukf(self, y, z):
        '''
        変分ベイズUnscentedカルマンフィルタで推定値を計算
        - 状態方程式
            x ~ N(f(x), Q)
            Q ~ IW(v, V)
        - 観測方程式
            y ~ N(h(x), R)
            R ~ IW(w, W)

        Inputs
        ======
        - y : 観測値
        - z : 分散Qの観測値

        Returns
        =======
        - x_ : 推定値
        - P_ : 推定値の分散
        - e  : 推定誤差の平均値
        '''
        x = self.x
        P = self.P
        v = self.v
        V = self.V
        w = self.w
        W = self.W

        # 予測ステップ
        sigma, ita = self.ut(self.f, x, P)

        v = self.rhoR*(v - self.ny - 1) + self.ny + 1
        V = self.BR.dot(V).dot(self.BR.T)

        w = self.rhoQ*(w - self.nx - 1) + self.nx + 1
        W = self.BQ.dot(W).dot(self.BQ.T)

        x_ = self.sigmaMean(ita)
        P_ = self.sigmaVar(ita, x_, ita, x_) + W/(w - self.nx - 1)

        # 更新ステップ
        for i in range(10):
            sigma, ita = self.ut(self.h, x_, P_)
            x_ = self.sigmaMean(sigma)
            y_ = self.sigmaMean(ita)
            e = y - y_
            T = self.sigmaVar(ita, y_, ita, y_)
            C = self.sigmaVar(sigma, x_, ita, y_)
            S = T + V/(v - self.ny - 1)

            K = C.dot(np.linalg.inv(S))
            x_ = x_ + K.dot(e)
            P_ = P_ - K.dot(S).dot(K.T)

            sigma, ita = self.ut(self.h, x_, P_)
            v = 1 + v
            V = V + self.sigmaVar2(y, ita)

            w = 1 + w
            W = W + np.outer(z, z)

        self.x = x_
        self.P = P_
        self.v = v
        self.V = V
        self.w = w
        self.W = W

        self.e = e

        return x_, P_, np.average(np.abs(e))
