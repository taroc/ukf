classdef Kalman
    properties
        x
        P
        v
        V
        w
        W

        e

        nx
        ny

        lamb
        Ws0
        Wc0
        Ws
        Wc

        rhoQ
        rhoR
        BQ
        BR

        Q
        R
    end
    methods
        function self = Kalman(sx, sP, sv, sw, nx, ny, lamb, rhoQ, rhoR, Q, R)
            %カルマンフィルタの初期値、パラメータを設定する
			%
	        %Inputs
	        %======
	        %- sx    : 推定値の初期値
	        %- sP    : 推定値の分散の初期値
	        % - sv    : 分散Rの初期値
	        %- sw    : 分散Qの初期値
	        %- model : 遷移関数fと観測関数hが定義された状態空間モデル
	        %- nx    : 推定値の次元
	        %- ny    : 観測値の次元
	        %- lamb  : Unscented変換のパラメータ
	        %- rhoQ  : 変分ベイズのパラメータ
	        %- rhoR  : 変分ベイズのパラメータ
	        %- Q     : 遷移分布の分散(UKFのときに必要)
	        %- R     : 観測分布の分散(UKFのときに必要)

            self.x = sx;
            self.P = sP;
            self.v = sv;
            self.V = eye(ny);
            self.w = sw;
            self.W = eye(nx);

            self.e = 0;

            self.nx = nx;
            self.ny = ny;

            %Unscented変換で使う係数
            self.lamb = lamb;
            self.Ws0 = lamb/(nx+lamb);
            self.Wc0 = lamb/(nx+lamb);
            self.Ws = 1/(2*(nx+lamb));
            self.Wc = self.Ws;

            self.rhoQ = rhoQ;
            self.rhoR = rhoR;
            self.BQ = sqrt(rhoQ)*eye(nx);
            self.BR = sqrt(rhoR)*eye(ny);

            self.Q = Q;
            self.R = R;
        end
        
        function [sigma, ita] = ut(self, f, x, P)
            %シグマ点を計算する
			%
	        %Inputs
	        %======
	        %- f : 非線形関数
	        %- x : 変換前の平均
	        %- P : 変換前の分散
			%
	        %Returns
	        %=======
	        %- sigma : シグマ点
	        %- ita   : シグマ点に関数fを適用したもの

            sqrtP = chol(P, 'lower');
            sigma = zeros(2*self.nx+1, self.nx);
            ita = zeros(2*self.nx+1, self.nx);
            sigma(1,:) = x;
            
            %1~nxまで
            for i = 2:self.nx+1
                sigma(i, :) = x + transpose(sqrt(self.nx + self.lamb)*sqrtP(:, i-1));
            end
            %nx+1から2nxまで
            for i = self.nx+2:2*self.nx+1
                sigma(i, :) = x - transpose(sqrt(self.nx + self.lamb)*sqrtP(:, i-self.nx-1));
            end
            for i = 1:2*self.nx+1
                ita(i, :) = f(sigma(i, :));
            end
        end
        function x_ = sigmaMean(self, sigma)
            %シグマ点の平均を計算する
			%
	        %Inputs
	        %======
	        %- sigma : シグマ点
			%
	        %Returns
	        %=======
	        %- x_ : シグマ点の平均
	                    
            ws = self.Ws*ones(1, 2*self.nx+1);
            ws(1) = self.Ws0;
            x_ = ws*sigma;
        end
        
        function out = outer(self, a, b)
            out = zeros(length(a), length(b));
            for i = 1:length(a)
                for j = 1:length(b)
                    out(i, j) = a(i) * b(j);
                end
            end
        end

        function P_ = sigmaVar(self, x, x_, y, y_)
            %シグマ点xとシグマ点yの重み付き共分散を計算する
			%
	        %Inputs
	        %======
	        %- x  : シグマ点
	        %- x_ : xの平均
	        %- y  : シグマ点
	        %- y_ : yの平均
			%
	        %Returns
	        %=======
	        %- P_ : シグマ点の共分散
            
            P_ = self.Wc0*self.outer(x(1,:)-x_, y(1,:)-y_);
            for i = 2:2*self.nx+1
                P_ = P_ + self.Wc*self.outer(x(i,:)-x_, y(i,:)-y_);
            end
        end
        
        function P_ = sigmaVar2(self, y, ita)
            %変換後のシグマ点の重み付き分散を計算する
	        %変分ベイズの計算に必要
			%
	        %Inputs
	        %======
	        %- y   : 変換後のシグマ点の平均
	        %- ita : 変換後のシグマ点
			%
	        %Returns
	        %=======
	        %- P_ : シグマ点の分散
			
            P_ = self.Wc0*self.outer(y-ita(1,:), y-ita(1,:));
            for i = 2:2*self.nx+1
                P_ = P_ + self.Wc*self.outer(y-ita(i,:), y-ita(i,:));
            end
        end
        
        function [self, x_, P_, e] = ukf(self, y)
            %Unscentedカルマンフィルタで推定値を計算
	        %- 状態方程式
	        %    x ~ N(f(x), Q)
	        %- 観測方程式
	        %   y ~ N(h(x), R)
			%
	        %Inputs
	        %======
	        %- y : 観測値
			%
	        %Returns
	        %=======
	        %- x_ : 推定値
	        %- P_ : 推定値の分散
	        %- e  : 推定誤差
	        
	        %予測ステップ
            [sigma, ita] = self.ut(@dlm.f, self.x, self.P);
            x_ = self.sigmaMean(ita);
            P_ = self.sigmaVar(ita, x_, ita, x_) + self.Q;

            % 更新ステップ
            [sigma, ita] = self.ut(@dlm.h, x_, P_);
            x_ = self.sigmaMean(sigma);
            y_ = self.sigmaMean(ita);
            V = self.sigmaVar(ita, y_, ita, y_) + self.R;
            U = self.sigmaVar(sigma, x_, ita, y_);
            e = y - y_;

            K = U * inv(V);
            x_ = x_ + (K * e')';
            P_ = P_ - K * U';

            self.x = x_;
            self.P = P_;
            self.e = e;
        end
        
        
        function [self, x_, P_, e] = vbukf(self, y, z)
            %変分ベイズUnscentedカルマンフィルタで推定値を計算
	        %- 状態方程式
	        %   x ~ N(f(x), Q)
	        %    Q ~ IW(v, V)
	        %- 観測方程式
	        %   y ~ N(h(x), R)
	        %    R ~ IW(w, W)
			%
	        %Inputs
	        %======
	        %- y : 観測値
	        %- z : 分散Qの観測値
			%	
	        %Returns
	        %=======
	        %- x_ : 推定値
	        %- P_ : 推定値の分散
	        %- e  : 推定誤差の平均値
        
            x = self.x;
            P = self.P;
            v = self.v;
            V = self.V;
            w = self.w;
            W = self.W;

            % 予測ステップ
            [sigma, ita] = self.ut(@dlm.f, x, P);

            v = self.rhoR*(v - self.ny - 1) + self.ny + 1;
            V = self.BR * V * self.BR';

            w = self.rhoQ*(w - self.nx - 1) + self.nx + 1;
            W = self.BQ * W * self.BQ';
            x_ = self.sigmaMean(ita);
            P_ = self.sigmaVar(ita, x_, ita, x_) + W/(w - self.nx - 1);

            % 更新ステップ
            for i = 1:10
                [sigma, ita] = self.ut(@dlm.h, x_, P_);
                x_ = self.sigmaMean(sigma);
                y_ = self.sigmaMean(ita);
                e = y - y_;
                T = self.sigmaVar(ita, y_, ita, y_);
                C = self.sigmaVar(sigma, x_, ita, y_);
                S = T + V/(v - self.ny - 1);

                K = C * inv(S);
                x_ = x_ + (K * e')';
                P_ = P_ - K * S * K';
                [sigma, ita] = self.ut(@dlm.h, x_, P_);
                v = 1 + v;
                V = V + self.sigmaVar2(y, ita);

                w = 1 + w;
                W = W + self.outer(z, z);
            end
            self.x = x_;
            self.P = P_;
            self.v = v;
            self.V = V;
            self.w = w;
            self.W = W;

            self.e = e;
        end
    end
end






