%MTLAB_R2015bで作成
clear

%カルマンフィルタを作成
sx      = [0, 1];
sP      = diag([0.1, 0.1]);
sv      = 4;
sw      = 4;
nx      = 2;
ny      = 2;
lamb    = 3;
rhoQ    = 0.99;
rhoR    = 0.99;
Q       = diag([0.1, 0.1]);
R       = diag([0.1, 0.1]);

kalman = Kalman(sx, sP, sv, sw, nx, ny, lamb, rhoQ, rhoR, Q, R);

%フレーム数
T = 100;
%推定値
estimate = zeros(T, 2);

%正解値を作成
%x1は徐々に分散が大きくなる
x1 = sin(linspace(0, 10*pi, T));
x2 = 2*sin(linspace(0, 10*pi, T) + (pi/3));
for i = 1:T
    x1(i) = x1(i) + normrnd(0, i*0.005);
    x2(i) = x2(i) + normrnd(0, 0.01);
end

answer = transpose([x1; x2]);

%観測値を作成
observe = ones(T, 2);
for i = 1:T
    w = [normrnd(0, 0.001), normrnd(0, 0.01)];
    observe(i,:) = dlm.h(answer(i,:)) + w;
end


for t = 1:T
    %最初の2フレームは観測値zを計算できないので適当に決める
    if t<3
        z = [0, 0];
    else
        z = estimate(t-1, :) - estimate(t-2, :);
    end

    %通常のUnscentedカルマンフィルタで推定
    [kalman, x, P, e] = kalman.ukf(observe(t, :));

    %変分ベイズUnscentedカルマンフィルタで推定
    %[kalman, x, P, e] = kalman.vbukf(observe(t, :), z);
    estimate(t, :) = x;
end

figure()
plot(answer(:,1))
hold
plot(estimate(:,1))

figure()
plot(answer(:,2))
hold
plot(estimate(:,2))
