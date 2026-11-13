% 9. 微分方程模型实现
% 功能：求解常微分方程和方程组
% 示例：人口增长模型、传染病SIR模型

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 人口增长模型（Logistic模型）
disp('=== 人口增长模型 ===');

% 定义微分方程: dP/dt = r*P*(1-P/K)
% r: 增长率, K: 环境容纳量
function dPdt = logisticModel(t, P, r, K)
    dPdt = r * P * (1 - P/K);
end

% 参数设置
r = 0.02;    % 增长率
K = 1000;    % 环境容纳量
P0 = 100;    % 初始人口
tSpan = [0, 200];  % 时间范围

% 求解微分方程
[t, P] = ode45(@(t,P) logisticModel(t, P, r, K), tSpan, P0);

% 绘制结果
figure;
plot(t, P, 'b', 'LineWidth', 1.5);
hold on;
plot(t, K*ones(size(t)), 'r--', 'LineWidth', 1);  % 环境容纳量
title('Logistic人口增长模型');
xlabel('时间');
ylabel('人口数量');
legend('人口数量', '环境容纳量');
grid on;

% 不同初始条件的对比
P0_values = [50, 200, 1500];  % 不同初始人口
figure;
for i = 1:length(P0_values)
    [t, P] = ode45(@(t,P) logisticModel(t, P, r, K), tSpan, P0_values(i));
    plot(t, P, 'LineWidth', 1.5);
    hold on;
end
plot(t, K*ones(size(t)), 'r--', 'LineWidth', 1);
title('不同初始条件下的人口增长');
xlabel('时间');
ylabel('人口数量');
legend('初始人口50', '初始人口200', '初始人口1500', '环境容纳量');
grid on;

% 2. 传染病SIR模型
disp('\n=== 传染病SIR模型 ===');

% SIR模型微分方程组:
% dS/dt = -beta*S*I/N
% dI/dt = beta*S*I/N - gamma*I
% dR/dt = gamma*I
% S: 易感者, I: 感染者, R: 康复者
% beta: 感染率, gamma: 康复率, N: 总人口

function dydt = sirModel(t, y, beta, gamma, N)
    S = y(1);
    I = y(2);
    R = y(3);
    
    dSdt = -beta * S * I / N;
    dIdt = beta * S * I / N - gamma * I;
    dRdt = gamma * I;
    
    dydt = [dSdt; dIdt; dRdt];
end

% 参数设置
N = 1000;       % 总人口
I0 = 1;         % 初始感染者
R0 = 0;         % 初始康复者
S0 = N - I0 - R0;  % 初始易感者
y0 = [S0; I0; R0];  % 初始条件

beta = 0.3;     % 感染率
gamma = 0.1;    % 康复率
tSpan = [0, 100];  % 时间范围

% 求解微分方程组
[t, y] = ode45(@(t,y) sirModel(t, y, beta, gamma, N), tSpan, y0);
S = y(:,1);
I = y(:,2);
R = y(:,3);

% 绘制结果
figure;
plot(t, S, 'b', t, I, 'r', t, R, 'g', 'LineWidth', 1.5);
title('SIR传染病模型');
xlabel('时间');
ylabel('人数');
legend('易感者', '感染者', '康复者');
grid on;

% 不同感染率的对比
beta_values = [0.2, 0.3, 0.4];
figure;
for i = 1:length(beta_values)
    [t, y] = ode45(@(t,y) sirModel(t, y, beta_values(i), gamma, N), tSpan, y0);
    plot(t, y(:,2), 'LineWidth', 1.5);
    hold on;
end
title('不同感染率下的感染者数量');
xlabel('时间');
ylabel('感染者数量');
legend('beta=0.2', 'beta=0.3', 'beta=0.4');
grid on;

% 3. 实际应用案例：药物动力学模型
function pharmacokinetics_demo()
    clear; clc;
    
    % 药物动力学一室模型
    % 药物在血液中的浓度变化
    % dC/dt = (F*D)/(V*Ka) * exp(-Ka*t) - Ke*C
    % F: 生物利用度, D: 剂量, V: 分布容积
    % Ka: 吸收速率常数, Ke: 消除速率常数
    
    function dCdt = drugModel(t, C, F, D, V, Ka, Ke)
        if t == 0
            absorption = 0;
        else
            absorption = (F*D)/(V*Ka) * exp(-Ka*t);
        end
        elimination = Ke * C;
        dCdt = absorption - elimination;
    end
    
    % 参数设置
    F = 0.8;    % 生物利用度
    D = 100;    % 剂量 (mg)
    V = 30;     % 分布容积 (L)
    Ka = 1.2;   % 吸收速率常数 (1/h)
    Ke = 0.3;   % 消除速率常数 (1/h)
    C0 = 0;     % 初始浓度
    tSpan = [0, 24];  % 时间范围 (h)
    
    % 求解
    [t, C] = ode45(@(t,C) drugModel(t, C, F, D, V, Ka, Ke), tSpan, C0);
    
    % 绘制结果
    figure;
    plot(t, C, 'b', 'LineWidth', 1.5);
    title('药物在血液中的浓度变化');
    xlabel('时间 (h)');
    ylabel('浓度 (mg/L)');
    grid on;
    
    % 计算峰浓度和达峰时间
    [Cmax, idx] = max(C);
    Tmax = t(idx);
    fprintf('\n药物动力学模型结果:\n');
    fprintf('峰浓度: %.2f mg/L\n', Cmax);
    fprintf('达峰时间: %.2f h\n', Tmax);
end

% 运行药物动力学案例
% pharmacokinetics_demo(); % 取消注释运行此案例