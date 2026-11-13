% 10. 多目标优化模型实现
% 功能：求解多目标优化问题
% 示例：测试函数和生产计划多目标优化

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 多目标测试函数（ZDT1）
disp('=== 多目标测试函数 ZDT1 ===');

% 定义ZDT1测试函数
function [f1, f2] = zdt1(x)
    f1 = x(1);
    g = 1 + 9/(length(x)-1)*sum(x(2:end));
    h = 1 - sqrt(f1/g);
    f2 = g * h;
end

% 参数设置
nVar = 30;              % 变量数量
lb = zeros(1, nVar);    % 下界
ub = ones(1, nVar);     % 上界

% 使用NSGA-II算法求解
options = optimoptions('gamultiobj', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 100, ...
    'CrossoverFraction', 0.8, ...
    'Display', 'iter', ...
    'PlotFcn', @gaplotpareto);

% 运行多目标优化
[x, fval] = gamultiobj(@(x) deal(zdt1(x)), nVar, [], [], [], [], lb, ub, options);

% 绘制Pareto前沿
figure;
plot(fval(:,1), fval(:,2), 'ro');
title('ZDT1函数的Pareto前沿');
xlabel('f_1(x)');
ylabel('f_2(x)');
grid on;

% 2. 生产计划多目标优化
disp('\n=== 生产计划多目标优化 ===');

% 问题：生产两种产品，同时最小化成本和最大化利润
% 约束：资源限制和生产能力

% 定义多目标函数
function [f1, f2] = productionObjectives(x)
    % x(1): 产品A产量, x(2): 产品B产量
    
    % 目标1：最小化生产成本
    costA = 3;   % 单位成本
    costB = 2;   % 单位成本
    f1 = costA*x(1) + costB*x(2);
    
    % 目标2：最大化利润（作为负的最小化目标）
    priceA = 8;  % 单位售价
    priceB = 5;  % 单位售价
    profit = (priceA - costA)*x(1) + (priceB - costB)*x(2);
    f2 = -profit;  % 加负号因为gamultiobj求最小值
end

% 定义非线性约束
function [c, ceq] = productionConstraints(x)
    % 不等式约束 c <= 0
    % 资源1限制：2x1 + x2 <= 100
    % 资源2限制：x1 + 3x2 <= 120
    c(1) = 2*x(1) + x(2) - 100;
    c(2) = x(1) + 3*x(2) - 120;
    
    % 等式约束 ceq = 0（无）
    ceq = [];
end

% 参数设置
nVar = 2;                % 变量数量（两种产品）
lb = zeros(1, nVar);     % 下界（产量非负）
ub = [100, 100];         % 上界

% 设置优化选项
options = optimoptions('gamultiobj', ...
    'PopulationSize', 50, ...
    'MaxGenerations', 50, ...
    'CrossoverFraction', 0.8, ...
    'Display', 'iter', ...
    'PlotFcn', @gaplotpareto);

% 运行多目标优化
[x, fval] = gamultiobj(@(x) deal(productionObjectives(x)), ...
    nVar, [], [], [], [], lb, ub, @productionConstraints, options);

% 处理结果（将利润转换回正值）
fval(:,2) = -fval(:,2);

% 绘制Pareto前沿
figure;
plot(fval(:,1), fval(:,2), 'bo');
title('生产计划的Pareto前沿');
xlabel('生产成本');
ylabel('利润');
grid on;

% 显示几个代表性解
disp('代表性解决方案:');
indices = [1, round(size(fval,1)/2), size(fval,1)];
for i = 1:length(indices)
    idx = indices(i);
    fprintf('产品A: %.0f, 产品B: %.0f → 成本: %.0f, 利润: %.0f\n', ...
        x(idx,1), x(idx,2), fval(idx,1), fval(idx,2));
end

% 绘制产量散点图
figure;
scatter(x(:,1), x(:,2), 50, fval(:,2), 'filled');
colorbar;
title('Pareto最优生产计划');
xlabel('产品A产量');
ylabel('产品B产量');
grid on;

% 3. 实际应用案例：供应链多目标优化
function supply_chain_demo()
    clear; clc;
    
    % 问题：3个工厂向4个仓库运输产品
    % 目标：最小化运输成本和最小化运输时间
    
    % 定义多目标函数
    function [f1, f2] = supplyChainObjectives(x)
        % x是一个向量，包含从每个工厂到每个仓库的运输量
        % 转换为3x4矩阵
        shipments = reshape(x, 3, 4);
        
        % 运输成本矩阵
        costMatrix = [
            5, 8, 6, 7;
            7, 4, 9, 5;
            6, 7, 5, 8
        ];
        
        % 运输时间矩阵
        timeMatrix = [
            2, 4, 3, 5;
            3, 2, 5, 3;
            4, 3, 2, 4
        ];
        
        % 目标1：最小化总成本
        f1 = sum(sum(shipments .* costMatrix));
        
        % 目标2：最小化总时间（加权）
        f2 = sum(sum(shipments .* timeMatrix));
    end
    
    % 定义约束条件
    function [c, ceq] = supplyChainConstraints(x)
        shipments = reshape(x, 3, 4);
        
        % 工厂产能限制（不等式约束 c <= 0）
        capacities = [200, 300, 250];
        c(1:3) = sum(shipments, 2) - capacities;
        
        % 仓库需求（等式约束 ceq = 0）
        demands = [150, 200, 180, 120];
        ceq(1:4) = sum(shipments, 1) - demands;
    end
    
    % 参数设置
    nVar = 3*4;             % 3个工厂×4个仓库
    lb = zeros(1, nVar);    % 运输量非负
    ub = 200*ones(1, nVar); % 最大运输量
    
    % 设置优化选项
    options = optimoptions('gamultiobj', ...
        'PopulationSize', 80, ...
        'MaxGenerations', 80, ...
        'CrossoverFraction', 0.8, ...
        'Display', 'iter', ...
        'PlotFcn', @gaplotpareto);
    
    % 运行优化
    [x, fval] = gamultiobj(@(x) deal(supplyChainObjectives(x)), ...
        nVar, [], [], [], [], lb, ub, @supplyChainConstraints, options);
    
    % 绘制Pareto前沿
    figure;
    plot(fval(:,1), fval(:,2), 'go');
    title('供应链优化的Pareto前沿');
    xlabel('总成本');
    ylabel('总时间');
    grid on;
    
    fprintf('\n供应链优化结果示例:\n');
    fprintf('最优解范围内的最小成本: %.0f\n', min(fval(:,1)));
    fprintf('最优解范围内的最小时间: %.0f\n', min(fval(:,2)));
end

% 运行供应链优化案例
% supply_chain_demo(); % 取消注释运行此案例