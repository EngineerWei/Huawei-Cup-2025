% 1. 遗传算法（Genetic Algorithm）实现
% 功能：求解函数优化问题
% 示例：寻找Rastrigin函数的最小值

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 定义问题参数
numVars = 5;               % 变量数量
lb = -5.12*ones(1, numVars); % 变量下界
ub = 5.12*ones(1, numVars);  % 变量上界

% 2. 定义适应度函数（Rastrigin函数）
fitnessFcn = @(x) rastrigin(x);

% 3. 遗传算法参数设置
options = gaoptimset(...
    'PopulationSize', 50, ...      % 种群大小
    'MaxGenerations', 100, ...     % 最大迭代次数
    'EliteCount', 2, ...           % 精英个体数量
    'CrossoverFraction', 0.8, ...  % 交叉概率
    'MutationFcn', @mutationgaussian, ... % 变异函数
    'Display', 'iter', ...         % 显示迭代过程
    'PlotFcns', @gaplotbestf);     % 绘制最优适应度曲线

% 4. 运行遗传算法
[xBest, fBest] = ga(fitnessFcn, numVars, [], [], [], [], lb, ub, [], options);

% 5. 显示结果
fprintf('\n最优解: x = [');
fprintf('%.4f ', xBest);
fprintf(']\n');
fprintf('最优目标函数值: f = %.4f\n', fBest);

% 定义Rastrigin函数（测试函数）
function f = rastrigin(x)
    A = 10;
    n = length(x);
    f = A*n + sum(x.^2 - A*cos(2*pi*x));
end

% 实际应用案例：资源分配问题
function resource_allocation_demo()
    clear; clc;
    
    % 问题：5个项目分配100万元预算，最大化总收益
    % 变量：x(1)-x(5)表示每个项目的预算
    
    % 1. 定义适应度函数（总收益）
    fitnessFcn = @(x) -resource_fitness(x);  % 加负号因为GA默认求最小值
    
    % 2. 约束条件：预算总和为100
    A = ones(1,5);
    b = 100;
    
    % 3. 变量范围
    lb = zeros(1,5);
    ub = 100*ones(1,5);
    
    % 4. 参数设置
    options = gaoptimset(...
        'PopulationSize', 30, ...
        'MaxGenerations', 50, ...
        'Display', 'iter', ...
        'PlotFcns', @gaplotbestf);
    
    % 5. 运行遗传算法
    [xBest, fBest] = ga(fitnessFcn, 5, A, b, [], [], lb, ub, [], options);
    
    % 6. 显示结果
    fprintf('\n资源分配最优解:\n');
    for i = 1:5
        fprintf('项目%d: %.2f万元 ', i, xBest(i));
    end
    fprintf('\n最大总收益: %.2f万元\n', -fBest);
    
    % 绘制分配饼图
    figure;
    pie(xBest, {'项目1','项目2','项目3','项目4','项目5'});
    title('最优资源分配方案');
end

% 资源分配的收益函数
function profit = resource_fitness(x)
    % 每个项目的收益函数（示例）
    profit = 10*sqrt(x(1)) + 8*sqrt(x(2)) + 12*sqrt(x(3)) + 9*sqrt(x(4)) + 15*sqrt(x(5));
end

% 运行资源分配案例
% resource_allocation_demo(); % 取消注释运行此案例