% 5. 整数规划实现
% 功能：求解整数规划问题
% 示例：生产计划问题和背包问题

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 生产计划问题示例
disp('=== 生产计划问题 ===');

% 问题：生产A、B两种产品，最大化利润
% 约束：
% 1. 原材料限制：3A + 2B <= 100
% 2. 人工限制：2A + 4B <= 120
% 3. 产品A最多生产20个

% 目标函数系数（利润）
f = [-10; -15];  % 加负号因为linprog求最小值

% 不等式约束 A*x <= b
A = [3 2;        % 原材料约束
     2 4];       % 人工约束
b = [100; 120];

% 变量下界和上界
lb = [0; 0];     % 非负约束
ub = [20; Inf];  % 产品A最多20个

% 变量类型（整数）
intcon = [1, 2];  % 两个变量都是整数

% 求解整数规划
options = optimoptions('intlinprog', 'Display', 'off');
[x, fval, exitflag] = intlinprog(f, intcon, A, b, [], [], lb, ub, options);

% 显示结果
if exitflag == 1
    fprintf('最优生产计划:\n');
    fprintf('产品A: %d 个\n', round(x(1)));
    fprintf('产品B: %d 个\n', round(x(2)));
    fprintf('最大利润: %.2f 元\n', -fval);
else
    fprintf('未找到可行解\n');
end

% 2. 背包问题示例
disp('\n=== 背包问题 ===');

% 问题：有5个物品，重量分别为[3, 4, 2, 6, 5]，价值分别为[4, 5, 3, 7, 6]
% 背包容量为10，选择物品使总价值最大，每个物品只能选一次

% 目标函数（价值）
values = [4; 5; 3; 7; 6];
f = -values;  % 加负号求最大值

% 约束条件（重量）
weights = [3; 4; 2; 6; 5];
A = weights';
b = 10;

% 变量类型（0-1整数）
intcon = 1:5;  % 所有变量都是整数
lb = zeros(5, 1);  % 下界
ub = ones(5, 1);   % 上界（0或1）

% 求解
[x, fval, exitflag] = intlinprog(f, intcon, A, b, [], [], lb, ub, options);

% 显示结果
if exitflag == 1
    fprintf('最优选择:\n');
    for i = 1:5
        fprintf('物品%d: %s ', i, x(i) == 1 ? '选中' : '未选中');
    end
    fprintf('\n总重量: %.0f\n', A*x);
    fprintf('总价值: %.0f\n', -fval);
end

% 3. 实际应用案例：选址问题
function facility_location_demo()
    clear; clc;
    
    % 问题：在5个候选地点中选择不超过2个建设设施，
    % 为4个需求点提供服务，使总成本最小
    
    % 固定建设成本
    fixedCosts = [100, 120, 90, 150, 80];
    
    % 运输成本（从设施i到需求点j）
    transportCosts = [
        [5, 8, 6, 7];   % 设施1到各需求点
        [7, 4, 9, 5];   % 设施2到各需求点
        [6, 7, 5, 8];   % 设施3到各需求点
        [4, 6, 7, 9];   % 设施4到各需求点
        [8, 5, 4, 6]    % 设施5到各需求点
    ];
    
    % 需求点需求
    demands = [10, 15, 8, 12];
    
    % 构建目标函数
    % 变量: x(1-5)表示是否建设设施(0-1), x(6-25)表示运输量
    nFacilities = 5;
    nDemands = 4;
    f = zeros(nFacilities + nFacilities*nDemands, 1);
    
    % 固定成本
    f(1:nFacilities) = fixedCosts';
    
    % 运输成本
    for i = 1:nFacilities
        for j = 1:nDemands
            idx = nFacilities + (i-1)*nDemands + j;
            f(idx) = transportCosts(i,j);
        end
    end
    
    % 约束条件
    % Aeq: 等式约束，需求点必须被满足
    Aeq = zeros(nDemands, length(f));
    beq = demands';
    
    for j = 1:nDemands
        for i = 1:nFacilities
            idx = nFacilities + (i-1)*nDemands + j;
            Aeq(j, idx) = 1;
        end
    end
    
    % A: 不等式约束，只有建设了设施才能有运输量
    A = zeros(nFacilities*nDemands, length(f));
    b = zeros(nFacilities*nDemands, 1);
    
    for i = 1:nFacilities
        for j = 1:nDemands
            row = (i-1)*nDemands + j;
            % 运输量 <= M*x_i (M是一个大常数)
            idx = nFacilities + (i-1)*nDemands + j;
            A(row, idx) = 1;
            A(row, i) = -1000;  % M=1000
        end
    end
    
    % 最多建设2个设施
    A = [A; zeros(1, length(f))];
    A(end, 1:nFacilities) = 1;
    b = [b; 2];
    
    % 变量上下界
    lb = zeros(length(f), 1);
    ub = ones(length(f), 1);
    for i = nFacilities+1:length(f)
        ub(i) = 1000;  % 运输量上限
    end
    
    % 整数变量
    intcon = 1:nFacilities;  % 只有设施选择是0-1变量
    
    % 求解
    options = optimoptions('intlinprog', 'Display', 'off');
    [x, fval, exitflag] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, options);
    
    % 显示结果
    if exitflag == 1
        fprintf('\n选址问题最优解:\n');
        fprintf('建设的设施: ');
        for i = 1:nFacilities
            if x(i) > 0.5
                fprintf('%d ', i);
            end
        end
        fprintf('\n总成本: %.2f\n', fval);
    end
end

% 运行选址问题案例
% facility_location_demo(); % 取消注释运行此案例