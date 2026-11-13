% 6. 动态规划（Dynamic Programming）实现
% 功能：求解多阶段决策问题
% 示例：背包问题、最短路径问题

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 0-1背包问题（动态规划解法）
disp('=== 0-1背包问题 ===');

% 物品信息：重量和价值
weights = [3, 4, 2, 6, 5];
values = [4, 5, 3, 7, 6];
capacity = 10;  % 背包容量
n = length(weights);  % 物品数量

% 创建DP表
dp = zeros(n+1, capacity+1);

% 填充DP表
for i = 1:n
    for j = 1:capacity
        % 不选第i个物品
        dp(i+1, j+1) = dp(i, j+1);
        
        % 选第i个物品（如果能装下）
        if j >= weights(i) && dp(i, j-weights(i)+1) + values(i) > dp(i+1, j+1)
            dp(i+1, j+1) = dp(i, j-weights(i)+1) + values(i);
        end
    end
end

% 回溯找到最优选择
selected = false(1, n);
remaining = capacity;

for i = n:-1:1
    if dp(i+1, remaining+1) ~= dp(i, remaining+1)
        selected(i) = true;
        remaining = remaining - weights(i);
    end
end

% 显示结果
fprintf('背包最大价值: %d\n', dp(n+1, capacity+1));
fprintf('选中的物品: ');
for i = 1:n
    if selected(i)
        fprintf('%d ', i);
    end
end
fprintf('\n总重量: %d\n', sum(weights(selected)));

% 绘制DP表
figure;
imagesc(dp);
title('0-1背包问题的DP表');
xlabel('背包容量');
ylabel('物品数量');
colorbar;

% 2. 最短路径问题（动态规划解法）
disp('\n=== 最短路径问题 ===');

% 定义有向图的邻接矩阵
% graph(i,j)表示从节点i到节点j的距离，0表示无直接连接
graph = [
    0 2 4 0 0;
    0 0 1 5 0;
    0 0 0 3 6;
    0 0 0 0 2;
    0 0 0 0 0
];
nNodes = size(graph, 1);  % 节点数量
startNode = 1;            % 起始节点
endNode = nNodes;         % 目标节点

% 初始化距离数组和路径记录
distance = inf(1, nNodes);
distance(startNode) = 0;
prev = zeros(1, nNodes);  % 记录前驱节点

% 动态规划计算最短路径
for i = startNode:nNodes-1
    for j = i+1:nNodes
        if graph(i,j) > 0 && distance(j) > distance(i) + graph(i,j)
            distance(j) = distance(i) + graph(i,j);
            prev(j) = i;
        end
    end
end

% 回溯找到最短路径
path = [];
current = endNode;
while current ~= 0
    path = [current, path];
    current = prev(current);
end

% 显示结果
fprintf('从节点%d到节点%d的最短距离: %d\n', startNode, endNode, distance(endNode));
fprintf('最短路径: ');
for i = 1:length(path)
    fprintf('%d', path(i));
    if i < length(path)
        fprintf(' -> ');
    end
end
fprintf('\n');

% 3. 实际应用案例：生产计划问题
function production_scheduling_demo()
    clear; clc;
    
    % 问题：4个月的生产计划，已知各月需求
    % 目标：最小化生产成本（正常生产+加班+库存）
    
    % 参数设置
    months = 4;                  % 月份
    demand = [50, 80, 120, 70];  % 各月需求
    capRegular = 100;            % 每月正常生产能力
    capOvertime = 50;            % 每月加班生产能力
    costRegular = 5;             % 正常生产成本
    costOvertime = 8;            % 加班生产成本
    costInventory = 1;           % 每月库存成本
    
    % 创建DP表：dp(i,j)表示第i月底库存为j时的最小成本
    maxInventory = sum(demand);  % 最大可能库存
    dp = inf(months+1, maxInventory+1);
    dp(1, 1) = 0;  % 初始库存为0
    
    % 填充DP表
    for i = 1:months
        for inv_prev = 0:maxInventory
            % 如果前一个状态不可行，跳过
            if dp(i, inv_prev+1) == inf
                continue;
            end
            
            % 计算可能的生产量
            minProd = max(0, demand(i) - inv_prev);  % 至少需要生产的数量
            maxProd = capRegular + capOvertime;      % 最大可生产数量
            
            for prod = minProd:maxProd
                % 计算正常生产和加班生产的数量
                prodRegular = min(prod, capRegular);
                prodOvertime = prod - prodRegular;
                
                % 计算成本
                cost = prodRegular * costRegular + ...
                       prodOvertime * costOvertime;
                
                % 计算月末库存
                inv_current = inv_prev + prod - demand(i);
                
                % 确保库存非负且不超过最大值
                if inv_current < 0 || inv_current > maxInventory
                    continue;
                end
                
                % 更新DP表
                totalCost = dp(i, inv_prev+1) + cost + inv_current * costInventory;
                if totalCost < dp(i+1, inv_current+1)
                    dp(i+1, inv_current+1) = totalCost;
                end
            end
        end
    end
    
    % 找到最小成本（最后一个月库存应为0）
    minCost = dp(months+1, 1);  % 索引+1是因为MATLAB从1开始
    
    % 回溯找到最优生产计划
    production = zeros(1, months);
    inv = 0;  % 最后一个月库存为0
    
    for i = months:-1:1
        for inv_prev = 0:maxInventory
            if dp(i, inv_prev+1) == inf
                continue;
            end
            
            prod = demand(i) + inv - inv_prev;
            if prod < 0 || prod > capRegular + capOvertime
                continue;
            end
            
            prodRegular = min(prod, capRegular);
            prodOvertime = prod - prodRegular;
            cost = prodRegular * costRegular + prodOvertime * costOvertime;
            
            if dp(i, inv_prev+1) + cost + inv * costInventory == dp(i+1, inv+1)
                production(i) = prod;
                inv = inv_prev;
                break;
            end
        end
    end
    
    % 显示结果
    fprintf('\n生产计划问题最优解:\n');
    fprintf('总成本: %d\n', minCost);
    fprintf('各月生产量: ');
    for i = 1:months
        fprintf('%d ', production(i));
    end
    
    % 绘制生产计划图
    figure;
    bar([production; demand]', 'grouped');
    title('各月生产量与需求量对比');
    xlabel('月份');
    ylabel('数量');
    legend('生产量', '需求量');
    grid on;
end

% 运行生产计划案例
% production_scheduling_demo(); % 取消注释运行此案例