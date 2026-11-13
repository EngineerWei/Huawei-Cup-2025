% 2. 粒子群算法（Particle Swarm Optimization）实现
% 功能：求解连续函数优化问题
% 示例：寻找Sphere函数的最小值

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 参数设置
numVars = 3;               % 变量数量
numParticles = 30;         % 粒子数量
maxIter = 100;             % 最大迭代次数
c1 = 1.5;                  % 认知系数
c2 = 1.5;                  % 社会系数
w = 0.8;                   % 惯性权重

% 2. 变量范围
lb = -10*ones(1, numVars); % 下界
ub = 10*ones(1, numVars);  % 上界

% 3. 初始化粒子
particles = lb + (ub - lb).*rand(numParticles, numVars);  % 粒子位置
velocities = zeros(numParticles, numVars);                % 粒子速度

% 4. 计算初始适应度
fitness = sphere_function(particles');

% 5. 初始化个体最优和全局最优
pbest = particles;                  % 个体最优位置
pbestFitness = fitness;             % 个体最优适应度
[gbestFitness, idx] = min(fitness); % 全局最优适应度
gbest = particles(idx, :);          % 全局最优位置

% 6. 记录收敛过程
convergence = zeros(maxIter, 1);

% 7. 粒子群优化主循环
for iter = 1:maxIter
    % 更新每个粒子的速度和位置
    for i = 1:numParticles
        % 计算速度
        r1 = rand(1, numVars);
        r2 = rand(1, numVars);
        velocities(i, :) = w*velocities(i, :) + ...
            c1*r1.*(pbest(i, :) - particles(i, :)) + ...
            c2*r2.*(gbest - particles(i, :));
        
        % 更新位置
        particles(i, :) = particles(i, :) + velocities(i, :);
        
        % 边界处理
        particles(i, :) = max(particles(i, :), lb);
        particles(i, :) = min(particles(i, :), ub);
    end
    
    % 计算适应度
    fitness = sphere_function(particles');
    
    % 更新个体最优和全局最优
    for i = 1:numParticles
        if fitness(i) < pbestFitness(i)
            pbest(i, :) = particles(i, :);
            pbestFitness(i) = fitness(i);
        end
    end
    
    % 更新全局最优
    [currentBest, idx] = min(pbestFitness);
    if currentBest < gbestFitness
        gbestFitness = currentBest;
        gbest = pbest(idx, :);
    end
    
    % 记录收敛过程
    convergence(iter) = gbestFitness;
    
    % 显示迭代信息
    if mod(iter, 10) == 0 || iter == 1
        fprintf('迭代 %d: 最优值 = %.4f\n', iter, gbestFitness);
    end
end

% 8. 显示结果
fprintf('\n优化完成！\n');
fprintf('最优解: x = [');
fprintf('%.4f ', gbest);
fprintf(']\n');
fprintf('最优目标函数值: f = %.4f\n', gbestFitness);

% 9. 绘制收敛曲线
figure;
plot(convergence);
title('粒子群算法收敛曲线');
xlabel('迭代次数');
ylabel('最优适应度值');
grid on;

% 定义Sphere测试函数
function f = sphere_function(x)
    % x是列向量，每行一个变量
    f = sum(x.^2, 1);  % 对每个粒子计算适应度
end

% 实际应用案例：投资组合优化
function portfolio_optimization_demo()
    clear; clc;
    
    % 问题：4种资产的投资比例优化，最大化夏普比率
    numAssets = 4;      % 资产数量
    numParticles = 30;  % 粒子数量
    maxIter = 100;      % 迭代次数
    
    % 资产收益率和协方差矩阵（示例数据）
    returns = [0.08, 0.12, 0.15, 0.09];  % 预期收益率
    covMatrix = [0.01   0.003  0.002  0.001;
                 0.003  0.02   0.005  0.002;
                 0.002  0.005  0.03   0.003;
                 0.001  0.002  0.003  0.015];  % 协方差矩阵
    
    % 初始化粒子（权重）
    particles = rand(numParticles, numAssets);
    % 归一化，确保权重和为1
    particles = particles ./ sum(particles, 2);
    
    velocities = zeros(numParticles, numAssets);
    w = 0.7;   % 惯性权重
    c1 = 1.4;  % 认知系数
    c2 = 1.4;  % 社会系数
    
    % 计算初始适应度（夏普比率，越大越好）
    fitness = zeros(1, numParticles);
    for i = 1:numParticles
        weights = particles(i, :);
        portReturn = weights * returns';
        portRisk = sqrt(weights * covMatrix * weights');
        fitness(i) = portReturn / portRisk;  % 夏普比率
    end
    
    % 初始化个体最优和全局最优
    pbest = particles;
    pbestFitness = fitness;
    [gbestFitness, idx] = max(fitness);
    gbest = particles(idx, :);
    
    % 记录收敛过程
    convergence = zeros(maxIter, 1);
    
    % 主循环
    for iter = 1:maxIter
        for i = 1:numParticles
            % 更新速度
            r1 = rand(1, numAssets);
            r2 = rand(1, numAssets);
            velocities(i, :) = w*velocities(i, :) + ...
                c1*r1.*(pbest(i, :) - particles(i, :)) + ...
                c2*r2.*(gbest - particles(i, :));
            
            % 更新位置
            particles(i, :) = particles(i, :) + velocities(i, :);
            
            % 边界处理
            particles(i, :) = max(particles(i, :), 0);  % 权重不能为负
            
            % 重新归一化
            sumWeights = sum(particles(i, :));
            if sumWeights > 0
                particles(i, :) = particles(i, :) / sumWeights;
            else
                particles(i, :) = ones(1, numAssets) / numAssets;
            end
        end
        
        % 计算适应度
        for i = 1:numParticles
            weights = particles(i, :);
            portReturn = weights * returns';
            portRisk = sqrt(weights * covMatrix * weights');
            fitness(i) = portReturn / portRisk;
        end
        
        % 更新最优解
        for i = 1:numParticles
            if fitness(i) > pbestFitness(i)
                pbest(i, :) = particles(i, :);
                pbestFitness(i) = fitness(i);
            end
        end
        
        [currentBest, idx] = max(pbestFitness);
        if currentBest > gbestFitness
            gbestFitness = currentBest;
            gbest = pbest(idx, :);
        end
        
        convergence(iter) = gbestFitness;
        
        % 显示信息
        if mod(iter, 10) == 0 || iter == 1
            fprintf('迭代 %d: 最大夏普比率 = %.4f\n', iter, gbestFitness);
        end
    end
    
    % 显示结果
    fprintf('\n最优投资组合:\n');
    assets = {'股票A', '股票B', '股票C', '债券'};
    for i = 1:numAssets
        fprintf('%s: %.2f%% ', assets{i}, gbest(i)*100);
    end
    
    % 计算最优组合的收益和风险
    portReturn = gbest * returns';
    portRisk = sqrt(gbest * covMatrix * gbest');
    fprintf('\n预期收益率: %.2f%%, 风险(标准差): %.2f%%\n', ...
        portReturn*100, portRisk*100);
    
    % 绘制收敛曲线
    figure;
    plot(convergence);
    title('投资组合优化收敛曲线');
    xlabel('迭代次数');
    ylabel('夏普比率');
    grid on;
    
    % 绘制饼图
    figure;
    pie(gbest, assets);
    title('最优投资组合分配');
end

% 运行投资组合优化案例
% portfolio_optimization_demo(); % 取消注释运行此案例