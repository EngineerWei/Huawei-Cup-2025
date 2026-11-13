% 4. 随机森林（Random Forest）实现
% 功能：分类与回归问题
% 示例：基于随机森林的鸢尾花分类和波士顿房价预测

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 分类问题示例：鸢尾花数据集分类
disp('=== 分类问题示例：鸢尾花数据集 ===');

% 加载数据
load fisheriris;
X = meas;          % 特征
Y = species;       % 标签

% 划分训练集和测试集
cv = cvpartition(Y, 'HoldOut', 0.3);  % 30%作为测试集
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

% 训练随机森林分类器
numTrees = 50;  % 树的数量
model = TreeBagger(numTrees, X_train, Y_train, 'Method', 'classification');

% 预测
Y_pred = predict(model, X_test);
Y_pred = cell2mat(Y_pred);  % 转换为矩阵

% 评估结果
accuracy = sum(strcmp(Y_pred, Y_test)) / length(Y_test);
fprintf('分类准确率: %.2f%%\n', accuracy * 100);

% 混淆矩阵
figure;
confusionchart(Y_test, Y_pred);
title('随机森林分类混淆矩阵');

% 特征重要性
featureImportance = model.OOBPermutedPredictorImportance;
figure;
bar(featureImportance);
title('特征重要性');
xlabel('特征索引');
ylabel('重要性');
grid on;

% 2. 回归问题示例：预测房价
disp('\n=== 回归问题示例：房价预测 ===');

% 生成模拟数据（1000个样本，5个特征）
rng(42);  % 设置随机种子
n = 1000;
X = randn(n, 5);  % 特征
% 生成目标值（房价）
y = 250 + 30*X(:,1) - 15*X(:,2) + 20*abs(X(:,3)) + 10*X(:,4).*X(:,5) + 15*randn(n,1);

% 划分训练集和测试集
cv = cvpartition(n, 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% 训练随机森林回归器
numTrees = 50;
regModel = TreeBagger(numTrees, X_train, y_train, 'Method', 'regression');

% 预测
y_pred = predict(regModel, X_test);
y_pred = cell2mat(y_pred);

% 评估结果
rmse = sqrt(mean((y_pred - y_test).^2));
r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);
fprintf('均方根误差 (RMSE): %.2f\n', rmse);
fprintf('决定系数 (R²): %.4f\n', r2);

% 绘制预测值与实际值对比
figure;
scatter(y_test, y_pred, 'b', 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--');
title('房价预测：实际值 vs 预测值');
xlabel('实际房价');
ylabel('预测房价');
grid on;
legend('预测值', '理想线');

% 3. 实际应用案例：客户流失预测
function customer_churn_prediction_demo()
    clear; clc;
    
    % 生成模拟的客户数据
    rng(42);
    n = 1000;  % 客户数量
    
    % 特征：消费金额、使用时长、投诉次数、月费、年龄段
    X = [randn(n,1)*500 + 2000;  % 消费金额
         rand(n,1)*36 + 6;       % 使用时长（月）
         randi([0,5],n,1);       % 投诉次数
         rand(n,1)*50 + 30;      % 月费
         randi([1,5],n,1)]';     % 年龄段（1-5）
    
    % 计算流失概率（基于特征）
    churnProb = 1 ./ (1 + exp(-(-0.001*X(1,:) + 0.05*X(2,:) - 0.3*X(3,:) + 0.02*X(4,:) - 0.1*X(5,:) + 1)));
    
    % 生成流失标签
    Y = double(rand(n,1) < churnProb');
    
    % 划分训练集和测试集
    cv = cvpartition(Y, 'HoldOut', 0.3);
    X_train = X(:, training(cv))';
    Y_train = Y(training(cv), :);
    X_test = X(:, test(cv))';
    Y_test = Y(test(cv), :);
    
    % 训练随机森林
    model = TreeBagger(50, X_train, Y_train, 'Method', 'classification');
    
    % 预测
    Y_pred = predict(model, X_test);
    Y_pred = cell2mat(Y_pred);
    Y_pred = str2double(Y_pred);
    
    % 评估
    accuracy = mean(Y_pred == Y_test);
    fprintf('客户流失预测准确率: %.2f%%\n', accuracy * 100);
    
    % 混淆矩阵
    figure;
    confusionchart(Y_test, Y_pred);
    title('客户流失预测混淆矩阵');
    
    % 特征重要性
    figure;
    bar(model.OOBPermutedPredictorImportance);
    title('客户流失预测特征重要性');
    xticklabels({'消费金额', '使用时长', '投诉次数', '月费', '年龄段'});
    grid on;
end

% 运行客户流失预测案例
% customer_churn_prediction_demo(); % 取消注释运行此案例