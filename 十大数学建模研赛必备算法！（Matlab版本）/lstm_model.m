% 3. LSTM模型实现（时间序列预测）
% 功能：使用LSTM进行时间序列预测
% 示例：预测模拟的月度销售额数据

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 生成示例数据（月度销售额）
np.random.seed(42);
months = 120;  % 10年数据
t = 1:months;

% 生成带趋势和季节性的时间序列
trend = 0.5 * t;
seasonality = 10 * sin(2*pi*t/12);  % 年度季节性
noise = 2 * randn(1, months);       % 随机噪声
sales = trend + seasonality + noise;
sales = max(sales, 5);  % 确保销售额为正数

% 绘制原始数据
figure;
plot(t, sales, 'b');
title('月度销售额数据');
xlabel('月份');
ylabel('销售额');
grid on;

% 2. 数据预处理
% 归一化到[0,1]
[scaledData, minVal, maxVal] = normalizeData(sales);

% 创建序列数据
seqLength = 12;  % 使用过去12个月预测下一个月
[X, y] = createSequences(scaledData, seqLength);

% 划分训练集和测试集（80%训练，20%测试）
trainRatio = 0.8;
trainSize = floor(trainRatio * length(y));
X_train = X(1:trainSize, :, :);
y_train = y(1:trainSize, :);
X_test = X(trainSize+1:end, :, :);
y_test = y(trainSize+1:end, :);

% 3. 创建LSTM模型
model = createLSTMModel(seqLength);

% 4. 训练模型
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'ValidationSplit', 0.2, ...
    'EarlyStoppingPatience', 5, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

[trainedModel, trainingInfo] = trainNetwork(X_train, y_train, model, options);

% 5. 模型预测
y_pred_train = predict(trainedModel, X_train);
y_pred_test = predict(trainedModel, X_test);

% 反归一化
y_pred_train = denormalizeData(y_pred_train, minVal, maxVal);
y_pred_test = denormalizeData(y_pred_test, minVal, maxVal);
y_train_actual = denormalizeData(y_train, minVal, maxVal);
y_test_actual = denormalizeData(y_test, minVal, maxVal);

% 6. 评估模型
trainRMSE = sqrt(mean((y_pred_train - y_train_actual).^2));
testRMSE = sqrt(mean((y_pred_test - y_test_actual).^2));
fprintf('\n训练集RMSE: %.2f\n', trainRMSE);
fprintf('测试集RMSE: %.2f\n', testRMSE);

% 7. 绘制预测结果
figure;
% 绘制所有数据
plot(t, sales, 'b', 'LineWidth', 1, 'DisplayName', '原始数据');
% 绘制训练集预测
trainIndices = seqLength+1:seqLength+trainSize;
plot(trainIndices, y_pred_train, 'g', 'LineWidth', 1.5, 'DisplayName', '训练集预测');
% 绘制测试集预测
testIndices = seqLength+trainSize+1:months;
plot(testIndices, y_pred_test, 'r', 'LineWidth', 1.5, 'DisplayName', '测试集预测');
title('LSTM时间序列预测结果');
xlabel('月份');
ylabel('销售额');
legend();
grid on;

% 数据归一化函数
function [normalizedData, minVal, maxVal] = normalizeData(data)
    minVal = min(data);
    maxVal = max(data);
    normalizedData = (data - minVal) / (maxVal - minVal);
end

% 数据反归一化函数
function denormalizedData = denormalizeData(normalizedData, minVal, maxVal)
    denormalizedData = normalizedData * (maxVal - minVal) + minVal;
end

% 创建序列数据函数
function [X, y] = createSequences(data, seqLength)
    n = length(data);
    X = zeros(n - seqLength, seqLength, 1);
    y = zeros(n - seqLength, 1);
    
    for i = 1:n - seqLength
        X(i, :, 1) = data(i:i+seqLength-1);
        y(i) = data(i+seqLength);
    end
end

% 创建LSTM模型函数
function model = createLSTMModel(seqLength)
    model = sequential();
    model.addLayers(lstmLayer(64, 'ReturnSequences', false, ...
        'InputShape', [seqLength, 1]));
    model.addLayers(dropoutLayer(0.2));
    model.addLayers(fullyConnectedLayer(1));
    model.addLayers(regressionLayer());
end

% 实际应用案例：电力负荷预测
function electricity_load_forecasting_demo()
    % 此函数实现电力负荷预测，与上面的销售额预测类似
    % 取消下面一行的注释即可运行
    % disp('电力负荷预测案例：请参考上面的销售额预测代码结构进行修改');
end