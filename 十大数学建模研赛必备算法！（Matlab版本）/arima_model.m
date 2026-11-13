% 7. ARIMA模型实现（时间序列预测）
% 功能：使用ARIMA模型进行时间序列预测
% 示例：预测月度销售额数据

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 生成示例数据（月度销售额）
rng(42);  % 设置随机种子
months = 120;  % 10年数据
t = 1:months;

% 生成带趋势和季节性的时间序列
trend = 0.5 * t;
seasonality = 10 * sin(2*pi*t/12);  % 年度季节性
noise = 2 * randn(1, months);       % 随机噪声
sales = trend + seasonality + noise;
sales = max(sales, 5);  % 确保销售额为正数

% 转换为时间序列对象
sales_ts = timeseries(sales, t);

% 绘制原始数据
figure;
plot(sales_ts);
title('月度销售额数据');
xlabel('月份');
ylabel('销售额');
grid on;

% 2. 检查序列平稳性（ADF检验）
[h, pValue, stat, cValue] = adftest(sales);
fprintf('ADF检验结果:\n');
fprintf('是否平稳（原假设：非平稳）: %s\n', h ? '是' : '否');
fprintf('p值: %.4f\n', pValue);

% 如果非平稳，进行差分
d = 0;
sales_stationary = sales;
while ~h && d < 3  % 最多差分3次
    d = d + 1;
    sales_stationary = diff(sales_stationary);
    [h, pValue] = adftest(sales_stationary);
end
fprintf('使序列平稳所需的差分次数: %d\n', d);

% 3. 确定ARIMA模型阶数
figure;
autocorr(sales_stationary);
title('自相关函数 (ACF)');

figure;
parcorr(sales_stationary);
title('偏自相关函数 (PACF)');

% 根据ACF和PACF确定p和q值（这里手动设置为p=1, q=1）
p = 1;
q = 1;
modelOrder = [p d q];

% 4. 拟合ARIMA模型
model = arima(modelOrder);
[fitModel, ~, logL] = estimate(model, sales);
fprintf('\nARIMA模型参数估计结果:\n');
disp(fitModel);

% 5. 模型诊断（检查残差是否为白噪声）
figure;
plotDiagnostics(fitModel, 'acf');

% 6. 预测
numPeriods = 12;  % 预测未来12个月
[forecast, ~, forecastCI] = forecast(fitModel, numPeriods, 'Y0', sales);

% 7. 可视化预测结果
figure;
plot(sales, 'b', 'LineWidth', 1.5);
hold on;
plot(length(sales)+1:length(sales)+numPeriods, forecast, 'r', 'LineWidth', 1.5);
plot(length(sales)+1:length(sales)+numPeriods, forecastCI, 'r--', 'LineWidth', 1);
title('ARIMA模型预测结果');
xlabel('月份');
ylabel('销售额');
legend('历史数据', '预测值', '95%置信区间');
grid on;

% 8. 评估模型（使用最后12个月作为测试集）
testSize = 12;
trainData = sales(1:end-testSize);
testData = sales(end-testSize+1:end);

% 重新拟合模型
modelTrain = arima(modelOrder);
fitModelTrain = estimate(modelTrain, trainData);

% 预测测试集
[forecastTest, ~] = forecast(fitModelTrain, testSize);

% 计算误差
rmse = sqrt(mean((forecastTest - testData).^2));
mape = mean(abs(forecastTest - testData)./abs(testData)) * 100;

fprintf('\n模型评估结果:\n');
fprintf('测试集RMSE: %.2f\n', rmse);
fprintf('测试集MAPE: %.2f%%\n', mape);

% 绘制测试集预测结果
figure;
plot(1:testSize, testData, 'b', 'LineWidth', 1.5);
hold on;
plot(1:testSize, forecastTest, 'r', 'LineWidth', 1.5);
title('测试集预测结果对比');
xlabel('测试集月份');
ylabel('销售额');
legend('实际值', '预测值');
grid on;

% 9. 实际应用案例：电力负荷预测
function electricity_load_forecasting_demo()
    % 此函数实现电力负荷ARIMA预测
    % 取消下面一行的注释即可运行
    % disp('电力负荷ARIMA预测案例：请参考上面的销售额预测代码结构进行修改');
end