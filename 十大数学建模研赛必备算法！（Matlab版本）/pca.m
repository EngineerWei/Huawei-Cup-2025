% 8. 主成分分析（PCA）实现
% 功能：数据降维和特征提取
% 示例：对鸢尾花数据集和随机高维数据进行降维

% 清除工作区和命令窗口
clear; clc; close all;

% 1. 鸢尾花数据集降维示例
disp('=== 鸢尾花数据集降维 ===');

% 加载数据
load fisheriris;
X = meas;          % 特征数据（4维）
labels = species;  % 标签

% 数据标准化（均值为0，方差为1）
[X_std, mu, sigma] = zscore(X);

% 执行PCA
[coeff, score, latent, ~, explained] = pca(X_std);

% 显示主成分贡献率
fprintf('主成分贡献率:\n');
for i = 1:length(explained)
    fprintf('主成分%d: %.2f%% ', i, explained(i));
end
fprintf('\n');

% 绘制累积贡献率
figure;
plot(cumsum(explained), 'b-o', 'LineWidth', 1.5);
title('主成分累积贡献率');
xlabel('主成分数量');
ylabel('累积贡献率 (%)');
grid on;
ylim([0, 100]);

% 使用前2个主成分进行可视化
figure;
gscatter(score(:,1), score(:,2), labels);
title('鸢尾花数据集PCA降维（前2主成分）');
xlabel(['主成分1 (', num2str(explained(1)), '%)']);
ylabel(['主成分2 (', num2str(explained(2)), '%)']);
grid on;

% 2. 高维随机数据降维示例
disp('\n=== 高维随机数据降维 ===');

% 生成10维数据（实际有效维度为3）
rng(42);
n = 200;  % 样本数量
X = randn(n, 3);  % 3维有效数据

% 添加7个冗余维度（由前3维线性组合而成）
X(:,4) = 0.5*X(:,1) + 0.3*X(:,2) + 0.2*X(:,3) + 0.1*randn(n,1);
X(:,5) = 0.2*X(:,1) + 0.6*X(:,2) + 0.2*X(:,3) + 0.1*randn(n,1);
X(:,6:10) = 0.5*randn(n,5) + 0.3*X(:,1:5);  % 更多冗余维度

% 标准化
X_std = zscore(X);

% 执行PCA
[coeff2, score2, latent2, ~, explained2] = pca(X_std);

% 显示主成分贡献率
fprintf('高维数据主成分贡献率:\n');
for i = 1:10
    fprintf('主成分%d: %.2f%% ', i, explained2(i));
    if mod(i,5) == 0
        fprintf('\n');
    end
end

% 绘制碎石图
figure;
plot(latent2, 'b-o', 'LineWidth', 1.5);
title('高维数据的碎石图');
xlabel('主成分');
ylabel('特征值');
grid on;

% 3. 实际应用案例：图像降维
function image_pca_demo()
    clear; clc;
    
    % 加载示例图像数据
    load('olivettifaces.mat');  % 包含400张64x64的人脸图像
    
    % 显示一些原始图像
    figure;
    for i = 1:4
        subplot(2,2,i);
        imshow(reshape(faces(:,i),64,64)', 'InitialMagnification', 200);
        title(['人脸 ', num2str(i)]);
    end
    
    % 数据标准化
    X_std = zscore(faces');  % 转置为样本x特征
    
    % 执行PCA
    [coeff, score, latent, ~, explained] = pca(X_std);
    
    % 显示累积贡献率
    figure;
    plot(cumsum(explained), 'b-', 'LineWidth', 1.5);
    title('人脸图像主成分累积贡献率');
    xlabel('主成分数量');
    ylabel('累积贡献率 (%)');
    grid on;
    ylim([0, 100]);
    
    % 确定保留的主成分数量（保留95%的信息）
    nComp = find(cumsum(explained) >= 95, 1);
    fprintf('\n保留95%%信息所需的主成分数量: %d\n', nComp);
    
    % 使用前nComp个主成分重构图像
    X_reduced = score(:,1:nComp) * coeff(:,1:nComp)';
    
    % 显示重构图像
    figure;
    for i = 1:4
        subplot(2,2,i);
        img_recon = reshape(X_reduced(i,:) + mean(faces,2)', 64,64)';
        imshow(img_recon, 'InitialMagnification', 200);
        title(['重构人脸 ', num2str(i)]);
    end
    
    % 显示主成分（特征脸）
    figure;
    for i = 1:4
        subplot(2,2,i);
        eigenface = reshape(coeff(:,i),64,64)';
        imshow(mat2gray(eigenface), 'InitialMagnification', 200);
        title(['特征脸 ', num2str(i)]);
    end
end

% 运行图像降维案例
% image_pca_demo(); % 取消注释运行此案例