# 8. 主成分分析（PCA）
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_classification

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def pca_demo(data, n_components=2, feature_names=None, target=None, target_names=None):
    """
    主成分分析演示
    
    参数:
        data: 输入数据，形状为(n_samples, n_features)
        n_components: 要保留的主成分数量
        feature_names: 特征名称列表
        target: 目标变量（用于可视化）
        target_names: 目标变量名称
    
    返回:
        pca: 拟合的PCA模型
        transformed_data: 降维后的数据
    """
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 应用PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(scaled_data)
    
    # 解释方差比例
    explained_variance = pca.explained_variance_ratio_
    print(f"主成分解释方差比例: {explained_variance.round(4)}")
    print(f"累计解释方差比例: {np.sum(explained_variance).round(4)}")
    
    # 可视化主成分
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        
        if target is not None:
            # 带有类别标签的散点图
            unique_targets = np.unique(target)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))
            
            for t, c in zip(unique_targets, colors):
                mask = (target == t)
                label = target_names[t] if target_names else f'类别{t}'
                plt.scatter(transformed_data[mask, 0], transformed_data[mask, 1], 
                           c=[c], label=label, alpha=0.7)
            plt.legend()
        else:
            # 无类别标签的散点图
            plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.7)
        
        plt.title(f'PCA降维结果 (保留{n_components}个主成分)')
        plt.xlabel(f'主成分1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'主成分2 ({explained_variance[1]:.2%})')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 显示主成分载荷（特征重要性）
    if feature_names and n_components >= 1:
        print("\n主成分载荷（特征重要性）:")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'主成分{i+1}' for i in range(n_components)],
            index=feature_names
        )
        print(loadings.round(4))
        
        # 可视化主成分载荷
        plt.figure(figsize=(12, 6))
        loadings.plot(kind='bar')
        plt.title('各特征在主成分上的载荷')
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
    
    # 可视化累计解释方差
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比例')
    plt.title('PCA累计解释方差曲线')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca, transformed_data

# 实际应用：客户分群
def customer_segmentation_demo():
    """客户分群PCA分析示例"""
    print("=== 客户分群PCA分析 ===")
    
    # 生成模拟客户数据
    np.random.seed(42)
    n_customers = 200
    
    # 客户特征：消费频率、平均消费金额、最近消费时间(天)、购买品类数
    freq = np.random.normal(20, 8, n_customers)
    amount = np.random.normal(500, 200, n_customers)
    recency = np.random.normal(30, 15, n_customers)
    categories = np.random.randint(1, 10, n_customers)
    
    # 标准化特征（最近消费时间需要反向，因为值越小越好）
    freq = (freq - np.mean(freq)) / np.std(freq)
    amount = (amount - np.mean(amount)) / np.std(amount)
    recency = -(recency - np.mean(recency)) / np.std(recency)  # 反向标准化
    categories = (categories - np.mean(categories)) / np.std(categories)
    
    # 构建特征矩阵
    data = np.column_stack([freq, amount, recency, categories])
    feature_names = ['消费频率', '平均消费金额', '最近消费时间(标准化)', '购买品类数']
    
    # 应用PCA
    pca, transformed_data = pca_demo(
        data, n_components=2, feature_names=feature_names
    )
    
    # 基于PCA结果进行简单聚类可视化
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(transformed_data)
    
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        mask = (clusters == i)
        plt.scatter(transformed_data[mask, 0], transformed_data[mask, 1], 
                   c=colors[i], label=f'客户群{i+1}', alpha=0.7)
    
    plt.title('基于PCA的客户分群结果')
    plt.xlabel(f'主成分1')
    plt.ylabel(f'主成分2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca, transformed_data

# 演示
if __name__ == "__main__":
    # 1. 使用鸢尾花数据集演示
    print("=== 鸢尾花数据集PCA分析 ===")
    iris = load_iris()
    pca_iris, transformed_iris = pca_demo(
        iris.data, 
        n_components=2,
        feature_names=iris.feature_names,
        target=iris.target,
        target_names=iris.target_names
    )
    
    # 2. 客户分群应用
    customer_segmentation_demo()
