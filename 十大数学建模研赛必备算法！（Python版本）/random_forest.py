# 4. 随机森林（Random Forest）
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def random_forest_classification_demo():
    """随机森林分类演示"""
    # 生成示例分类数据
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=3, n_classes=3, random_state=42
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建并训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"分类准确率: {accuracy:.4f}")
    
    # 特征重要性
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), [f'特征{i+1}' for i in range(len(importances))])
    plt.title('特征重要性')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

def random_forest_regression_demo():
    """随机森林回归演示"""
    # 生成示例回归数据
    X, y = make_regression(
        n_samples=500, n_features=8, n_informative=5,
        noise=10, random_state=42
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建并训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差 (MSE): {mse:.4f}")
    
    # 预测值与实际值对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想线
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

# 实际应用：客户流失预测
def customer_churn_prediction():
    """客户流失预测演示"""
    print("=== 客户流失预测 ===")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 特征：使用时长(月)、月消费、投诉次数、服务类型(0-2)
    tenure = np.random.randint(1, 61, n_samples)
    monthly_charge = np.random.normal(50, 20, n_samples)
    complaints = np.random.randint(0, 5, n_samples)
    service_type = np.random.randint(0, 3, n_samples)
    
    # 构建特征矩阵
    X = np.column_stack([tenure, monthly_charge, complaints, service_type])
    
    # 生成目标变量：流失概率与使用时长负相关，与投诉次数正相关
    churn_prob = 0.5 - 0.01*tenure + 0.1*complaints + 0.001*monthly_charge
    churn_prob = np.clip(churn_prob, 0.05, 0.95)  # 限制在合理范围内
    y = np.random.binomial(1, churn_prob)  # 二分类结果
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"预测准确率: {accuracy:.4f}")
    
    # 特征重要性
    features = ['使用时长', '月消费', '投诉次数', '服务类型']
    importances = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.bar(features, importances)
    plt.title('客户流失预测特征重要性')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

# 演示
if __name__ == "__main__":
    print("=== 随机森林分类示例 ===")
    clf = random_forest_classification_demo()
    
    print("\n=== 随机森林回归示例 ===")
    reg = random_forest_regression_demo()
    
    # 客户流失预测应用
    customer_churn_prediction()
