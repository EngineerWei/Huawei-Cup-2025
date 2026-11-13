# 7. ARIMA模型
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def arima_forecast(data, p=1, d=1, q=1, test_size=0.2):
    """
    ARIMA时间序列预测
    
    参数:
        data: 时间序列数据
        p: AR阶数
        d: 差分阶数
        q: MA阶数
        test_size: 测试集比例
    
    返回:
        model: 拟合的ARIMA模型
        predictions: 测试集预测结果
        y_test: 测试集实际值
    """
    # 划分训练集和测试集
    split = int(len(data) * (1 - test_size))
    train, test = data[:split], data[split:]
    
    # 拟合ARIMA模型
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    
    # 预测
    predictions = model_fit.forecast(steps=len(test))
    
    # 计算误差
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print(f"ARIMA({p},{d},{q}) 测试集RMSE: {rmse:.4f}")
    
    return model_fit, predictions, test, train

def plot_acf_pacf(data, lags=20):
    """绘制自相关和偏自相关图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    ax1.set_title('自相关图 (ACF)')
    ax2.set_title('偏自相关图 (PACF)')
    plt.tight_layout()
    plt.show()

# 演示
if __name__ == "__main__":
    # 生成示例数据（月度销售量）
    np.random.seed(42)
    n_periods = 120  # 10年数据
    time = np.arange(n_periods)
    
    # 生成带有趋势和季节性的数据
    trend = 0.3 * time
    seasonality = 5 * np.sin(2 * np.pi * time / 12)  # 年度季节性
    noise = np.random.normal(0, 1.5, n_periods)
    sales = trend + seasonality + noise + 10  # 基础销量10
    
    # 可视化原始数据
    plt.figure(figsize=(12, 6))
    plt.plot(time, sales)
    plt.title('月度销售量数据')
    plt.xlabel('月份')
    plt.ylabel('销售量')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 绘制ACF和PACF图，帮助确定p和q值
    plot_acf_pacf(sales)
    
    # 尝试不同的ARIMA参数组合
    print("=== ARIMA模型预测结果 ===")
    models = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]
    predictions_list = []
    
    for p, d, q in models:
        model_fit, predictions, test, train = arima_forecast(sales, p, d, q)
        predictions_list.append((p, d, q, predictions))
    
    # 选择表现最好的模型进行可视化
    best_idx = np.argmin([np.sqrt(mean_squared_error(test, pred)) 
                         for _, _, _, pred in predictions_list])
    best_p, best_d, best_q, best_pred = predictions_list[best_idx]
    
    # 可视化最佳模型的预测结果
    plt.figure(figsize=(12, 6))
    split = len(sales) - len(test)
    plt.plot(time, sales, label='原始数据', alpha=0.5)
    plt.plot(time[split:], best_pred, label=f'ARIMA({best_p},{best_d},{best_q}) 预测', color='red')
    plt.axvline(x=split, color='gray', linestyle='--', alpha=0.7)
    plt.title('ARIMA模型预测结果')
    plt.xlabel('月份')
    plt.ylabel('销售量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 未来预测
    model_fit, _, _, _ = arima_forecast(sales, best_p, best_d, best_q)
    future_steps = 12  # 预测未来12个月
    future_pred = model_fit.forecast(steps=future_steps)
    
    # 可视化未来预测
    plt.figure(figsize=(12, 6))
    plt.plot(time, sales, label='历史数据')
    plt.plot(np.arange(n_periods, n_periods+future_steps), 
             future_pred, label=f'未来{future_steps}个月预测', color='red', linestyle='--')
    plt.title('ARIMA模型未来销售量预测')
    plt.xlabel('月份')
    plt.ylabel('销售量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
