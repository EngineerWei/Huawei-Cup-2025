# 3. LSTM模型（长短期记忆网络）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def create_sequences(data, seq_length):
    """创建输入序列和目标值"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # 输入序列
        y.append(data[i+seq_length])    # 目标值
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, units=50):
    """构建LSTM模型"""
    model = Sequential()
    model.add(LSTM(units, input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def lstm_forecast(data, seq_length=10, epochs=50, units=50):
    """
    使用LSTM进行时间序列预测
    
    参数:
        data: 时间序列数据
        seq_length: 序列长度
        epochs: 训练轮数
        units: LSTM单元数量
    
    返回:
        model: 训练好的模型
        scaler: 数据缩放器
        train_predict: 训练集预测
        test_predict: 测试集预测
    """
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 创建序列
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # 转换为LSTM输入格式
    
    # 划分训练集和测试集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 构建和训练模型
    model = build_lstm_model(seq_length, units)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    # 预测
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # 反归一化
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return model, scaler, train_predict, test_predict, y_train, y_test

# 演示
if __name__ == "__main__":
    # 生成示例数据（模拟月度销售额）
    np.random.seed(42)
    months = 120  # 10年数据
    time = np.arange(months)
    
    # 生成带有趋势和季节性的数据
    trend = 0.5 * time
    seasonality = 10 * np.sin(2 * np.pi * time / 12)  # 年度季节性
    noise = np.random.normal(0, 2, months)
    sales = trend + seasonality + noise
    sales = np.maximum(sales, 5)  # 确保销售额为正
    
    # 可视化原始数据
    plt.figure(figsize=(12, 6))
    plt.plot(time, sales)
    plt.title('月度销售额数据')
    plt.xlabel('月份')
    plt.ylabel('销售额')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 应用LSTM预测
    seq_length = 12  # 使用过去12个月预测下一个月
    model, scaler, train_predict, test_predict, y_train, y_test = lstm_forecast(
        sales, seq_length=seq_length, epochs=50
    )
    
    # 计算误差
    train_rmse = np.sqrt(np.mean((train_predict - y_train)**2))
    test_rmse = np.sqrt(np.mean((test_predict - y_test)** 2))
    print(f"训练集RMSE: {train_rmse:.2f}")
    print(f"测试集RMSE: {test_rmse:.2f}")
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(time, sales, label='原始数据', alpha=0.5)
    plt.plot(time[seq_length:seq_length+len(train_predict)], 
             train_predict, label='训练集预测', color='green')
    plt.plot(time[seq_length+len(train_predict):], 
             test_predict, label='测试集预测', color='red')
    plt.title('LSTM销售额预测')
    plt.xlabel('月份')
    plt.ylabel('销售额')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 未来预测
    future_months = 12
    last_sequence = sales[-seq_length:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    
    future_predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(future_months):
        x_input = current_sequence.reshape(1, seq_length, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        future_predictions.append(pred)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # 可视化未来预测
    plt.figure(figsize=(12, 6))
    plt.plot(time, sales, label='历史数据')
    plt.plot(np.arange(months, months+future_months), 
             future_predictions, label='未来预测', color='red', linestyle='--')
    plt.title(f'未来{future_months}个月销售额预测')
    plt.xlabel('月份')
    plt.ylabel('销售额')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
