# 2. 粒子群算法（Particle Swarm Optimization）
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def particle_swarm_optimization(objective_func, dim, num_particles=30, max_iter=100,
                              lb=-5, ub=5, w=0.5, c1=1, c2=1):
    """
    粒子群优化算法实现
    
    参数:
        objective_func: 目标函数（需要最小化）
        dim: 问题维度
        num_particles: 粒子数量
        max_iter: 最大迭代次数
        lb: 变量下界
        ub: 变量上界
        w: 惯性权重
        c1: 认知系数
        c2: 社会系数
    
    返回:
        gbest_position: 全局最优位置
        gbest_value: 全局最优值
        history: 每代最优值历史
    """
    # 初始化粒子位置和速度
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.random.uniform(-(ub-lb), ub-lb, (num_particles, dim))
    
    # 初始化个体最优和全局最优
    pbest_positions = positions.copy()
    pbest_values = np.array([objective_func(p) for p in positions])
    gbest_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_value = pbest_values[gbest_idx]
    
    # 记录历史最优
    history = [gbest_value]
    
    for _ in range(max_iter):
        for i in range(num_particles):
            # 生成随机数
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            
            # 更新速度
            cognitive = c1 * r1 * (pbest_positions[i] - positions[i])
            social = c2 * r2 * (gbest_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social
            
            # 更新位置
            positions[i] += velocities[i]
            
            # 边界处理
            positions[i] = np.clip(positions[i], lb, ub)
            
            # 计算适应度
            current_value = objective_func(positions[i])
            
            # 更新个体最优
            if current_value < pbest_values[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_values[i] = current_value
                
                # 更新全局最优
                if current_value < gbest_value:
                    gbest_position = positions[i].copy()
                    gbest_value = current_value
        
        # 记录历史
        history.append(gbest_value)
    
    return gbest_position, gbest_value, history

# 测试函数：Rosenbrock函数
def rosenbrock_function(x):
    return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])** 2)

# 实际应用：投资组合优化
def portfolio_optimization(weights):
    """投资组合优化：最小化风险同时考虑收益"""
    # 确保权重和为1
    weights = np.maximum(weights, 0)  # 不允许卖空
    weights = weights / np.sum(weights)
    
    # 资产收益率和协方差矩阵（示例数据）
    returns = np.array([0.08, 0.12, 0.15, 0.09, 0.11])
    cov_matrix = np.array([
        [0.01, 0.003, 0.002, 0.001, 0.004],
        [0.003, 0.02, 0.005, 0.002, 0.006],
        [0.002, 0.005, 0.03, 0.003, 0.007],
        [0.001, 0.002, 0.003, 0.015, 0.002],
        [0.004, 0.006, 0.007, 0.002, 0.025]
    ])
    
    # 计算风险和收益
    portfolio_return = np.sum(returns * weights)
    portfolio_risk = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # 最小化负的夏普比率（风险调整后收益）
    return -portfolio_return / (np.sqrt(portfolio_risk) + 1e-8)

# 演示
if __name__ == "__main__":
    # 1. 测试Rosenbrock函数优化
    print("=== Rosenbrock函数优化 ===")
    dim = 3
    best_pos, best_val, history = particle_swarm_optimization(
        rosenbrock_function, dim, max_iter=100, lb=-2, ub=2
    )
    print(f"最优解: {best_pos.round(4)}")
    print(f"最优值: {best_val:.6f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title("Rosenbrock函数优化收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("最优值")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. 投资组合优化
    print("\n=== 投资组合优化 ===")
    num_assets = 5  # 5种资产
    best_weights, best_sharpe, history = particle_swarm_optimization(
        portfolio_optimization, num_assets, max_iter=100, lb=0, ub=1
    )
    
    # 处理结果
    best_weights = np.maximum(best_weights, 0)
    best_weights = best_weights / np.sum(best_weights)
    
    print(f"最优投资组合权重: {best_weights.round(4)}")
    print(f"最大夏普比率: {(-best_sharpe):.4f}")
    
    # 可视化投资组合
    plt.figure(figsize=(8, 6))
    plt.pie(best_weights, labels=[f"资产{i+1}" for i in range(num_assets)],
            autopct='%1.1f%%', startangle=90)
    plt.title("最优投资组合分配")
    plt.axis('equal')
    plt.show()
