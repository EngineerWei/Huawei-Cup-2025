# 1. 遗传算法（Genetic Algorithm）
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def genetic_algorithm(objective_func, dim, pop_size=50, max_iter=100, 
                     lb=-5, ub=5, crossover_rate=0.8, mutation_rate=0.1):
    """
    遗传算法实现
    
    参数:
        objective_func: 目标函数（需要最小化）
        dim: 问题维度
        pop_size: 种群大小
        max_iter: 最大迭代次数
        lb: 变量下界
        ub: 变量上界
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
    
    返回:
        best_solution: 最优解
        best_fitness: 最优适应度值
        history: 每代最优适应度历史
    """
    # 初始化种群
    population = np.random.uniform(lb, ub, (pop_size, dim))
    history = []
    
    for _ in range(max_iter):
        # 计算适应度
        fitness = np.array([objective_func(ind) for ind in population])
        
        # 记录当代最优
        best_idx = np.argmin(fitness)
        history.append(fitness[best_idx])
        
        # 选择操作（轮盘赌选择）
        # 注意：这里使用适应度的倒数，因为我们要最小化目标函数
        prob = 1 / (fitness + 1e-8)  # 防止除以零
        prob = prob / np.sum(prob)
        selected_indices = np.random.choice(pop_size, size=pop_size, p=prob)
        selected_pop = population[selected_indices]
        
        # 交叉操作
        for i in range(0, pop_size, 2):
            if np.random.rand() < crossover_rate and i+1 < pop_size:
                # 单点交叉
                point = np.random.randint(1, dim)
                selected_pop[i, point:], selected_pop[i+1, point:] = \
                selected_pop[i+1, point:].copy(), selected_pop[i, point:].copy()
        
        # 变异操作
        for i in range(pop_size):
            for j in range(dim):
                if np.random.rand() < mutation_rate:
                    # 随机重置为[lb, ub]范围内的值
                    selected_pop[i, j] = np.random.uniform(lb, ub)
        
        # 精英保留策略：保留上一代最优个体
        population = selected_pop
        population[0] = population[best_idx]  # 保留最优个体
    
    # 最终结果
    best_idx = np.argmin([objective_func(ind) for ind in population])
    best_solution = population[best_idx]
    best_fitness = objective_func(best_solution)
    
    return best_solution, best_fitness, history

# 测试函数：Sphere函数（用于验证算法）
def sphere_function(x):
    return np.sum(x**2)

# 实际应用：资源分配问题
def resource_allocation_problem(x):
    """
    资源分配优化问题：将资源分配给5个项目，最大化总收益
    
    参数:
        x: 资源分配比例，总和应接近1
    """
    # 确保资源分配总和为1
    x = np.abs(x)
    x = x / np.sum(x)
    
    # 各项目的收益函数（示例）
    profits = [
        10 * np.sqrt(x[0]),    # 项目1：边际收益递减
        8 * np.log(x[1] + 1),  # 项目2：对数增长
        15 * x[2],             # 项目3：线性增长
        20 * (1 - np.exp(-5*x[3])),  # 项目4：S型增长
        12 * x[4] ** 0.7       # 项目5：缓慢增长
    ]
    
    # 转换为最小化问题
    return -np.sum(profits)

# 演示
if __name__ == "__main__":
    # 1. 测试Sphere函数优化
    print("=== Sphere函数优化 ===")
    dim = 5
    best_sol, best_fit, history = genetic_algorithm(
        sphere_function, dim, max_iter=50, lb=-10, ub=10
    )
    print(f"最优解: {best_sol.round(4)}")
    print(f"最优值: {best_fit:.6f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title("Sphere函数优化收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("最优适应度值")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. 资源分配问题优化
    print("\n=== 资源分配问题优化 ===")
    dim = 5  # 5个项目
    best_alloc, best_profit, history = genetic_algorithm(
        resource_allocation_problem, dim, max_iter=100, lb=0, ub=1
    )
    
    # 处理结果
    best_alloc = np.abs(best_alloc)
    best_alloc = best_alloc / np.sum(best_alloc)
    total_profit = -best_profit
    
    print(f"最优资源分配: {best_alloc.round(4)}")
    print(f"最大总收益: {total_profit:.4f}")
    
    # 绘制资源分配饼图
    plt.figure(figsize=(8, 6))
    plt.pie(best_alloc, labels=[f"项目{i+1}" for i in range(dim)], 
            autopct='%1.1f%%', startangle=90)
    plt.title("最优资源分配方案")
    plt.axis('equal')
    plt.show()
