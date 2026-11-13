# 5. 整数规划
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def integer_programming_demo():
    """整数规划示例：生产计划优化"""
    print("=== 生产计划优化 ===")
    
    # 问题：某工厂生产A、B两种产品，最大化利润
    # 产品A：利润50元/件，需2小时工时，1单位材料
    # 产品B：利润30元/件，需1小时工时，1单位材料
    # 约束：总工时≤100小时，总材料≤80单位
    
    # 目标函数系数（最大化问题转为最小化）
    c = [-50, -30]  # 注意符号取反
    
    # 不等式约束 A_ub * x <= b_ub
    A_ub = [
        [2, 1],  # 工时约束
        [1, 1]   # 材料约束
    ]
    b_ub = [100, 80]  # 资源上限
    
    # 变量边界（x1, x2 ≥ 0 且为整数）
    bounds = [(0, None), (0, None)]
    
    # 求解（scipy的linprog不直接支持整数约束，这里用分支定界法思想近似）
    # 注意：实际应用中建议使用专业整数规划求解器如Gurobi或OR-Tools
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # 处理整数约束（四舍五入）
    x_int = np.round(result.x).astype(int)
    x_int = np.maximum(x_int, 0)  # 确保非负
    
    # 检查约束是否满足
    hours_used = 2*x_int[0] + x_int[1]
    materials_used = x_int[0] + x_int[1]
    
    # 如果违反约束，适当调整
    if hours_used > 100 or materials_used > 80:
        if x_int[0] > 0:
            x_int[0] -= 1
        else:
            x_int[1] -= 1
    
    # 计算结果
    profit = 50*x_int[0] + 30*x_int[1]
    hours_used = 2*x_int[0] + x_int[1]
    materials_used = x_int[0] + x_int[1]
    
    print(f"最优生产计划：")
    print(f"产品A: {x_int[0]}件")
    print(f"产品B: {x_int[1]}件")
    print(f"总利润: {profit}元")
    print(f"使用工时: {hours_used}/100小时")
    print(f"使用材料: {materials_used}/80单位")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.bar(['产品A', '产品B'], x_int, color=['blue', 'orange'])
    plt.title('最优生产计划')
    plt.ylabel('生产数量')
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()
    
    return x_int, profit

def knapsack_problem_demo():
    """整数规划应用：0-1背包问题"""
    print("\n=== 0-1背包问题 ===")
    
    # 问题：有5个物品，重量分别为[2, 3, 4, 5, 6]，价值分别为[3, 4, 5, 6, 7]
    # 背包容量为10，选择物品使总价值最大，每个物品要么选要么不选
    
    # 物品数据
    weights = [2, 3, 4, 5, 6]
    values = [3, 4, 5, 6, 7]
    capacity = 10
    n_items = len(weights)
    
    # 目标函数系数（最大化问题转为最小化）
    c = [-v for v in values]
    
    # 约束条件：总重量 <= 容量
    A_ub = [weights]
    b_ub = [capacity]
    
    # 变量边界（0或1）
    bounds = [(0, 1) for _ in range(n_items)]
    
    # 求解
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # 处理整数约束
    x_int = np.round(result.x).astype(int)
    
    # 计算结果
    total_value = sum(v * x for v, x in zip(values, x_int))
    total_weight = sum(w * x for w, x in zip(weights, x_int))
    
    print(f"最优选择：")
    for i in range(n_items):
        status = "选中" if x_int[i] == 1 else "未选中"
        print(f"物品{i+1}: 重量{weights[i]}, 价值{values[i]} - {status}")
    
    print(f"总价值: {total_value}")
    print(f"总重量: {total_weight}/{capacity}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    selected = [values[i] if x_int[i] == 1 else 0 for i in range(n_items)]
    plt.bar([f'物品{i+1}' for i in range(n_items)], selected, color=['green' if x else 'red' for x in x_int])
    plt.title('背包问题最优选择')
    plt.ylabel('物品价值')
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()
    
    return x_int, total_value

# 演示
if __name__ == "__main__":
    # 生产计划优化
    prod_plan, profit = integer_programming_demo()
    
    # 0-1背包问题
    knapsack_sol, knapsack_val = knapsack_problem_demo()
    
    print("\n提示：对于更复杂的整数规划问题，建议使用专业求解器如Gurobi、CPLEX或OR-Tools")
