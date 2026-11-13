# 10. 多目标优化模型
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 定义多目标优化问题
def multi_objective_function(x):
    """
    示例多目标优化函数
    
    参数:
        x: 决策变量数组
    
    返回:
        两个目标函数值（需要最小化）
    """
    # 目标1：最小化函数值
    f1 = np.sum((x - 1)**2)
    
    # 目标2：最小化另一个函数值
    f2 = np.sum((x + 1)** 2)
    
    return [f1, f2]

# 使用pymoo库定义问题
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):
    def __init__(self, n_var=2, n_obj=2, n_constr=0, xl=-5, xu=5, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu,** kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.sum((x - 1)** 2)
        f2 = np.sum((x + 1)**2)
        out["F"] = [f1, f2]

# 实际应用：生产计划多目标优化
class ProductionProblem(ElementwiseProblem):
    def __init__(self):
        # 2个决策变量：产品A和B的产量
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=[0, 0], xu=[100, 100])

    def _evaluate(self, x, out, *args, **kwargs):
        a, b = x  # a: 产品A产量, b: 产品B产量
        
        # 目标1：最小化生产成本
        cost = 3*a + 2*b
        
        # 目标2：最小化生产时间
        time = 2*a + 3*b
        
        # 约束条件
        # 约束1：原材料限制：2a + b <= 150
        constr1 = 2*a + b - 150
        
        # 约束2：劳动力限制：a + 2b <= 100
        constr2 = a + 2*b - 100
        
        out["F"] = [cost, time]
        out["G"] = [constr1, constr2]  # 约束：G <= 0

# 多目标优化函数
def multi_objective_optimization(problem, n_var=2, n_gen=100):
    """
    执行多目标优化
    
    参数:
        problem: 问题定义
        n_var: 变量数量
        n_gen: 迭代次数
    
    返回:
        result: 优化结果
    """
    # 定义算法
    algorithm = NSGA2(pop_size=100)
    
    # 定义终止条件
    termination = get_termination("n_gen", n_gen)
    
    # 执行优化
    result = minimize(problem,
                      algorithm,
                      termination,
                      seed=42,
                      save_history=True,
                      verbose=False)
    
    return result

# 演示
if __name__ == "__main__":
    # 1. 简单多目标优化示例
    print("=== 简单多目标优化示例 ===")
    problem = MyProblem(n_var=2)
    result = multi_objective_optimization(problem)
    
    # 可视化Pareto前沿
    plot = Scatter(title="Pareto前沿")
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(result.F, color="red")
    plt.title("多目标优化的Pareto前沿")
    plt.xlabel("目标函数1")
    plt.ylabel("目标函数2")
    plt.show()
    
    # 2. 生产计划多目标优化
    print("\n=== 生产计划多目标优化 ===")
    prod_problem = ProductionProblem()
    prod_result = multi_objective_optimization(prod_problem)
    
    # 可视化生产计划的Pareto前沿
    plot = Scatter(title="生产计划的Pareto前沿")
    plot.add(prod_result.F, color="blue")
    plt.title("生产计划优化的Pareto前沿")
    plt.xlabel("生产成本")
    plt.ylabel("生产时间")
    plt.show()
    
    # 显示几个代表性解
    print("代表性解决方案:")
    indices = [0, len(prod_result.F)//2, -1]  # 选择三个代表性点
    for i in indices:
        a, b = prod_result.X[i].round(2)
        cost, time = prod_result.F[i].round(2)
        print(f"产品A: {a}, 产品B: {b} → 成本: {cost}, 时间: {time}")
    
    # 可视化生产计划
    plt.figure(figsize=(10, 6))
    plt.scatter(prod_result.X[:, 0], prod_result.X[:, 1], color="blue", alpha=0.7)
    plt.title("Pareto最优生产计划")
    plt.xlabel("产品A产量")
    plt.ylabel("产品B产量")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. 选择一个折中解（例如，成本和时间加权相等）
    weights = [0.5, 0.5]  # 权重相等
    weighted_sum = np.sum(prod_result.F * weights, axis=1)
    best_idx = np.argmin(weighted_sum)
    
    best_a, best_b = prod_result.X[best_idx].round(2)
    best_cost, best_time = prod_result.F[best_idx].round(2)
    
    print(f"\n折中最优解: 产品A: {best_a}, 产品B: {best_b} → 成本: {best_cost}, 时间: {best_time}")
