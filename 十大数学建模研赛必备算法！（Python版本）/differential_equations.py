# 9. 微分方程模型
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 设置中文显示和全局绘图样式
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.style.use('seaborn-v0_8-notebook')  # 使用美观的绘图样式


def exponential_growth(t, y, r):
    """指数增长模型: dy/dt = r*y"""
    return r * y


def logistic_growth(t, y, r, K):
    """逻辑斯蒂增长模型: dy/dt = r*y*(1 - y/K)"""
    return r * y * (1 - y / K)


def sirs_model(t, y, beta, gamma, sigma):
    """
    SIRS模型（易感-感染-恢复-易感）
    y = [S, I, R]
    dS/dt = -beta*S*I + sigma*R
    dI/dt = beta*S*I - gamma*I
    dR/dt = gamma*I - sigma*R
    """
    S, I, R = y
    dSdt = -beta * S * I + sigma * R
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I - sigma * R
    return [dSdt, dIdt, dRdt]


def solve_ode(deriv_func, t_span, y0, params):
    """求解微分方程并返回解"""
    return solve_ivp(
        deriv_func, t_span, y0, args=params,
        method='RK45', dense_output=True
    )


def plot_solution(solution, t_span, labels, title, figsize=(12, 6)):
    """绘制微分方程的解"""
    t = np.linspace(t_span[0], t_span[1], 1000)
    y = solution.sol(t)
    
    plt.figure(figsize=figsize)
    for i in range(y.shape[0]):
        plt.plot(t, y[i], label=labels[i], linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # 自动调整布局
    plt.show()


def run_demo():
    """运行所有演示案例"""
    # 1. 指数增长模型
    print("=== 指数增长模型 ===")
    # 模型参数
    exp_model_params = {
        'r': 0.1,
        't_span': (0, 50),
        'y0': [10]
    }
    # 绘图参数
    exp_plot_params = {
        't_span': (0, 50),
        'labels': ['种群数量'],
        'title': f'指数增长模型 (r={exp_model_params["r"]})'
    }
    solution = solve_ode(exponential_growth, 
                        exp_model_params['t_span'], 
                        exp_model_params['y0'], 
                        (exp_model_params['r'],))
    plot_solution(solution, **exp_plot_params)
    
    # 2. 逻辑斯蒂增长模型
    print("\n=== 逻辑斯蒂增长模型 ===")
    log_model_params = {
        'r': 0.1,
        'K': 1000,
        't_span': (0, 100),
        'y0': [10]
    }
    log_plot_params = {
        't_span': (0, 100),
        'labels': ['种群数量'],
        'title': f'逻辑斯蒂增长模型 (r={log_model_params["r"]}, K={log_model_params["K"]})'
    }
    solution = solve_ode(logistic_growth, 
                        log_model_params['t_span'], 
                        log_model_params['y0'], 
                        (log_model_params['r'], log_model_params['K']))
    plot_solution(solution,** log_plot_params)
    
    # 3. SIRS传染病模型
    print("\n=== SIRS传染病模型 ===")
    sirs_model_params = {
        'beta': 0.3,
        'gamma': 0.1,
        'sigma': 0.05,
        't_span': (0, 100),
        'y0': [999/1000, 1/1000, 0]  # 归一化处理
    }
    sirs_plot_params = {
        't_span': (0, 100),
        'labels': ['易感者', '感染者', '恢复者'],
        'title': f'SIRS模型 (β={sirs_model_params["beta"]}, γ={sirs_model_params["gamma"]}, σ={sirs_model_params["sigma"]})'
    }
    solution = solve_ode(sirs_model, 
                        sirs_model_params['t_span'], 
                        sirs_model_params['y0'], 
                        (sirs_model_params['beta'], sirs_model_params['gamma'], sirs_model_params['sigma']))
    plot_solution(solution, **sirs_plot_params)
    
    # 4. 不同参数对比
    print("\n=== 不同感染率下的SIRS模型对比 ===")
    t = np.linspace(0, 100, 1000)
    betas = [0.2, 0.3, 0.4]
    gamma = 0.1
    sigma = 0.05
    y0 = [999/1000, 1/1000, 0]
    
    plt.figure(figsize=(12, 6))
    for beta in betas:
        solution = solve_ode(sirs_model, (0, 100), y0, (beta, gamma, sigma))
        y = solution.sol(t)
        plt.plot(t, y[1], label=f'β={beta}', linewidth=2)
    
    plt.title('不同感染率下的感染者数量变化', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('感染者比例', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
