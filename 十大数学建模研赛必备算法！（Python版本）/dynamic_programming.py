# 动态规划核心算法简化版
def fibonacci_dp(n):
    """斐波那契数列动态规划解法"""
    if n <= 0:
        return 0, []
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n], dp

def knapsack_dp(weights, values, capacity):
    """0-1背包问题动态规划解法"""
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    
    # 填充DP表
    for i in range(1, n+1):
        for j in range(1, capacity+1):
            if weights[i-1] > j:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + values[i-1])
    
    # 回溯找解
    selected = []
    remaining = capacity
    for i in range(n, 0, -1):
        if dp[i][remaining] != dp[i-1][remaining]:
            selected.append(i-1)
            remaining -= weights[i-1]
    return dp[n][capacity], selected[::-1]

def investment_dp(projects, budget):
    """投资分配问题动态规划解法"""
    n = len(projects)
    dp = [[0]*(budget+1) for _ in range(n+1)]
    
    # 填充DP表
    for i in range(1, n+1):
        for j in range(budget+1):
            max_val = 0
            for k in range(min(j, len(projects[i-1])-1)+1):
                max_val = max(max_val, projects[i-1][k] + dp[i-1][j-k])
            dp[i][j] = max_val
    
    # 回溯找解
    allocation = [0]*n
    remaining = budget
    for i in range(n, 0, -1):
        for k in range(min(remaining, len(projects[i-1])-1), -1, -1):
            if dp[i][remaining] == projects[i-1][k] + dp[i-1][remaining-k]:
                allocation[i-1] = k
                remaining -= k
                break
    return dp[n][budget], allocation

# 演示
if __name__ == "__main__":
    # 斐波那契示例
    fib, _ = fibonacci_dp(10)
    print(f"第10个斐波那契数: {fib}")
    
    # 0-1背包示例
    weights = [2,3,4,5,6]
    values = [3,4,5,6,7]
    max_val, selected = knapsack_dp(weights, values, 10)
    print(f"背包最大价值: {max_val}, 选中物品: {selected}")
    
    # 投资分配示例
    projects = [[0,2,5,8,10], [0,3,6,8,11], [0,1,4,9,12]]
    profit, alloc = investment_dp(projects, 4)
    print(f"最大收益: {profit}, 投资分配: {alloc}")
