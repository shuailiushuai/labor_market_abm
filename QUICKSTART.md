# Labor Market ABM - Quick Start Guide

## 快速开始指南

### 安装依赖

```bash
pip install numpy matplotlib seaborn scipy pandas
```

### 基本使用

#### 1. 运行原始模型
```python
from model_class import Model
import numpy as np

np.random.seed(42)
m = Model(T=500, H=200, F=20)
m.run()

# 查看结果
print(f"Final unemployment: {m.u_r_arr[-1]:.3f}")
print(f"Mean wage: {m.mean_w_arr[-1]:.3f}")
```

#### 2. 技能失配分析（最重要的新功能）
```python
from skill_mismatch_analysis import SkillMismatchAnalyzer
from model_class import Model
import numpy as np

# 运行模型（允许NR工人从事R工作）
np.random.seed(42)
m = Model(T=500, H=200, F=20, nr_to_r=True)
m.run()

# 分析错配
analyzer = SkillMismatchAnalyzer(m)

# 获取错配指标
metrics = analyzer.get_mismatch_metrics()
print(f"Over-qualified: {metrics['overqualified_rate']:.2%}")
print(f"Under-qualified: {metrics['underqualified_rate']:.2%}")
print(f"Total mismatch: {metrics['total_mismatch_rate']:.2%}")

# 获取工资损失
wage_metrics = analyzer.get_wage_penalties()
print(f"Overqualification penalty: {wage_metrics['overqualification_penalty']:.2%}")

# 生成图表和报告
fig, df = analyzer.plot_mismatch_evolution(start=100, save_path='mismatch.png')
analyzer.generate_mismatch_report(start=100, save_path='mismatch_report.txt')
```

#### 3. 数字化与自动化研究
```python
from digitalization_regulation import DigitalizationModel, DigitalizationAnalyzer

# 比较不同政策场景
scenarios = {
    'Baseline': {
        'H': 200, 'F': 20, 'T': 500,
        'automation_rate': 0.0,
    },
    'High Automation': {
        'H': 200, 'F': 20, 'T': 500,
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
        'automation_cost_reduction': 0.3,
    },
    'Automation + Retraining': {
        'H': 200, 'F': 20, 'T': 500,
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
        'retraining_policy': True,
        'training_effectiveness': 0.3,
    },
}

analyzer = DigitalizationAnalyzer()
results = analyzer.compare_scenarios(scenarios, T=500, n_runs=3)

# 生成报告
analyzer.plot_scenario_comparison('scenarios.png')
analyzer.generate_policy_report('policy_report.txt')

# 查看结果
for scenario, metrics in results.items():
    print(f"\n{scenario}:")
    print(f"  Unemployment: {metrics['unemployment_mean']:.3f}")
    print(f"  Skill mismatch: {metrics['mismatch_mean']:.3f}")
```

#### 4. 敏感性分析
```python
from sensitivity_analysis import SensitivityAnalysis

# 定义要测试的参数范围
param_ranges = {
    'sigma_m': (0.001, 0.01),
    'sigma_w': (0.001, 0.01),
    'lambda_LM': (1, 10),
    'mu_r': (0.2, 0.5),
}

# 基准参数
base_params = {
    'H': 200, 'F': 20, 'T': 300,
    'alpha_2': 0.25, 'chi_C': 0.2,
    'nu': 0.1, 'u_r': 0.08,
    'beta': 1, 'lambda_exp': 0.5,
    'N_app': 4, 'sigma': 1.5,
    'nr_to_r': False, 'a': 100,
    'min_w_par': 0.3, 'W_r': 1,
}

sa = SensitivityAnalysis(param_ranges, T=300, burnin=100)

# OAT分析
oat_results = sa.oat_sensitivity(base_params, n_samples=5)
sa.plot_oat_results('sensitivity_oat.png')

# Morris筛选
morris_results = sa.morris_screening(base_params, n_trajectories=5)
sa.plot_morris_results('sensitivity_morris.png')

# 保存结果
sa.save_results('sensitivity_results.csv')
```

#### 5. 一键运行所有分析
```python
from analysis_runner import ComprehensiveAnalyzer

# 创建分析器
analyzer = ComprehensiveAnalyzer(output_dir='my_analysis_results')

# 运行分析
analyzer.run_full_analysis(
    run_sensitivity=True,    # 敏感性分析
    run_mismatch=True,       # 技能错配分析
    run_digitalization=True, # 数字化场景
    run_validation=False,    # 经验验证（需要真实数据）
    run_estimation=False     # 参数估计（耗时较长）
)

# 所有结果保存在 my_analysis_results/ 目录
```

## 核心研究问题与代码对应

### 研究问题1：技能失配的程度和影响

**代码：**
```python
from skill_mismatch_analysis import SkillMismatchAnalyzer

analyzer = SkillMismatchAnalyzer(model)
metrics = analyzer.get_mismatch_metrics()
wage_metrics = analyzer.get_wage_penalties()
```

**关键指标：**
- `overqualified_rate`: 过度技能率（NR工人在R岗位）
- `underqualified_rate`: 技能不足率（R工人在NR岗位）
- `overqualification_penalty`: 过度技能的工资损失
- `unemployment_gap`: 不同技能类型的失业率差异

### 研究问题2：数字化对技能失配的影响

**代码：**
```python
from digitalization_regulation import DigitalizationModel

# 场景对比
m_baseline = Model(T=500, nr_to_r=True)
m_automation = DigitalizationModel(
    T=500, 
    nr_to_r=True,
    automation_rate=0.02,
    automation_start=100
)

# 比较两个场景的技能错配
```

**关键参数：**
- `automation_rate`: 自动化速度
- `automation_routine_impact`: 对常规工作的影响程度
- `automation_cost_reduction`: 成本降低幅度

### 研究问题3：劳动力市场规制的效果

**代码：**
```python
scenarios = {
    'No Regulation': {...},
    'Higher Min Wage': {
        'min_wage_policy': {'start': 100, 'level': 0.6}
    },
    'Retraining Program': {
        'retraining_policy': True,
        'training_effectiveness': 0.3
    },
    'Hiring/Firing Costs': {
        'firing_cost': 0.5,
        'hiring_subsidy': 0.2
    }
}
```

### 研究问题4：参数敏感性和稳健性

**代码：**
```python
from sensitivity_analysis import SensitivityAnalysis

sa = SensitivityAnalysis(param_ranges)
oat_results = sa.oat_sensitivity(base_params)
morris_results = sa.morris_screening(base_params)
```

**关键输出：**
- Morris μ* 值：参数对输出的平均影响
- Morris σ 值：参数的非线性效应和交互效应

## 常见问题

**Q1: 如何设置让NR工人可以从事R工作？**

A: 在模型初始化时设置 `nr_to_r=True`：
```python
m = Model(T=500, nr_to_r=True)
```

**Q2: 如何加速分析运行？**

A: 减少模拟期数和重复次数：
```python
# 快速测试
m = Model(T=200)  # 减少期数
analyzer.compare_scenarios(scenarios, n_runs=2)  # 减少重复
```

**Q3: 如何使用自己的经验数据？**

A: 替换经验矩字典：
```python
empirical_moments = {
    'mean_unemployment': 0.055,  # 你的数据
    'mean_real_wage': 1.02,      # 你的数据
    'wage_inequality_9010': 3.8, # 你的数据
}
```

**Q4: 如何保存模型结果？**

A: 使用pandas保存：
```python
import pandas as pd

results = {
    'unemployment': m.u_r_arr,
    'wages': m.mean_w_arr,
    'gdp': m.GDP
}
df = pd.DataFrame(results)
df.to_csv('model_results.csv', index=False)
```

## 输出文件说明

运行分析后，会在输出目录生成以下文件：

- `sensitivity_*.png/csv`: 敏感性分析结果
- `mismatch_*.png/txt`: 技能错配分析结果
- `scenario_comparison.png`: 场景比较图
- `policy_report.txt`: 政策分析报告
- `ANALYSIS_SUMMARY.txt`: 总体分析摘要

## 下一步

1. 阅读 `ANALYSIS_MODULES_README.md` 了解详细功能
2. 运行 `analysis_runner.py` 进行完整分析
3. 根据研究需求修改参数和场景
4. 使用真实数据进行经验验证和参数估计

## 技术支持

遇到问题请查看：
- 详细文档：`ANALYSIS_MODULES_README.md`
- 示例代码：各模块的 `example_*` 函数
- GitHub Issues

祝研究顺利！
