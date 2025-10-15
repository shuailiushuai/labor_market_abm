# Labor Market ABM - Analysis Modules

## 概述 (Overview)

本项目扩展了劳动力市场代理建模（ABM），添加了以下分析模块：

1. **敏感性分析** (Sensitivity Analysis)
2. **技能失配分析** (Skill Mismatch Analysis)
3. **数字化与劳动力市场规制** (Digitalization & Labor Market Regulation)
4. **经验验证** (Empirical Validation)
5. **参数估计** (Parameter Estimation)

## 新增模块说明

### 1. 敏感性分析 (sensitivity_analysis.py)

**功能：**
- Sobol全局敏感性分析
- One-At-A-Time (OAT) 局部敏感性分析
- Morris筛选法识别关键参数
- 参数重要性排序

**使用示例：**
```python
from sensitivity_analysis import SensitivityAnalysis, example_sensitivity_analysis

# 快速运行示例
sa = example_sensitivity_analysis()

# 或自定义分析
param_ranges = {
    'sigma_m': (0.001, 0.01),
    'lambda_LM': (1, 10),
    'mu_r': (0.2, 0.5),
}

base_params = {
    'H': 200, 'F': 20, 'T': 500,
    # ... 其他参数
}

sa = SensitivityAnalysis(param_ranges, T=500, burnin=200)
oat_results = sa.oat_sensitivity(base_params, n_samples=10)
morris_results = sa.morris_screening(base_params, n_trajectories=10)

sa.plot_oat_results('sensitivity_oat.png')
sa.plot_morris_results('sensitivity_morris.png')
sa.save_results('sensitivity_results.csv')
```

**输出：**
- `sensitivity_oat.png`: OAT敏感性图
- `sensitivity_morris.png`: Morris筛选图
- `sensitivity_results.csv`: 详细结果

### 2. 技能失配分析 (skill_mismatch_analysis.py)

**功能：**
- 过度技能 (Over-qualification): NR工人从事R工作
- 技能不足 (Under-qualification): R工人从事NR工作
- 技能错配 (Skill Mismatch): 总体错配率
- 工资损失/溢价分析
- 失业率按技能类型分解

**关键指标：**
- `overqualified_rate`: 过度技能率
- `underqualified_rate`: 技能不足率
- `total_mismatch_rate`: 总错配率
- `overqualification_penalty`: 过度技能的工资损失
- `underqualification_premium`: 技能不足的工资溢价

**使用示例：**
```python
from skill_mismatch_analysis import SkillMismatchAnalyzer, example_skill_mismatch_analysis
from model_class import Model

# 快速运行示例
analyzer, df = example_skill_mismatch_analysis()

# 或使用自己的模型
m = Model(T=500, nr_to_r=True)  # 允许NR工人从事R工作
m.run()

analyzer = SkillMismatchAnalyzer(m)

# 获取当前错配指标
metrics = analyzer.get_mismatch_metrics()
print(f"Over-qualification rate: {metrics['overqualified_rate']:.2%}")
print(f"Under-qualification rate: {metrics['underqualified_rate']:.2%}")

# 获取工资损失
wage_metrics = analyzer.get_wage_penalties()
print(f"Overqualification penalty: {wage_metrics['overqualification_penalty']:.2%}")

# 绘制时间演化
fig, df = analyzer.plot_mismatch_evolution(start=100)

# 生成报告
analyzer.generate_mismatch_report(start=100, save_path='mismatch_report.txt')
```

**输出：**
- `mismatch_evolution.png`: 错配演化图
- `mismatch_report.txt`: 详细分析报告

### 3. 数字化与劳动力市场规制 (digitalization_regulation.py)

**功能：**
- 自动化对常规工作的影响
- 劳动力市场规制（最低工资、雇佣/解雇成本）
- 技能提升与再培训政策
- 政策干预效果评估

**政策场景：**
1. **基准场景** (Baseline): 无干预
2. **自动化场景** (Automation): 逐步自动化常规工作
3. **自动化+再培训** (Automation + Retraining): 为失业工人提供培训
4. **自动化+更高最低工资** (Automation + Higher Min Wage): 提高最低工资保护

**使用示例：**
```python
from digitalization_regulation import DigitalizationModel, DigitalizationAnalyzer

# 单一场景模拟
m = DigitalizationModel(
    T=500,
    automation_rate=0.02,
    automation_start=100,
    automation_routine_impact=0.8,
    automation_cost_reduction=0.3,
    retraining_policy=True,
    training_effectiveness=0.3,
    training_cost=0.5
)
m.run()

# 比较多个场景
scenarios = {
    'Baseline': {'automation_rate': 0.0},
    'High Automation': {
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
    },
    'Automation + Retraining': {
        'automation_rate': 0.02,
        'automation_start': 100,
        'retraining_policy': True,
        'training_effectiveness': 0.3,
    }
}

analyzer = DigitalizationAnalyzer()
results = analyzer.compare_scenarios(scenarios, T=500, n_runs=5)
analyzer.plot_scenario_comparison('scenario_comparison.png')
analyzer.generate_policy_report('policy_report.txt')
```

**输出：**
- `scenario_comparison.png`: 场景比较图
- `policy_report.txt`: 政策分析报告

### 4. 经验验证 (empirical_validation.py)

**功能：**
- 矩匹配 (Moment Matching)
- 分布比较 (Distribution Comparison)
- 拟合优度检验 (Goodness-of-Fit Tests)
- 时间序列验证 (Time Series Validation)

**使用示例：**
```python
from empirical_validation import EmpiricalValidator
from model_class import Model

# 运行模型
m = Model(T=500)
m.run()

# 创建验证器
validator = EmpiricalValidator(m)

# 定义经验矩（使用真实数据替换）
empirical_moments = {
    'mean_unemployment': 0.06,
    'mean_real_wage': 0.95,
    'std_unemployment': 0.015,
    'wage_inequality_9010': 3.5,
    'corr_gdp_unemployment': -0.5,
}

# 矩匹配
moment_results = validator.moment_matching(empirical_moments)

# 拟合优度评估
gof = validator.goodness_of_fit(empirical_moments)
print(f"Fit score: {gof['fit_score']:.2f}")
print(f"Assessment: {gof['overall_assessment']}")

# 可视化
validator.plot_moment_comparison('moment_comparison.png')
validator.generate_validation_report('validation_report.txt')
```

**输出：**
- `moment_comparison.png`: 矩比较图
- `validation_report.txt`: 验证报告

### 5. 参数估计 (parameter_estimation.py)

**功能：**
- 模拟矩法 (Method of Simulated Moments, MSM)
- 网格搜索 (Grid Search)
- 近似贝叶斯计算 (Approximate Bayesian Computation, ABC)
- 参数优化

**使用示例：**
```python
from parameter_estimation import ParameterEstimator

# 经验数据
empirical_data = {
    'mean_unemployment': 0.06,
    'mean_real_wage': 0.95,
    'wage_inequality': 3.5,
}

# 要估计的参数范围
param_bounds = {
    'lambda_LM': (1, 10),
    'sigma_w': (0.001, 0.01),
    'mu_r': (0.2, 0.5),
}

# 固定参数
base_params = {
    'H': 200, 'F': 20,
    # ... 其他参数
}

# 创建估计器
estimator = ParameterEstimator(empirical_data, param_bounds, T=500, burnin=100)

# 方法1: 网格搜索（快速，粗略）
grid_results = estimator.grid_search(base_params, n_points=5)

# 方法2: MSM优化（精确）
msm_results = estimator.estimate_msm(base_params, method='differential_evolution')

# 方法3: ABC（贝叶斯）
abc_results = estimator.abc_estimation(base_params, n_samples=1000, tolerance=0.1)

# 可视化
estimator.plot_estimation_results('msm', 'msm_results.png')
```

**输出：**
- `msm_results.png`: MSM结果图
- 最佳参数估计

## 统一分析运行器 (analysis_runner.py)

提供一键运行所有分析的接口：

```python
from analysis_runner import ComprehensiveAnalyzer, run_quick_demo, run_full_suite

# 快速演示
analyzer = run_quick_demo()

# 完整分析
analyzer = run_full_suite()

# 自定义分析
analyzer = ComprehensiveAnalyzer(output_dir='my_results')
analyzer.run_full_analysis(
    run_sensitivity=True,
    run_mismatch=True,
    run_digitalization=True,
    run_validation=False,
    run_estimation=False
)
```

## 研究内容与代码扩展对应关系

### 技能失配研究

| 研究内容 | 代码模块 | 关键指标 |
|---------|---------|---------|
| 技能不足 (Under-skilling) | `skill_mismatch_analysis.py` | `underqualified_rate`, `underqualification_premium` |
| 技能过度 (Over-skilling) | `skill_mismatch_analysis.py` | `overqualified_rate`, `overqualification_penalty` |
| 技能错配 (Skill Mismatch) | `skill_mismatch_analysis.py` | `total_mismatch_rate` |
| 工资损失分析 | `skill_mismatch_analysis.py` | `get_wage_penalties()` |
| 错配演化 | `skill_mismatch_analysis.py` | `track_mismatch_over_time()` |

### 数字化影响研究

| 研究内容 | 代码模块 | 关键参数 |
|---------|---------|---------|
| 自动化对就业的影响 | `digitalization_regulation.py` | `automation_rate`, `automation_routine_impact` |
| 劳动力市场摩擦变化 | `digitalization_regulation.py` | `lambda_LM`, `N_app` |
| 再培训政策效果 | `digitalization_regulation.py` | `retraining_policy`, `training_effectiveness` |
| 最低工资政策 | `digitalization_regulation.py` | `min_wage_policy` |
| 场景比较 | `digitalization_regulation.py` | `DigitalizationAnalyzer.compare_scenarios()` |

### 劳动力市场规制研究

| 研究内容 | 代码模块 | 关键参数 |
|---------|---------|---------|
| 雇佣成本 | `digitalization_regulation.py` | `hiring_subsidy` |
| 解雇成本 | `digitalization_regulation.py` | `firing_cost` |
| 培训成本 | `digitalization_regulation.py` | `training_cost` |
| 最低工资 | `digitalization_regulation.py` | `min_wage_policy` |

## 完整研究流程

### 步骤1: 基础模型理解
```python
from model_class import Model
m = Model(T=500)
m.run()
# 检查基本输出
```

### 步骤2: 敏感性分析
```python
from sensitivity_analysis import example_sensitivity_analysis
sa = example_sensitivity_analysis()
# 识别关键参数
```

### 步骤3: 技能失配分析
```python
from skill_mismatch_analysis import example_skill_mismatch_analysis
analyzer, df = example_skill_mismatch_analysis()
# 理解错配动态
```

### 步骤4: 政策场景分析
```python
from digitalization_regulation import example_digitalization_analysis
digi_analyzer = example_digitalization_analysis()
# 评估政策效果
```

### 步骤5: 模型验证
```python
from empirical_validation import example_empirical_validation
validator = example_empirical_validation()
# 验证模型有效性
```

### 步骤6: 参数校准
```python
from parameter_estimation import example_parameter_estimation
estimator = example_parameter_estimation()
# 估计关键参数
```

## 依赖项

```bash
pip install numpy matplotlib seaborn scipy pandas
```

## 输出文件说明

所有分析输出保存在指定目录（默认 `analysis_results/`）：

- `sensitivity_oat.png`: OAT敏感性分析图
- `sensitivity_morris.png`: Morris筛选结果
- `sensitivity_results.csv`: 敏感性分析数据
- `mismatch_evolution.png`: 技能错配演化
- `mismatch_report.txt`: 技能错配报告
- `scenario_comparison.png`: 政策场景比较
- `policy_report.txt`: 政策分析报告
- `moment_comparison.png`: 矩匹配比较
- `validation_report.txt`: 验证报告
- `ANALYSIS_SUMMARY.txt`: 总体分析摘要

## 示例应用

### 示例1: 研究自动化对技能错配的影响

```python
from digitalization_regulation import DigitalizationModel, DigitalizationAnalyzer
from skill_mismatch_analysis import SkillMismatchAnalyzer

# 场景1: 无自动化
m_baseline = Model(T=500, nr_to_r=True)
m_baseline.run()
analyzer_baseline = SkillMismatchAnalyzer(m_baseline)
metrics_baseline = analyzer_baseline.get_mismatch_metrics()

# 场景2: 高度自动化
m_auto = DigitalizationModel(
    T=500, 
    nr_to_r=True,
    automation_rate=0.02,
    automation_start=100,
    automation_routine_impact=0.8
)
m_auto.run()
analyzer_auto = SkillMismatchAnalyzer(m_auto)
metrics_auto = analyzer_auto.get_mismatch_metrics()

# 比较
print(f"Baseline mismatch: {metrics_baseline['total_mismatch_rate']:.2%}")
print(f"Automation mismatch: {metrics_auto['total_mismatch_rate']:.2%}")
```

### 示例2: 评估再培训政策效果

```python
scenarios = {
    'No Policy': {
        'automation_rate': 0.02,
        'automation_start': 100,
    },
    'Retraining Policy': {
        'automation_rate': 0.02,
        'automation_start': 100,
        'retraining_policy': True,
        'training_effectiveness': 0.3,
    }
}

analyzer = DigitalizationAnalyzer()
results = analyzer.compare_scenarios(scenarios, T=500, n_runs=5)
analyzer.plot_scenario_comparison('retraining_effect.png')
```

## 引用

如使用本代码进行研究，请引用：

```
Labor Market ABM with Skill Mismatch and Digitalization Analysis
GitHub: shuailiushuai/labor_market_abm
```

## 联系方式

问题或建议请提交 GitHub Issue。

## 许可证

见 LICENSE 文件。
