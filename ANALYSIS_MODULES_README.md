# Labor Market ABM Analysis Extensions

## 概述 / Overview

本项目为劳动力市场基于主体建模(Agent-Based Model, ABM)添加了全面的分析模块，包括敏感性分析、经验验证、劳动力市场规制分析以及技能失配研究。

This project extends the labor market agent-based model with comprehensive analysis modules including sensitivity analysis, empirical validation, labor market regulation analysis, and skill mismatch studies.

## 新增模块 / New Modules

### 1. 敏感性分析 (Sensitivity Analysis)

**文件**: `sensitivity_analysis.py`

**功能** / **Features**:
- 单因素敏感性分析 (One-at-a-time analysis)
- 蒙特卡洛敏感性分析与拉丁超立方采样 (Monte Carlo with Latin Hypercube Sampling)
- Sobol指数全局敏感性分析 (Global sensitivity with Sobol indices)
- 参数-指标相关性分析 (Parameter-metric correlation analysis)

**主要类** / **Main Class**: `SensitivityAnalyzer`

**使用示例** / **Usage Example**:
```python
from sensitivity_analysis import SensitivityAnalyzer

# 初始化分析器
analyzer = SensitivityAnalyzer(T=500, burn_in=200)

# 定义参数范围
param_ranges = {
    'lambda_LM': (1, 10),      # 劳动力市场匹配强度
    'sigma_m': (0.001, 0.01),   # 标价方差
    'sigma_w': (0.001, 0.01),   # 工资调整方差
    'alpha_2': (0.1, 0.4),      # 财富消费倾向
}

# 运行蒙特卡洛分析
results = analyzer.monte_carlo_analysis(param_ranges, n_samples=100)

# 生成相关性热图
analyzer.plot_correlation_heatmap(save_path='sensitivity_results.png')

# 生成报告
analyzer.generate_sensitivity_report('sensitivity_report.txt')
```

**关键输出** / **Key Outputs**:
- 参数影响力排序 (Parameter influence ranking)
- 相关系数矩阵 (Correlation coefficient matrix)
- 敏感性可视化图表 (Sensitivity visualization plots)

---

### 2. 经验验证 (Empirical Validation)

**文件**: `empirical_validation.py`

**功能** / **Features**:
- 统计矩比较(均值、方差、偏度、峰度) (Statistical moment comparison)
- Beveridge曲线验证 (Beveridge curve validation)
- 工资分布验证(KS检验) (Wage distribution validation with KS test)
- 失业动态验证 (Unemployment dynamics validation)
- 菲利普斯曲线验证 (Phillips curve validation)

**主要类** / **Main Class**: `EmpiricalValidator`

**使用示例** / **Usage Example**:
```python
from empirical_validation import EmpiricalValidator
from model_class import Model

# 运行模型
model = Model(T=500, H=500, F=80)
model.run()

# 初始化验证器
validator = EmpiricalValidator()
validator.set_model_results(model)

# 加载经验数据(示例)
emp_unemployment = load_empirical_unemployment_data()
emp_vacancies = load_empirical_vacancy_data()

# 验证Beveridge曲线
results = validator.validate_beveridge_curve(emp_unemployment, emp_vacancies)

# 生成验证报告
validator.generate_validation_report('validation_report.txt')
```

**关键验证指标** / **Key Validation Metrics**:
- KS统计量和p值 (KS statistic and p-value)
- 相关系数比较 (Correlation comparison)
- 自相关函数 (Autocorrelation functions)
- 矩匹配度 (Moment matching)

---

### 3. 劳动力市场规制与技能失配分析 (Regulation & Skill Mismatch Analysis)

**文件**: `regulation_analysis.py`

**功能** / **Features**:

#### 3.1 劳动力市场规制分析
- **最低工资政策** (Minimum wage policies)
  - 不同最低工资水平的影响分析
  - 对就业、产出和工资分配的影响
  
- **解雇成本分析** (Firing costs analysis)
  - 通过工资粘性建模解雇成本
  - 对劳动力市场流动性的影响
  
- **劳动力市场效率** (Labor market efficiency)
  - 匹配效率参数分析
  - 职位空缺与失业率关系

#### 3.2 技能失配分析
- **技能构成影响** (Skill composition effects)
  - 例行性与非例行性工人比例
  - 技能失配率测量
  
- **失业差异** (Unemployment differentials)
  - 按技能类型的失业率
  - 工资差距分析
  
- **技能需求匹配** (Skill-demand matching)
  - 非例行性工人从事例行性工作比例
  - 技能错配的经济影响

**主要类** / **Main Class**: `RegulationAnalyzer`

**使用示例** / **Usage Example**:
```python
from regulation_analysis import RegulationAnalyzer

# 初始化分析器
analyzer = RegulationAnalyzer(T=500, burn_in=200)

# 基准参数
base_params = {
    'H': 500, 'F': 80, 'alpha_2': 0.25,
    'lambda_LM': 5, 'sigma_w': 0.005,
    'gamma_nr': 0.33, 'min_w_par': 0.3,
}

# 1. 最低工资政策分析
min_wage_results = analyzer.analyze_minimum_wage_policy(
    base_params,
    min_wage_levels=[0.2, 0.3, 0.4, 0.5, 0.6],
    n_replications=5
)

# 2. 技能失配分析
skill_results = analyzer.analyze_skill_mismatch(
    base_params,
    skill_ratios=[0.2, 0.33, 0.4, 0.5, 0.6],
    n_replications=5
)

# 3. 解雇成本分析
firing_results = analyzer.analyze_firing_costs(
    base_params,
    firing_cost_levels=[0.001, 0.005, 0.01],
    n_replications=5
)

# 生成可视化
analyzer.plot_multi_metric_comparison(
    'minimum_wage', 'min_wage_par',
    save_path='minimum_wage_effects.png'
)

# 生成综合报告
analyzer.generate_policy_report('regulation_report.txt')
```

**研究内容与代码模块对应** / **Research Topics and Code Modules**:

| 研究内容 | 代码模块/方法 | 输出指标 |
|---------|--------------|---------|
| 最低工资影响 | `analyze_minimum_wage_policy()` | 失业率、GDP、工资分布、不平等 |
| 解雇成本效应 | `analyze_firing_costs()` | 劳动力流动性、失业持续期 |
| 技能失配程度 | `analyze_skill_mismatch()` | 技能错配率、按技能失业率 |
| 劳动力市场摩擦 | `analyze_labor_market_efficiency()` | 匹配效率、职位空缺 |
| 综合政策效应 | `analyze_combined_policies()` | 多维度政策影响 |

---

### 4. 比较分析 (Comparative Analysis)

**文件**: `comparative_analysis.py`

**功能** / **Features**:
- 多场景对比 (Multi-scenario comparison)
- 统计显著性检验 (Statistical significance tests)
- 场景排序 (Scenario ranking)
- 雷达图对比 (Radar chart comparison)
- 反事实分析 (Counterfactual analysis)

**主要类** / **Main Class**: `ComparativeAnalyzer`

**使用示例** / **Usage Example**:
```python
from comparative_analysis import ComparativeAnalyzer

# 初始化
analyzer = ComparativeAnalyzer()

# 添加不同场景的结果
analyzer.add_scenario('Baseline', baseline_results, 
                     "标准参数配置")
analyzer.add_scenario('High Min Wage', high_minwage_results,
                     "高最低工资")
analyzer.add_scenario('High Skill', high_skill_results,
                     "高技能比例")

# 比较场景
comparison = analyzer.compare_scenarios(
    ['Baseline', 'High Min Wage', 'High Skill']
)

# 统计检验
test_result = analyzer.statistical_comparison(
    'Baseline', 'High Min Wage', 'mean_unemployment'
)

# 生成可视化
analyzer.plot_scenario_comparison(
    ['Baseline', 'High Min Wage', 'High Skill'],
    ['mean_unemployment', 'mean_GDP', 'wage_inequality_9_1']
)

analyzer.plot_radar_comparison(
    ['Baseline', 'High Min Wage', 'High Skill'],
    ['mean_unemployment', 'mean_GDP', 'mean_real_wage']
)

# 生成报告
analyzer.generate_comparison_report('comparison_report.txt')
```

---

## 完整示例 / Complete Example

**文件**: `example_comprehensive_analysis.py`

这是一个完整的示例脚本，展示如何使用所有新模块进行综合分析：

This is a complete example script demonstrating how to use all new modules for comprehensive analysis:

```bash
python example_comprehensive_analysis.py
```

该脚本执行以下分析 / The script performs:
1. 敏感性分析 (Sensitivity analysis)
2. 经验验证 (Empirical validation)
3. 劳动力市场规制分析 (Labor market regulation analysis)
4. 技能失配分析 (Skill mismatch analysis)
5. 比较分析 (Comparative analysis)

---

## 研究扩展建议 / Research Extension Recommendations

### 劳动力市场规制研究方向 / Labor Market Regulation Research Directions:

1. **最低工资政策** (Minimum Wage Policies)
   - 最优最低工资水平
   - 区域差异化最低工资
   - 动态调整机制

2. **就业保护政策** (Employment Protection)
   - 解雇成本对劳动力市场流动性的影响
   - 临时与永久合同的权衡
   - 试用期制度设计

3. **技能培训政策** (Skill Training Policies)
   - 再培训项目效果评估
   - 技能升级激励机制
   - 在职培训补贴

4. **失业保险** (Unemployment Insurance)
   - 失业救济金水平
   - 失业救济期限
   - 工作搜寻激励

### 技能失配研究方向 / Skill Mismatch Research Directions:

1. **技能供需匹配** (Skill Supply-Demand Matching)
   - 技能结构演化
   - 技术进步与技能需求
   - 教育系统与劳动力市场对接

2. **技能错配测量** (Skill Mismatch Measurement)
   - 过度教育与不足教育
   - 技能错配的持续性
   - 技能错配的成本

3. **劳动力市场分割** (Labor Market Segmentation)
   - 一级与二级劳动力市场
   - 技能型与非技能型工作分离
   - 流动性壁垒

---

## 输出文件说明 / Output Files Description

### 报告文件 (Reports)
- `sensitivity_report.txt`: 敏感性分析详细报告
- `validation_report.txt`: 经验验证报告
- `regulation_analysis_report.txt`: 规制分析综合报告
- `comparison_report.txt`: 比较分析报告

### 可视化文件 (Visualizations)
- `sensitivity_correlations.png`: 参数-指标相关性热图
- `validation_beveridge.png`: Beveridge曲线验证
- `validation_wages.png`: 工资分布验证
- `minimum_wage_effects.png`: 最低工资影响
- `skill_mismatch_effects.png`: 技能失配影响
- `scenario_comparison.png`: 场景对比箱线图
- `radar_comparison.png`: 场景对比雷达图

---

## 依赖包 / Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
```

安装依赖 / Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy
```

---

## 快速开始 / Quick Start

1. **运行敏感性分析** / **Run Sensitivity Analysis**:
```bash
python sensitivity_analysis.py
```

2. **运行经验验证** / **Run Empirical Validation**:
```bash
python empirical_validation.py
```

3. **运行规制分析** / **Run Regulation Analysis**:
```bash
python regulation_analysis.py
```

4. **运行完整分析** / **Run Complete Analysis**:
```bash
python example_comprehensive_analysis.py
```

---

## 引用 / Citation

如果您使用这些模块进行研究，请引用：

If you use these modules for research, please cite:

```
Labor Market ABM Analysis Extensions
Author: [Your Name]
Year: 2025
GitHub: https://github.com/shuailiushuai/labor_market_abm
```

---

## 联系方式 / Contact

如有问题或建议，请通过GitHub Issues联系。

For questions or suggestions, please contact via GitHub Issues.

---

## 许可证 / License

本项目遵循原项目的许可证。

This project follows the license of the original project.
