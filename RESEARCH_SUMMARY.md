# 劳动力市场ABM扩展研究内容总结
# Labor Market ABM Extension Research Summary

## 研究扩展内容 / Research Extensions

本项目为劳动力市场ABM添加了全面的分析框架，具体研究内容如下：

This project adds a comprehensive analysis framework to the labor market ABM, with the following specific research content:

---

## 一、劳动力市场规制研究 / Labor Market Regulation Research

### 1.1 最低工资政策 (Minimum Wage Policies)

**研究问题** / **Research Questions**:
- 最低工资对总体失业率的影响
- 最低工资对不同技能群体的差异化影响
- 最低工资与经济产出(GDP)的权衡关系
- 最优最低工资水平的确定

**代码实现** / **Code Implementation**:
```python
from regulation_analysis import RegulationAnalyzer

analyzer = RegulationAnalyzer(T=500, burn_in=200)
results = analyzer.analyze_minimum_wage_policy(
    base_params,
    min_wage_levels=[0.2, 0.3, 0.4, 0.5, 0.6],
    n_replications=5
)
```

**输出指标** / **Output Metrics**:
- 失业率 (unemployment rate)
- GDP水平
- 例行性/非例行性工人失业率差距
- 工资分布和不平等
- 企业违约率

---

### 1.2 解雇成本分析 (Firing Costs Analysis)

**研究问题** / **Research Questions**:
- 解雇成本对劳动力市场流动性的影响
- 就业保护与失业率的关系
- 解雇成本对工资动态的影响

**代码实现** / **Code Implementation**:
```python
results = analyzer.analyze_firing_costs(
    base_params,
    firing_cost_levels=[0.001, 0.003, 0.005, 0.007, 0.01],
    n_replications=5
)
```

**建模说明** / **Modeling Notes**:
- 通过`sigma_w`参数（工资调整成本）建模解雇成本
- 更高的`sigma_w`意味着更高的工资粘性和更高的隐含解雇成本

---

### 1.3 劳动力市场匹配效率 (Labor Market Matching Efficiency)

**研究问题** / **Research Questions**:
- 匹配效率对失业率和职位空缺的影响
- Beveridge曲线的形状和位置
- 劳动力市场摩擦的经济成本

**代码实现** / **Code Implementation**:
```python
results = analyzer.analyze_labor_market_efficiency(
    base_params,
    lambda_LM_values=[1, 3, 5, 7, 10],
    n_replications=5
)
```

**关键参数** / **Key Parameter**:
- `lambda_LM`: 劳动力市场匹配强度
  - 更高的值 → 更高的匹配效率 → 更低的失业率

---

### 1.4 综合政策分析 (Combined Policy Analysis)

**研究问题** / **Research Questions**:
- 不同政策的组合效应
- 政策互补性和替代性
- 最优政策组合设计

**代码实现** / **Code Implementation**:
```python
policy_combinations = [
    {'min_w_par': 0.3, 'lambda_LM': 5, 'sigma_w': 0.005},
    {'min_w_par': 0.4, 'lambda_LM': 7, 'sigma_w': 0.003},
    {'min_w_par': 0.35, 'lambda_LM': 10, 'sigma_w': 0.007},
]

results = analyzer.analyze_combined_policies(
    base_params,
    policy_combinations,
    n_replications=5
)
```

---

## 二、技能失配研究 / Skill Mismatch Research

### 2.1 技能构成影响 (Skill Composition Effects)

**研究问题** / **Research Questions**:
- 例行性与非例行性工人比例对劳动力市场的影响
- 技能结构与失业率的关系
- 技能结构与经济产出的关系
- 最优技能比例

**代码实现** / **Code Implementation**:
```python
results = analyzer.analyze_skill_mismatch(
    base_params,
    skill_ratios=[0.2, 0.25, 0.33, 0.4, 0.5, 0.6],
    n_replications=5
)
```

**关键参数** / **Key Parameter**:
- `gamma_nr`: 非例行性工人比例
  - 0.33 表示33%的工人是非例行性工人

**输出指标** / **Output Metrics**:
- 总体失业率
- 按技能类型的失业率
- 技能失配率 (非例行性工人在例行性岗位工作)
- 工资差距 (技能溢价)

---

### 2.2 技能失配测量 (Skill Mismatch Measurement)

**直接测量指标** / **Direct Metrics**:
- `skill_mismatch_rate`: 非例行性工人从事例行性工作的比例
- `routine_unemployment`: 例行性工人失业率
- `nonroutine_unemployment`: 非例行性工人失业率
- `unemployment_gap`: 失业率差距

**经济成本指标** / **Economic Cost Metrics**:
- GDP损失
- 工资差距扩大
- 工资不平等加剧
- 企业生产效率下降

---

### 2.3 技能失配动态 (Skill Mismatch Dynamics)

**研究问题** / **Research Questions**:
- 技能失配如何随时间演化
- 劳动力市场冲击对技能失配的影响
- 技能失配的持续性

**分析方法** / **Analysis Method**:
```python
# 时间序列分析
model = Model(T=500, gamma_nr=0.4)
model.run()

# 提取技能失配时间序列
mismatch_series = model.share_nr_in_r

# 分析趋势和周期
import numpy as np
mean_mismatch = np.mean(mismatch_series[200:])
mismatch_volatility = np.std(mismatch_series[200:])
```

---

### 2.4 技能失配与政策交互 (Skill Mismatch-Policy Interactions)

**研究问题** / **Research Questions**:
- 最低工资政策对不同技能群体的差异化影响
- 劳动力市场摩擦与技能匹配效率
- 技能培训政策的有效性

**专题分析示例** / **Focused Analysis Example**:
见 `example_focused_research.py` 中的三个研究问题

---

## 三、敏感性分析 / Sensitivity Analysis

### 3.1 参数敏感性 (Parameter Sensitivity)

**研究目的** / **Research Purpose**:
- 识别关键参数
- 理解参数影响机制
- 评估模型稳健性

**方法** / **Methods**:

1. **单因素敏感性分析** (One-at-a-time)
```python
from sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(T=500, burn_in=200)
results = analyzer.one_at_a_time_analysis(
    base_params,
    param_ranges,
    n_samples=10,
    n_replications=5
)
```

2. **蒙特卡洛敏感性** (Monte Carlo)
```python
results = analyzer.monte_carlo_analysis(
    param_ranges,
    n_samples=100
)
correlations = analyzer.calculate_correlation_coefficients()
```

3. **全局敏感性** (Global Sensitivity)
   - 使用拉丁超立方采样确保参数空间覆盖
   - 计算Sobol指数(通过相关系数近似)

---

### 3.2 关键参数识别 (Key Parameter Identification)

**主要参数及其作用** / **Main Parameters and Effects**:

| 参数 | 含义 | 主要影响 |
|------|------|---------|
| `lambda_LM` | 劳动力市场匹配强度 | 失业率、职位空缺 |
| `sigma_w` | 工资调整方差 | 工资动态、解雇成本 |
| `sigma_m` | 标价方差 | 价格波动、企业竞争 |
| `alpha_2` | 财富消费倾向 | 总需求、GDP |
| `chi_C` | 消费匹配参数 | 商品市场清算 |
| `gamma_nr` | 非例行性工人比例 | 技能失配、工资差距 |
| `min_w_par` | 最低工资参数 | 失业率、工资分布 |

---

## 四、经验验证 / Empirical Validation

### 4.1 验证方法 (Validation Methods)

1. **统计矩比较** (Statistical Moments)
   - 均值、方差、偏度、峰度

2. **分布检验** (Distribution Tests)
   - Kolmogorov-Smirnov检验
   - Anderson-Darling检验

3. **关系验证** (Relationship Validation)
   - Beveridge曲线(失业率vs职位空缺)
   - 菲利普斯曲线(失业率vs工资增长)
   - 奥肯定律(失业变化vs GDP增长)

4. **动态验证** (Dynamics Validation)
   - 自相关函数
   - 持续性测量

**代码实现** / **Code Implementation**:
```python
from empirical_validation import EmpiricalValidator

validator = EmpiricalValidator()
validator.set_model_results(model)

# Beveridge曲线验证
bev_results = validator.validate_beveridge_curve(
    emp_unemployment, 
    emp_vacancies
)

# 工资分布验证
wage_results = validator.validate_wage_distribution(
    emp_wages
)

# 生成验证报告
validator.generate_validation_report('validation_report.txt')
```

---

## 五、比较分析 / Comparative Analysis

### 5.1 场景比较 (Scenario Comparison)

**方法** / **Methods**:
```python
from comparative_analysis import ComparativeAnalyzer

analyzer = ComparativeAnalyzer()

# 添加多个场景
analyzer.add_scenario('Baseline', baseline_results)
analyzer.add_scenario('High MinWage', high_minwage_results)
analyzer.add_scenario('High Skill', high_skill_results)

# 比较分析
comparison = analyzer.compare_scenarios(
    ['Baseline', 'High MinWage', 'High Skill']
)

# 统计检验
test = analyzer.statistical_comparison(
    'Baseline', 'High MinWage', 'mean_unemployment'
)
```

### 5.2 可视化方法 (Visualization Methods)

1. **箱线图比较** (Box plots)
2. **雷达图比较** (Radar charts)
3. **时间序列对比** (Time series comparison)

---

## 六、使用示例 / Usage Examples

### 快速开始 (Quick Start)
```bash
python example_quick_start.py
```

### 综合分析 (Comprehensive Analysis)
```bash
python example_comprehensive_analysis.py
```

### 专题研究 (Focused Research)
```bash
python example_focused_research.py
```

---

## 七、研究产出 / Research Outputs

### 报告文件 (Reports)
- `sensitivity_report.txt`: 敏感性分析报告
- `validation_report.txt`: 经验验证报告
- `regulation_analysis_report.txt`: 规制分析报告
- `comparison_report.txt`: 比较分析报告

### 数据文件 (Data Files)
- `.csv` 文件包含详细的模拟结果
- 可用于进一步的统计分析和可视化

### 可视化文件 (Visualizations)
- `.png` 图表展示关键发现
- 包括热图、箱线图、雷达图、时间序列图等

---

## 八、扩展建议 / Extension Recommendations

### 8.1 进一步研究方向

1. **技能培训政策** (Skill Training Policies)
   - 实施工人技能升级机制
   - 评估培训项目效果
   - 最优培训投资

2. **失业保险** (Unemployment Insurance)
   - 添加失业救济金机制
   - 分析激励效应
   - 福利-效率权衡

3. **技术进步** (Technological Progress)
   - 建模自动化和技术变革
   - 例行性工作的替代
   - 长期技能需求演化

4. **异质性分析** (Heterogeneity Analysis)
   - 工人能力差异
   - 企业生产率分布
   - 学习和适应机制

### 8.2 方法改进

1. **校准和估计** (Calibration and Estimation)
   - 使用真实数据校准模型
   - 实施矩匹配估计
   - 贝叶斯参数估计

2. **网络效应** (Network Effects)
   - 工人-企业网络
   - 信息传播
   - 社会网络对就业的影响

3. **空间维度** (Spatial Dimension)
   - 地理位置
   - 通勤成本
   - 区域劳动力市场

---

## 九、技术文档 / Technical Documentation

详细的技术文档和API说明请参考：
- `ANALYSIS_MODULES_README.md` - 完整模块文档
- 各模块源代码中的docstring

---

## 十、引用和致谢 / Citation and Acknowledgments

如使用本框架进行研究，请引用原始模型和本扩展。

If using this framework for research, please cite both the original model and this extension.

---

**最后更新** / **Last Updated**: 2025-10-15
**版本** / **Version**: 1.0
**作者** / **Author**: GitHub Copilot for shuailiushuai/labor_market_abm
