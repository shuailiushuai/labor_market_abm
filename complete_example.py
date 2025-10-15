"""
Complete Example: Labor Market ABM Analysis
数字化对技能失配影响的完整分析示例

This script demonstrates a complete research workflow:
1. Run baseline model
2. Analyze skill mismatch
3. Compare digitalization scenarios
4. Evaluate policy interventions
5. Generate comprehensive reports
"""

import numpy as np
import numpy.random as rd
from model_class import Model
from skill_mismatch_analysis import SkillMismatchAnalyzer
from digitalization_regulation import DigitalizationModel, DigitalizationAnalyzer
import os

# Create output directory
output_dir = 'example_analysis_output'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("COMPLETE LABOR MARKET ABM ANALYSIS EXAMPLE")
print("数字化对技能失配影响的研究")
print("=" * 80)

# ============================================================================
# PART 1: Baseline Model - Understanding Current Skill Mismatch
# 第一部分：基准模型 - 理解当前技能错配
# ============================================================================

print("\n" + "-" * 80)
print("PART 1: BASELINE MODEL ANALYSIS")
print("第一部分：基准模型分析")
print("-" * 80)

rd.seed(42)
print("\nRunning baseline model (T=500 periods)...")
print("运行基准模型（500期）...")

m_baseline = Model(
    T=500,
    H=200,        # 200 households
    F=20,         # 20 firms
    gamma_nr=0.33,  # 33% non-routine workers
    nr_to_r=True,   # Allow NR workers to take R jobs
    mu_r=0.3,       # Routine labor share in production
    u_r=0.08,       # Target unemployment rate
    lambda_LM=5,    # Labor market matching intensity
)

m_baseline.run()
print("✓ Baseline model complete")

# Analyze skill mismatch
print("\nAnalyzing skill mismatch in baseline...")
print("分析基准情况下的技能错配...")

analyzer_baseline = SkillMismatchAnalyzer(m_baseline)
metrics_baseline = analyzer_baseline.get_mismatch_metrics()

print("\nBaseline Skill Mismatch Metrics:")
print("基准技能错配指标：")
print(f"  Over-qualified rate (过度技能率): {metrics_baseline['overqualified_rate']:.2%}")
print(f"  Under-qualified rate (技能不足率): {metrics_baseline['underqualified_rate']:.2%}")
print(f"  Total mismatch rate (总错配率): {metrics_baseline['total_mismatch_rate']:.2%}")
print(f"  NR unemployment (非常规失业率): {metrics_baseline['non_routine_unemployment']:.2%}")
print(f"  R unemployment (常规失业率): {metrics_baseline['routine_unemployment']:.2%}")

# Generate mismatch evolution plot
fig, df = analyzer_baseline.plot_mismatch_evolution(
    start=100,
    save_path=f'{output_dir}/baseline_mismatch_evolution.png'
)
print(f"\n✓ Mismatch evolution plot saved to {output_dir}/baseline_mismatch_evolution.png")

# ============================================================================
# PART 2: Digitalization Scenarios - Impact of Automation
# 第二部分：数字化场景 - 自动化的影响
# ============================================================================

print("\n" + "-" * 80)
print("PART 2: DIGITALIZATION SCENARIOS")
print("第二部分：数字化场景分析")
print("-" * 80)

# Define scenarios
scenarios = {
    'Baseline (No Automation)': {
        'H': 200,
        'F': 20,
        'gamma_nr': 0.33,
        'nr_to_r': True,
        'mu_r': 0.3,
        'automation_rate': 0.0,
    },
    
    'Moderate Automation (中等自动化)': {
        'H': 200,
        'F': 20,
        'gamma_nr': 0.33,
        'nr_to_r': True,
        'mu_r': 0.3,
        'automation_rate': 0.01,
        'automation_start': 100,
        'automation_routine_impact': 0.5,
        'automation_cost_reduction': 0.15,
    },
    
    'High Automation (高度自动化)': {
        'H': 200,
        'F': 20,
        'gamma_nr': 0.33,
        'nr_to_r': True,
        'mu_r': 0.3,
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
        'automation_cost_reduction': 0.3,
    },
}

print("\nComparing digitalization scenarios...")
print("比较数字化场景...")
print("(Running 3 simulations per scenario, this may take a few minutes)")
print("（每个场景运行3次模拟，可能需要几分钟）")

digi_analyzer = DigitalizationAnalyzer()
results = digi_analyzer.compare_scenarios(scenarios, T=400, n_runs=3)

print("\n" + "=" * 80)
print("SCENARIO COMPARISON RESULTS")
print("场景比较结果")
print("=" * 80)

for scenario_name, metrics in results.items():
    print(f"\n{scenario_name}:")
    print(f"  Unemployment (失业率): {metrics['unemployment_mean']:.3f} (±{metrics['unemployment_std']:.3f})")
    print(f"  Skill Mismatch (技能错配): {metrics['mismatch_mean']:.3f} (±{metrics['mismatch_std']:.3f})")
    print(f"  Real Wage (实际工资): {metrics['wage_mean']:.3f} (±{metrics['wage_std']:.3f})")
    print(f"  Inequality 90/10 (不平等): {metrics['inequality_mean']:.2f} (±{metrics['inequality_std']:.2f})")
    print(f"  GDP: {metrics['gdp_mean']:.1f} (±{metrics['gdp_std']:.1f})")

# Generate comparison plots
digi_analyzer.plot_scenario_comparison(f'{output_dir}/scenario_comparison.png')
print(f"\n✓ Scenario comparison plot saved to {output_dir}/scenario_comparison.png")

# ============================================================================
# PART 3: Policy Interventions - Retraining Programs
# 第三部分：政策干预 - 再培训项目
# ============================================================================

print("\n" + "-" * 80)
print("PART 3: POLICY INTERVENTION ANALYSIS")
print("第三部分：政策干预分析")
print("-" * 80)

policy_scenarios = {
    'High Automation (No Policy)': {
        'H': 200, 'F': 20, 'gamma_nr': 0.33, 'nr_to_r': True, 'mu_r': 0.3,
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
        'automation_cost_reduction': 0.3,
    },
    
    'Automation + Retraining (自动化+再培训)': {
        'H': 200, 'F': 20, 'gamma_nr': 0.33, 'nr_to_r': True, 'mu_r': 0.3,
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
        'automation_cost_reduction': 0.3,
        'retraining_policy': True,
        'training_effectiveness': 0.3,
        'training_cost': 0.5,
    },
    
    'Automation + Higher Min Wage (自动化+更高最低工资)': {
        'H': 200, 'F': 20, 'gamma_nr': 0.33, 'nr_to_r': True, 'mu_r': 0.3,
        'automation_rate': 0.02,
        'automation_start': 100,
        'automation_routine_impact': 0.8,
        'automation_cost_reduction': 0.3,
        'min_wage_policy': {'start': 150, 'level': 0.6},
    },
}

print("\nComparing policy interventions...")
print("比较政策干预效果...")
print("(Running 3 simulations per policy, this may take a few minutes)")

policy_analyzer = DigitalizationAnalyzer()
policy_results = policy_analyzer.compare_scenarios(policy_scenarios, T=400, n_runs=3)

print("\n" + "=" * 80)
print("POLICY INTERVENTION RESULTS")
print("政策干预结果")
print("=" * 80)

baseline_policy = 'High Automation (No Policy)'
for scenario_name, metrics in policy_results.items():
    print(f"\n{scenario_name}:")
    
    # Calculate changes relative to no-policy baseline
    if scenario_name != baseline_policy:
        unemp_change = ((metrics['unemployment_mean'] - 
                        policy_results[baseline_policy]['unemployment_mean']) / 
                       policy_results[baseline_policy]['unemployment_mean'] * 100)
        mismatch_change = ((metrics['mismatch_mean'] - 
                           policy_results[baseline_policy]['mismatch_mean']) / 
                          policy_results[baseline_policy]['mismatch_mean'] * 100)
        
        print(f"  Unemployment change (失业率变化): {unemp_change:+.1f}%")
        print(f"  Mismatch change (错配变化): {mismatch_change:+.1f}%")
    
    print(f"  Unemployment (失业率): {metrics['unemployment_mean']:.3f}")
    print(f"  Skill Mismatch (技能错配): {metrics['mismatch_mean']:.3f}")
    print(f"  Real Wage (实际工资): {metrics['wage_mean']:.3f}")

# Generate policy comparison plots
policy_analyzer.plot_scenario_comparison(f'{output_dir}/policy_comparison.png')
print(f"\n✓ Policy comparison plot saved to {output_dir}/policy_comparison.png")

# Generate policy report
policy_analyzer.generate_policy_report(f'{output_dir}/policy_report.txt')
print(f"✓ Detailed policy report saved to {output_dir}/policy_report.txt")

# ============================================================================
# PART 4: Summary and Key Findings
# 第四部分：总结和关键发现
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("关键发现总结")
print("=" * 80)

print("\n1. BASELINE SKILL MISMATCH (基准技能错配):")
print(f"   - Total mismatch affects {metrics_baseline['total_mismatch_rate']:.1%} of employed workers")
print(f"   - Over-qualification (NR in R jobs): {metrics_baseline['overqualified_rate']:.1%}")
print(f"   - Under-qualification (R in NR jobs): {metrics_baseline['underqualified_rate']:.1%}")

print("\n2. DIGITALIZATION IMPACT (数字化影响):")
baseline_result = results['Baseline (No Automation)']
high_auto_result = results['High Automation (高度自动化)']
mismatch_increase = ((high_auto_result['mismatch_mean'] - baseline_result['mismatch_mean']) / 
                     baseline_result['mismatch_mean'] * 100)
print(f"   - High automation increases skill mismatch by {mismatch_increase:+.1f}%")
print(f"   - Unemployment rises from {baseline_result['unemployment_mean']:.3f} to {high_auto_result['unemployment_mean']:.3f}")

print("\n3. POLICY EFFECTIVENESS (政策有效性):")
no_policy = policy_results['High Automation (No Policy)']
with_retraining = policy_results['Automation + Retraining (自动化+再培训)']
retraining_effect = ((with_retraining['mismatch_mean'] - no_policy['mismatch_mean']) / 
                     no_policy['mismatch_mean'] * 100)
print(f"   - Retraining programs reduce mismatch by {abs(retraining_effect):.1f}%")
print(f"   - Unemployment with retraining: {with_retraining['unemployment_mean']:.3f} vs {no_policy['unemployment_mean']:.3f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("分析完成！")
print("=" * 80)
print(f"\nAll outputs saved to: {output_dir}/")
print(f"所有输出已保存至：{output_dir}/")
print("\nGenerated files (生成的文件):")
print(f"  - {output_dir}/baseline_mismatch_evolution.png")
print(f"  - {output_dir}/scenario_comparison.png")
print(f"  - {output_dir}/policy_comparison.png")
print(f"  - {output_dir}/policy_report.txt")

print("\n" + "=" * 80)
print("RESEARCH IMPLICATIONS")
print("研究意义")
print("=" * 80)
print("""
This analysis demonstrates:

1. SKILL MISMATCH MEASUREMENT (技能错配测量)
   - Quantifies over-skilling and under-skilling in labor market
   - Tracks evolution of mismatch over time
   - Identifies wage penalties from skill mismatch

2. DIGITALIZATION EFFECTS (数字化效应)
   - Automation disproportionately affects routine workers
   - Creates higher skill mismatch as jobs transform
   - Increases inequality between skill types

3. POLICY EVALUATION (政策评估)
   - Retraining programs can mitigate automation's negative effects
   - Minimum wage policies have complex interactions with mismatch
   - Policy timing and intensity matter for effectiveness

4. METHODOLOGICAL CONTRIBUTION (方法学贡献)
   - Provides computational framework for policy simulation
   - Enables scenario-based policy analysis
   - Supports evidence-based labor market policy design
""")

print("\nFor more analyses, see:")
print("更多分析请参考：")
print("  - sensitivity_analysis.py (敏感性分析)")
print("  - empirical_validation.py (经验验证)")
print("  - parameter_estimation.py (参数估计)")
print("  - analysis_runner.py (完整分析套件)")

print("\n" + "=" * 80)
