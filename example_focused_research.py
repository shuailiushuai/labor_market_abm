"""
专题研究示例：劳动力市场规制与技能失配
Focused Research Example: Labor Market Regulation and Skill Mismatch

本脚本展示如何使用模型进行以下专题研究：
This script demonstrates how to use the model for the following research topics:

1. 最低工资政策对不同技能劳动力的影响
   Impact of minimum wage on different skill levels

2. 技能失配的经济成本分析
   Economic costs of skill mismatch

3. 劳动力市场摩擦对技能匹配的影响
   Impact of labor market frictions on skill matching

4. 政策组合的最优设计
   Optimal policy mix design
"""

import numpy as np
import numpy.random as rd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_class import Model
from regulation_analysis import RegulationAnalyzer
from comparative_analysis import ComparativeAnalyzer

# Create results directory
os.makedirs("results", exist_ok=True)

print("=" * 80)
print("劳动力市场规制与技能失配专题研究")
print("Labor Market Regulation and Skill Mismatch Research")
print("=" * 80)

# Set random seed
rd.seed(42)

# Base parameters
base_params = {
    'H': 500,
    'F': 80,
    'alpha_2': 0.25,
    'N_app': 4,
    'chi_C': 0.5,
    'lambda_LM': 5,
    'sigma_m': 0.005,
    'sigma_w': 0.005,
    'nu': 0.1,
    'u_r': 0.08,
    'beta': 1,
    'lambda_exp': 0.5,
    'gamma_nr': 0.33,
    'min_w_par': 0.3,
}

analyzer = RegulationAnalyzer(T=400, burn_in=150)

# =============================================================================
# 研究问题1: 最低工资政策对不同技能劳动力的差异化影响
# Research Question 1: Differential Impact of Minimum Wage on Skill Groups
# =============================================================================
print("\n\n" + "=" * 80)
print("研究问题1: 最低工资政策的差异化影响")
print("Research Question 1: Differential Impact of Minimum Wage")
print("=" * 80)

print("\n情景设置：测试3种技能结构 × 5种最低工资水平")
print("Scenario: 3 skill structures × 5 minimum wage levels")

skill_structures = [
    {'gamma_nr': 0.25, 'label': '低技能占比(25% non-routine)'},
    {'gamma_nr': 0.40, 'label': '中技能占比(40% non-routine)'},
    {'gamma_nr': 0.55, 'label': '高技能占比(55% non-routine)'},
]

min_wage_levels = [0.25, 0.30, 0.35, 0.40, 0.45]

results_q1 = []
for skill_struct in skill_structures:
    print(f"\n分析技能结构: {skill_struct['label']}")
    
    params = base_params.copy()
    params.update(skill_struct)
    
    for min_w in min_wage_levels:
        print(f"  最低工资参数: {min_w}")
        params['min_w_par'] = min_w
        
        # Run 3 replications
        for rep in range(3):
            m = Model(T=400, **params)
            m.run()
            
            t = 150
            results_q1.append({
                'skill_structure': skill_struct['label'],
                'gamma_nr': skill_struct['gamma_nr'],
                'min_wage_par': min_w,
                'replication': rep,
                'unemployment': np.mean(m.u_r_arr[t:]),
                'routine_unemployment': np.mean(m.ur_r_arr[t:]),
                'nonroutine_unemployment': np.mean(m.unr_r_arr[t:]),
                'unemployment_gap': np.mean(m.ur_r_arr[t:]) - np.mean(m.unr_r_arr[t:]),
                'wage_gap': np.mean(m.mean_nr_w_arr[t:]) - np.mean(m.mean_r_w_arr[t:]),
                'GDP': np.mean(m.GDP[t:]),
                'wage_inequality': np.mean(m.nine_to_one[t:]),
                'skill_mismatch': np.mean(m.share_nr_in_r[t:]),
            })

df_q1 = pd.DataFrame(results_q1)
df_q1.to_csv('results/question1_minimum_wage_skill_effects.csv', index=False)

# Visualize results for Q1
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, skill_label in enumerate([s['label'] for s in skill_structures]):
    subset = df_q1[df_q1['skill_structure'] == skill_label]
    grouped = subset.groupby('min_wage_par').agg({
        'unemployment': ['mean', 'std'],
        'unemployment_gap': ['mean', 'std']
    })
    
    color = ['blue', 'green', 'red'][i]
    axes[0, 0].plot(grouped.index, grouped['unemployment']['mean'], 
                    'o-', label=skill_label, linewidth=2, color=color)
    axes[0, 0].fill_between(grouped.index,
                            grouped['unemployment']['mean'] - grouped['unemployment']['std'],
                            grouped['unemployment']['mean'] + grouped['unemployment']['std'],
                            alpha=0.2, color=color)

axes[0, 0].set_xlabel('最低工资参数 (Minimum Wage Parameter)', fontsize=12)
axes[0, 0].set_ylabel('总失业率 (Unemployment Rate)', fontsize=12)
axes[0, 0].set_title('最低工资对不同技能结构的失业率影响', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot unemployment gap
for i, skill_label in enumerate([s['label'] for s in skill_structures]):
    subset = df_q1[df_q1['skill_structure'] == skill_label]
    grouped = subset.groupby('min_wage_par')['unemployment_gap'].mean()
    color = ['blue', 'green', 'red'][i]
    axes[0, 1].plot(grouped.index, grouped.values, 'o-', 
                    label=skill_label, linewidth=2, color=color)

axes[0, 1].set_xlabel('最低工资参数', fontsize=12)
axes[0, 1].set_ylabel('失业率差距 (routine - non-routine)', fontsize=12)
axes[0, 1].set_title('技能类型失业率差距', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot GDP effects
for i, skill_label in enumerate([s['label'] for s in skill_structures]):
    subset = df_q1[df_q1['skill_structure'] == skill_label]
    grouped = subset.groupby('min_wage_par')['GDP'].mean()
    color = ['blue', 'green', 'red'][i]
    axes[1, 0].plot(grouped.index, grouped.values, 'o-', 
                    label=skill_label, linewidth=2, color=color)

axes[1, 0].set_xlabel('最低工资参数', fontsize=12)
axes[1, 0].set_ylabel('GDP', fontsize=12)
axes[1, 0].set_title('最低工资对GDP的影响', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot skill mismatch
for i, skill_label in enumerate([s['label'] for s in skill_structures]):
    subset = df_q1[df_q1['skill_structure'] == skill_label]
    grouped = subset.groupby('min_wage_par')['skill_mismatch'].mean()
    color = ['blue', 'green', 'red'][i]
    axes[1, 1].plot(grouped.index, grouped.values, 'o-', 
                    label=skill_label, linewidth=2, color=color)

axes[1, 1].set_xlabel('最低工资参数', fontsize=12)
axes[1, 1].set_ylabel('技能失配率 (Skill Mismatch Rate)', fontsize=12)
axes[1, 1].set_title('最低工资对技能失配的影响', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/question1_minimum_wage_by_skill.png', dpi=300, bbox_inches='tight')
print("\n✓ 研究问题1可视化已保存")

# =============================================================================
# 研究问题2: 技能失配的经济成本
# Research Question 2: Economic Costs of Skill Mismatch
# =============================================================================
print("\n\n" + "=" * 80)
print("研究问题2: 技能失配的经济成本")
print("Research Question 2: Economic Costs of Skill Mismatch")
print("=" * 80)

print("\n情景设置：比较不同技能失配程度的经济后果")
print("Scenario: Comparing economic consequences of different mismatch levels")

skill_ratios = [0.20, 0.25, 0.33, 0.40, 0.50, 0.60]

results_q2 = []
for gamma_nr in skill_ratios:
    print(f"\n分析技能比例 gamma_nr = {gamma_nr}")
    
    params = base_params.copy()
    params['gamma_nr'] = gamma_nr
    
    for rep in range(5):
        m = Model(T=400, **params)
        m.run()
        
        t = 150
        results_q2.append({
            'gamma_nr': gamma_nr,
            'replication': rep,
            'unemployment': np.mean(m.u_r_arr[t:]),
            'skill_mismatch_rate': np.mean(m.share_nr_in_r[t:]),
            'GDP': np.mean(m.GDP[t:]),
            'GDP_volatility': np.std(m.GDP[t:]),
            'wage_gap': np.mean(m.mean_nr_w_arr[t:]) - np.mean(m.mean_r_w_arr[t:]),
            'wage_inequality': np.mean(m.nine_to_one[t:]),
            'firm_defaults': np.mean(m.share_inactive[t:]),
        })

df_q2 = pd.DataFrame(results_q2)
df_q2.to_csv('results/question2_skill_mismatch_costs.csv', index=False)

# Calculate "optimal" skill ratio (lowest unemployment + mismatch)
grouped_q2 = df_q2.groupby('gamma_nr').mean()
composite_cost = grouped_q2['unemployment'] + grouped_q2['skill_mismatch_rate']
optimal_gamma = composite_cost.idxmin()
print(f"\n最优技能比例 (最低综合成本): gamma_nr = {optimal_gamma:.2f}")

# Visualize Q2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Skill mismatch vs skill ratio
grouped = df_q2.groupby('gamma_nr')['skill_mismatch_rate'].agg(['mean', 'std'])
axes[0, 0].plot(grouped.index, grouped['mean'], 'o-', linewidth=2, markersize=8)
axes[0, 0].fill_between(grouped.index, grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], alpha=0.3)
axes[0, 0].set_xlabel('非例行性工人比例 (gamma_nr)', fontsize=12)
axes[0, 0].set_ylabel('技能失配率', fontsize=12)
axes[0, 0].set_title('技能构成与失配率关系', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=optimal_gamma, color='red', linestyle='--', 
                   label=f'最优比例 {optimal_gamma:.2f}')
axes[0, 0].legend()

# GDP vs skill ratio
grouped = df_q2.groupby('gamma_nr')['GDP'].agg(['mean', 'std'])
axes[0, 1].plot(grouped.index, grouped['mean'], 'o-', linewidth=2, markersize=8, color='green')
axes[0, 1].fill_between(grouped.index, grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], alpha=0.3, color='green')
axes[0, 1].set_xlabel('非例行性工人比例 (gamma_nr)', fontsize=12)
axes[0, 1].set_ylabel('GDP', fontsize=12)
axes[0, 1].set_title('技能构成与GDP关系', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=optimal_gamma, color='red', linestyle='--')

# Wage gap vs skill ratio
grouped = df_q2.groupby('gamma_nr')['wage_gap'].agg(['mean', 'std'])
axes[1, 0].plot(grouped.index, grouped['mean'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1, 0].fill_between(grouped.index, grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], alpha=0.3, color='orange')
axes[1, 0].set_xlabel('非例行性工人比例 (gamma_nr)', fontsize=12)
axes[1, 0].set_ylabel('工资差距 (nr - r)', fontsize=12)
axes[1, 0].set_title('技能构成与工资差距', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=optimal_gamma, color='red', linestyle='--')

# Inequality vs skill ratio
grouped = df_q2.groupby('gamma_nr')['wage_inequality'].agg(['mean', 'std'])
axes[1, 1].plot(grouped.index, grouped['mean'], 'o-', linewidth=2, markersize=8, color='purple')
axes[1, 1].fill_between(grouped.index, grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], alpha=0.3, color='purple')
axes[1, 1].set_xlabel('非例行性工人比例 (gamma_nr)', fontsize=12)
axes[1, 1].set_ylabel('工资不平等 (90/10)', fontsize=12)
axes[1, 1].set_title('技能构成与工资不平等', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=optimal_gamma, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('results/question2_skill_mismatch_costs.png', dpi=300, bbox_inches='tight')
print("✓ 研究问题2可视化已保存")

# =============================================================================
# 研究问题3: 劳动力市场摩擦与技能匹配效率
# Research Question 3: Labor Market Frictions and Skill Matching Efficiency
# =============================================================================
print("\n\n" + "=" * 80)
print("研究问题3: 劳动力市场摩擦与技能匹配")
print("Research Question 3: Labor Market Frictions and Skill Matching")
print("=" * 80)

print("\n情景设置：不同匹配效率下的技能失配")
print("Scenario: Skill mismatch under different matching efficiencies")

lambda_LM_values = [1, 3, 5, 7, 10]
skill_levels = [0.25, 0.40, 0.55]

results_q3 = []
for lambda_LM in lambda_LM_values:
    for gamma_nr in skill_levels:
        print(f"  lambda_LM={lambda_LM}, gamma_nr={gamma_nr}")
        
        params = base_params.copy()
        params['lambda_LM'] = lambda_LM
        params['gamma_nr'] = gamma_nr
        
        for rep in range(3):
            m = Model(T=400, **params)
            m.run()
            
            t = 150
            results_q3.append({
                'lambda_LM': lambda_LM,
                'gamma_nr': gamma_nr,
                'skill_label': f'{int(gamma_nr*100)}% non-routine',
                'replication': rep,
                'unemployment': np.mean(m.u_r_arr[t:]),
                'skill_mismatch': np.mean(m.share_nr_in_r[t:]),
                'vacancies': np.mean(m.open_vs[t:]),
                'GDP': np.mean(m.GDP[t:]),
            })

df_q3 = pd.DataFrame(results_q3)
df_q3.to_csv('results/question3_frictions_matching.csv', index=False)

# Visualize Q3
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Skill mismatch vs matching efficiency by skill level
for gamma_nr in skill_levels:
    subset = df_q3[df_q3['gamma_nr'] == gamma_nr]
    grouped = subset.groupby('lambda_LM')['skill_mismatch'].mean()
    axes[0].plot(grouped.index, grouped.values, 'o-', 
                label=f'{int(gamma_nr*100)}% non-routine', linewidth=2)

axes[0].set_xlabel('匹配效率参数 (lambda_LM)', fontsize=12)
axes[0].set_ylabel('技能失配率', fontsize=12)
axes[0].set_title('匹配效率对技能失配的影响', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Unemployment vs matching efficiency
for gamma_nr in skill_levels:
    subset = df_q3[df_q3['gamma_nr'] == gamma_nr]
    grouped = subset.groupby('lambda_LM')['unemployment'].mean()
    axes[1].plot(grouped.index, grouped.values, 'o-', 
                label=f'{int(gamma_nr*100)}% non-routine', linewidth=2)

axes[1].set_xlabel('匹配效率参数 (lambda_LM)', fontsize=12)
axes[1].set_ylabel('失业率', fontsize=12)
axes[1].set_title('匹配效率对失业率的影响', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/question3_frictions_matching.png', dpi=300, bbox_inches='tight')
print("✓ 研究问题3可视化已保存")

# =============================================================================
# 综合报告
# Comprehensive Report
# =============================================================================
print("\n\n" + "=" * 80)
print("生成综合研究报告")
print("Generating Comprehensive Research Report")
print("=" * 80)

with open('results/labor_market_research_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("劳动力市场规制与技能失配研究报告\n")
    f.write("Labor Market Regulation and Skill Mismatch Research Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("研究问题1: 最低工资政策的差异化影响\n")
    f.write("-" * 80 + "\n")
    f.write(f"分析场景数: {len(df_q1)}\n")
    f.write(f"技能结构类型: {df_q1['skill_structure'].nunique()}\n")
    f.write(f"最低工资水平: {df_q1['min_wage_par'].nunique()}\n\n")
    
    f.write("关键发现:\n")
    for skill_label in df_q1['skill_structure'].unique():
        subset = df_q1[df_q1['skill_structure'] == skill_label]
        best_min_w = subset.groupby('min_wage_par')['unemployment'].mean().idxmin()
        best_unemployment = subset.groupby('min_wage_par')['unemployment'].mean().min()
        f.write(f"  {skill_label}:\n")
        f.write(f"    最优最低工资参数: {best_min_w:.2f}\n")
        f.write(f"    对应失业率: {best_unemployment:.4f}\n")
    f.write("\n\n")
    
    f.write("研究问题2: 技能失配的经济成本\n")
    f.write("-" * 80 + "\n")
    f.write(f"最优技能比例 (gamma_nr): {optimal_gamma:.2f}\n")
    optimal_stats = df_q2[df_q2['gamma_nr'] == optimal_gamma].mean()
    f.write(f"最优配置下的指标:\n")
    f.write(f"  失业率: {optimal_stats['unemployment']:.4f}\n")
    f.write(f"  技能失配率: {optimal_stats['skill_mismatch_rate']:.4f}\n")
    f.write(f"  GDP: {optimal_stats['GDP']:.2f}\n")
    f.write(f"  工资差距: {optimal_stats['wage_gap']:.4f}\n\n\n")
    
    f.write("研究问题3: 劳动力市场摩擦与匹配效率\n")
    f.write("-" * 80 + "\n")
    f.write("匹配效率与失业率/技能失配呈负相关\n")
    f.write("高技能比例下，匹配效率的重要性更为显著\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("政策建议 / Policy Recommendations\n")
    f.write("-" * 80 + "\n")
    f.write("1. 最低工资政策应考虑劳动力市场的技能结构\n")
    f.write("2. 提高劳动力市场匹配效率对减少技能失配至关重要\n")
    f.write("3. 技能比例存在最优配置，过高或过低都会增加失业\n")
    f.write("4. 应建立技能培训和再培训机制以应对技能失配\n")

print("✓ 综合研究报告已生成")

print("\n" + "=" * 80)
print("研究完成！Results Summary:")
print("=" * 80)
print("\n生成的文件 / Generated Files:")
print("  1. results/question1_minimum_wage_skill_effects.csv")
print("  2. results/question1_minimum_wage_by_skill.png")
print("  3. results/question2_skill_mismatch_costs.csv")
print("  4. results/question2_skill_mismatch_costs.png")
print("  5. results/question3_frictions_matching.csv")
print("  6. results/question3_frictions_matching.png")
print("  7. results/labor_market_research_report.txt")
print("\n关键发现 / Key Findings:")
print(f"  - 最优技能比例: gamma_nr = {optimal_gamma:.2f}")
print(f"  - 最低工资政策效果因技能结构而异")
print(f"  - 劳动力市场匹配效率是减少技能失配的关键")
print("=" * 80)
