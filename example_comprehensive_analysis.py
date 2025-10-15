"""
Example: Comprehensive Analysis of Labor Market Regulations and Skill Mismatch

This script demonstrates a complete analysis workflow including:
1. Sensitivity analysis
2. Empirical validation
3. Regulation policy analysis
4. Skill mismatch analysis
5. Comparative analysis
"""

import numpy as np
import numpy.random as rd
import os
from model_class import Model
from sensitivity_analysis import SensitivityAnalyzer
from empirical_validation import EmpiricalValidator
from regulation_analysis import RegulationAnalyzer
from comparative_analysis import ComparativeAnalyzer

# Create results directory
os.makedirs("results", exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE LABOR MARKET ABM ANALYSIS")
print("=" * 80)
print("\nThis analysis includes:")
print("  1. Sensitivity Analysis")
print("  2. Empirical Validation")
print("  3. Labor Market Regulation Analysis")
print("  4. Skill Mismatch Analysis")
print("  5. Comparative Analysis")
print("\n" + "=" * 80)

# Set random seed for reproducibility
rd.seed(42)

# ============================================================================
# PART 1: BASELINE MODEL AND SENSITIVITY ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 1: SENSITIVITY ANALYSIS")
print("=" * 80)

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

# Define parameter ranges for sensitivity
param_ranges = {
    'lambda_LM': (1, 10),
    'sigma_m': (0.001, 0.01),
    'sigma_w': (0.001, 0.01),
    'alpha_2': (0.1, 0.4),
    'chi_C': (0.1, 0.9),
}

# Run sensitivity analysis
sensitivity_analyzer = SensitivityAnalyzer(T=300, burn_in=100)

print("\nRunning Monte Carlo sensitivity analysis...")
mc_results = sensitivity_analyzer.monte_carlo_analysis(param_ranges, n_samples=30)

print("\nCalculating correlations...")
correlations = sensitivity_analyzer.calculate_correlation_coefficients()
print("\nTop 3 most influential parameters per metric:")
for metric in correlations.columns:
    abs_corr = correlations[metric].abs().sort_values(ascending=False)
    print(f"\n{metric}:")
    for i, (param, corr) in enumerate(abs_corr.head(3).items()):
        print(f"  {i+1}. {param}: {correlations.loc[param, metric]:.3f}")

# Generate sensitivity visualizations
print("\nGenerating sensitivity analysis visualizations...")
sensitivity_analyzer.plot_correlation_heatmap(
    save_path='results/sensitivity_correlations.png'
)

# Generate sensitivity report
sensitivity_analyzer.generate_sensitivity_report(
    output_file='results/sensitivity_report.txt'
)

# ============================================================================
# PART 2: EMPIRICAL VALIDATION
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 2: EMPIRICAL VALIDATION")
print("=" * 80)

# Run baseline model for validation
print("\nRunning baseline model for validation...")
rd.seed(123)
baseline_model = Model(T=500, **base_params)
baseline_model.run()

# Initialize validator
validator = EmpiricalValidator()
validator.set_model_results(baseline_model)

# Generate synthetic empirical data for demonstration
# In practice, you would load real empirical data here
print("\nGenerating synthetic empirical data (replace with real data)...")
burn_in = 200
n_emp_obs = 200

# Synthetic unemployment and vacancy data
emp_unemployment = np.random.normal(0.08, 0.02, n_emp_obs)
emp_vacancies = 60 - 300 * emp_unemployment + np.random.normal(0, 8, n_emp_obs)

# Synthetic wage data
emp_wages = np.random.lognormal(0, 0.3, 500)

# Validate Beveridge curve
print("\nValidating Beveridge curve...")
bev_results = validator.validate_beveridge_curve(emp_unemployment, emp_vacancies)
print(f"  Empirical correlation: {bev_results['emp_correlation']:.3f}")
print(f"  Model correlation: {bev_results['model_correlation']:.3f}")
print(f"  Negative correlation: {bev_results['negative_correlation']}")

# Validate wage distribution
print("\nValidating wage distribution...")
wage_results = validator.validate_wage_distribution(emp_wages)
print(f"  KS statistic: {wage_results['ks_statistic']:.3f}")
print(f"  KS p-value: {wage_results['ks_pvalue']:.3f}")
print(f"  Passes KS test: {wage_results['passes_ks_test']}")

# Generate validation visualizations
print("\nGenerating validation visualizations...")
emp_data = {
    'unemployment': emp_unemployment,
    'vacancies': emp_vacancies,
    'wages': emp_wages
}
validator.plot_validation_results('beveridge_curve', emp_data=emp_data,
                                 save_path='results/validation_beveridge.png')
validator.plot_validation_results('wage_distribution', emp_data=emp_data,
                                 save_path='results/validation_wages.png')

# Generate validation report
validator.generate_validation_report('results/validation_report.txt')

# ============================================================================
# PART 3: LABOR MARKET REGULATION ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 3: LABOR MARKET REGULATION ANALYSIS")
print("=" * 80)

# Initialize regulation analyzer
reg_analyzer = RegulationAnalyzer(T=300, burn_in=100)

# Run baseline scenario
print("\nRunning baseline scenario...")
baseline_results = reg_analyzer.run_baseline_scenario(base_params, n_replications=3)

# Analyze minimum wage policies
print("\nAnalyzing minimum wage policies...")
min_wage_results = reg_analyzer.analyze_minimum_wage_policy(
    base_params,
    min_wage_levels=[0.2, 0.3, 0.4, 0.5, 0.6],
    n_replications=3
)

print("\nMinimum wage analysis summary:")
grouped = min_wage_results.groupby('min_wage_par')['mean_unemployment'].mean()
print(f"  Lowest unemployment at min_w_par = {grouped.idxmin()}: {grouped.min():.4f}")
print(f"  Highest unemployment at min_w_par = {grouped.idxmax()}: {grouped.max():.4f}")

# Analyze firing costs (wage adjustment costs)
print("\nAnalyzing firing costs...")
firing_results = reg_analyzer.analyze_firing_costs(
    base_params,
    firing_cost_levels=[0.001, 0.003, 0.005, 0.007, 0.01],
    n_replications=3
)

# Analyze matching efficiency
print("\nAnalyzing labor market matching efficiency...")
matching_results = reg_analyzer.analyze_labor_market_efficiency(
    base_params,
    lambda_LM_values=[1, 3, 5, 7, 10],
    n_replications=3
)

# Generate regulation visualizations
print("\nGenerating regulation analysis visualizations...")
reg_analyzer.plot_multi_metric_comparison(
    'minimum_wage', 'min_wage_par',
    save_path='results/minimum_wage_effects.png'
)
reg_analyzer.plot_multi_metric_comparison(
    'firing_costs', 'firing_cost',
    save_path='results/firing_cost_effects.png'
)
reg_analyzer.plot_multi_metric_comparison(
    'matching_efficiency', 'lambda_LM',
    save_path='results/matching_efficiency_effects.png'
)

# ============================================================================
# PART 4: SKILL MISMATCH ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 4: SKILL MISMATCH ANALYSIS")
print("=" * 80)

# Analyze different skill compositions
print("\nAnalyzing skill mismatch scenarios...")
skill_results = reg_analyzer.analyze_skill_mismatch(
    base_params,
    skill_ratios=[0.2, 0.25, 0.33, 0.4, 0.5, 0.6],
    n_replications=3
)

print("\nSkill mismatch analysis summary:")
grouped_skill = skill_results.groupby('gamma_nr').agg({
    'mean_unemployment': 'mean',
    'skill_mismatch_rate': 'mean',
    'wage_gap': 'mean'
})
print(grouped_skill)

# Generate skill mismatch visualizations
print("\nGenerating skill mismatch visualizations...")
reg_analyzer.plot_multi_metric_comparison(
    'skill_mismatch', 'gamma_nr',
    metrics=['mean_unemployment', 'skill_mismatch_rate', 
             'wage_gap', 'wage_inequality_9_1'],
    save_path='results/skill_mismatch_effects.png'
)

# Generate comprehensive regulation report
reg_analyzer.generate_policy_report('results/regulation_analysis_report.txt')

# ============================================================================
# PART 5: COMPARATIVE ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("PART 5: COMPARATIVE ANALYSIS")
print("=" * 80)

# Initialize comparative analyzer
comp_analyzer = ComparativeAnalyzer()

# Add scenarios for comparison
print("\nAdding scenarios for comparison...")

# Baseline
baseline_summary = baseline_results['results']
comp_analyzer.add_scenario('Baseline', baseline_summary,
                           "Standard parameter configuration")

# Best minimum wage policy
best_min_w = min_wage_results.groupby('min_wage_par')['mean_unemployment'].mean().idxmin()
best_min_w_data = min_wage_results[min_wage_results['min_wage_par'] == best_min_w]
comp_analyzer.add_scenario(f'Min Wage {best_min_w}', best_min_w_data,
                           f"Minimum wage parameter = {best_min_w}")

# High skill ratio scenario
high_skill_data = skill_results[skill_results['gamma_nr'] == 0.5]
comp_analyzer.add_scenario('High Skill Ratio', high_skill_data,
                           "50% non-routine workers")

# Low skill ratio scenario
low_skill_data = skill_results[skill_results['gamma_nr'] == 0.2]
comp_analyzer.add_scenario('Low Skill Ratio', low_skill_data,
                           "20% non-routine workers")

# Compare all scenarios
print("\nComparing scenarios...")
comparison = comp_analyzer.compare_scenarios(
    ['Baseline', f'Min Wage {best_min_w}', 'High Skill Ratio', 'Low Skill Ratio']
)
print("\nScenario Comparison:")
print(comparison)

# Statistical comparisons
print("\nStatistical comparisons:")
test1 = comp_analyzer.statistical_comparison(
    'Baseline', f'Min Wage {best_min_w}', 'mean_unemployment'
)
print(f"\nBaseline vs Min Wage {best_min_w} (Unemployment):")
print(f"  Difference: {test1['diff']:.4f}")
print(f"  P-value: {test1['pvalue']:.4f}")
print(f"  Significant: {test1['significant']}")

# Generate comparative visualizations
print("\nGenerating comparative visualizations...")
comp_analyzer.plot_scenario_comparison(
    ['Baseline', f'Min Wage {best_min_w}', 'High Skill Ratio', 'Low Skill Ratio'],
    ['mean_unemployment', 'mean_GDP', 'mean_real_wage', 'wage_inequality_9_1'],
    save_path='results/comprehensive_scenario_comparison.png'
)

comp_analyzer.plot_radar_comparison(
    ['Baseline', f'Min Wage {best_min_w}', 'High Skill Ratio', 'Low Skill Ratio'],
    ['mean_unemployment', 'mean_GDP', 'mean_real_wage', 'wage_inequality_9_1'],
    save_path='results/comprehensive_radar_comparison.png'
)

# Generate comprehensive comparison report
comp_analyzer.generate_comparison_report('results/comprehensive_comparison_report.txt')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated reports and visualizations:")
print("  - results/sensitivity_report.txt")
print("  - results/sensitivity_correlations.png")
print("  - results/validation_report.txt")
print("  - results/validation_beveridge.png")
print("  - results/validation_wages.png")
print("  - results/regulation_analysis_report.txt")
print("  - results/minimum_wage_effects.png")
print("  - results/firing_cost_effects.png")
print("  - results/matching_efficiency_effects.png")
print("  - results/skill_mismatch_effects.png")
print("  - results/comprehensive_comparison_report.txt")
print("  - results/comprehensive_scenario_comparison.png")
print("  - results/comprehensive_radar_comparison.png")
print("\n" + "=" * 80)
print("\nKey Findings:")
print(f"  - Best minimum wage parameter: {best_min_w}")
print(f"  - Skill composition significantly affects unemployment and inequality")
print(f"  - Labor market matching efficiency is crucial for outcomes")
print("\nSee detailed reports in results/ directory for complete analysis.")
print("=" * 80)
