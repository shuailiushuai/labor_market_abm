"""
Quick Start Example: Basic Usage of Analysis Modules
快速开始示例：分析模块基本使用

This is a minimal example showing how to quickly use each analysis module.
这是一个最小示例，展示如何快速使用每个分析模块。
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

print("Quick Start: Labor Market ABM Analysis Modules")
print("=" * 60)

# Set random seed for reproducibility
rd.seed(42)

# Define base parameters
base_params = {
    'H': 300,           # Number of households
    'F': 50,            # Number of firms
    'alpha_2': 0.25,    # Wealth consumption propensity
    'lambda_LM': 5,     # Labor market matching intensity
    'sigma_w': 0.005,   # Wage adjustment variance
    'gamma_nr': 0.33,   # Share of non-routine workers
    'min_w_par': 0.3,   # Minimum wage parameter
}

# =============================================================================
# Example 1: Quick Sensitivity Analysis
# =============================================================================
print("\n1. Running Quick Sensitivity Analysis...")
print("-" * 60)

analyzer = SensitivityAnalyzer(T=200, burn_in=100)

# Define a few parameters to test
param_ranges = {
    'lambda_LM': (2, 8),
    'sigma_w': (0.002, 0.008),
    'min_w_par': (0.2, 0.4),
}

# Run with fewer samples for quick demo
mc_results = analyzer.monte_carlo_analysis(param_ranges, n_samples=20)

print(f"✓ Analyzed {len(mc_results)} scenarios")
print(f"  Mean unemployment: {mc_results['mean_unemployment_rate'].mean():.4f}")
print(f"  Mean GDP: {mc_results['mean_GDP'].mean():.2f}")

# Get most influential parameters
correlations = analyzer.calculate_correlation_coefficients()
print("\nMost influential parameter for unemployment:")
abs_corr = correlations['mean_unemployment_rate'].abs()
top_param = abs_corr.idxmax()
print(f"  {top_param}: correlation = {correlations.loc[top_param, 'mean_unemployment_rate']:.3f}")

# Save results
analyzer.plot_correlation_heatmap(save_path='results/quick_sensitivity.png')
print("  Saved: results/quick_sensitivity.png")

# =============================================================================
# Example 2: Quick Model Validation
# =============================================================================
print("\n2. Running Quick Model Validation...")
print("-" * 60)

# Run a baseline model
m = Model(T=200, **base_params)
m.run()

validator = EmpiricalValidator()
validator.set_model_results(m)

# Create synthetic empirical data (in practice, load real data)
emp_unemployment = np.random.normal(0.08, 0.02, 100)
emp_vacancies = 50 - 200 * emp_unemployment + np.random.normal(0, 5, 100)

# Validate Beveridge curve
bev_results = validator.validate_beveridge_curve(emp_unemployment, emp_vacancies)

print(f"✓ Beveridge Curve Validation:")
print(f"  Empirical correlation: {bev_results['emp_correlation']:.3f}")
print(f"  Model correlation: {bev_results['model_correlation']:.3f}")
print(f"  Negative correlation: {bev_results['negative_correlation']}")

# Save validation plot
validator.plot_validation_results('beveridge_curve', 
                                 emp_data={'unemployment': emp_unemployment, 
                                          'vacancies': emp_vacancies},
                                 save_path='results/quick_validation.png')
print("  Saved: results/quick_validation.png")

# =============================================================================
# Example 3: Quick Regulation Analysis
# =============================================================================
print("\n3. Running Quick Regulation Analysis...")
print("-" * 60)

reg_analyzer = RegulationAnalyzer(T=200, burn_in=100)

# Test 3 minimum wage levels
min_wage_results = reg_analyzer.analyze_minimum_wage_policy(
    base_params,
    min_wage_levels=[0.25, 0.35, 0.45],
    n_replications=2
)

print(f"✓ Tested {min_wage_results['min_wage_par'].nunique()} minimum wage policies")

# Find best policy
grouped = min_wage_results.groupby('min_wage_par')['mean_unemployment'].mean()
best_min_w = grouped.idxmin()
print(f"  Lowest unemployment at min_w_par = {best_min_w:.2f}")
print(f"    Unemployment rate: {grouped[best_min_w]:.4f}")

# Save plot
reg_analyzer.plot_policy_comparison(
    'minimum_wage', 'min_wage_par', 'mean_unemployment',
    save_path='results/quick_minwage.png'
)
print("  Saved: results/quick_minwage.png")

# =============================================================================
# Example 4: Quick Skill Mismatch Analysis
# =============================================================================
print("\n4. Running Quick Skill Mismatch Analysis...")
print("-" * 60)

# Test 3 skill compositions
skill_results = reg_analyzer.analyze_skill_mismatch(
    base_params,
    skill_ratios=[0.25, 0.35, 0.45],
    n_replications=2
)

print(f"✓ Tested {skill_results['gamma_nr'].nunique()} skill compositions")

# Show skill mismatch rates
grouped = skill_results.groupby('gamma_nr').agg({
    'mean_unemployment': 'mean',
    'skill_mismatch': 'mean'
})
print("\nSkill composition effects:")
for gamma_nr, row in grouped.iterrows():
    print(f"  gamma_nr={gamma_nr:.2f}: unemployment={row['mean_unemployment']:.4f}, "
          f"mismatch={row['skill_mismatch']:.4f}")

# Save plot
reg_analyzer.plot_multi_metric_comparison(
    'skill_mismatch', 'gamma_nr',
    metrics=['mean_unemployment', 'skill_mismatch'],
    save_path='results/quick_skill_mismatch.png'
)
print("  Saved: results/quick_skill_mismatch.png")

# =============================================================================
# Example 5: Quick Scenario Comparison
# =============================================================================
print("\n5. Running Quick Scenario Comparison...")
print("-" * 60)

comp_analyzer = ComparativeAnalyzer()

# Add scenarios from previous analyses
baseline_data = min_wage_results[min_wage_results['min_wage_par'] == 0.35]
low_minw_data = min_wage_results[min_wage_results['min_wage_par'] == 0.25]
high_minw_data = min_wage_results[min_wage_results['min_wage_par'] == 0.45]

comp_analyzer.add_scenario('Baseline (0.35)', baseline_data, "Standard minimum wage")
comp_analyzer.add_scenario('Low MinWage (0.25)', low_minw_data, "Lower minimum wage")
comp_analyzer.add_scenario('High MinWage (0.45)', high_minw_data, "Higher minimum wage")

# Compare scenarios
comparison = comp_analyzer.compare_scenarios(
    ['Baseline (0.35)', 'Low MinWage (0.25)', 'High MinWage (0.45)']
)

print("✓ Scenario Comparison:")
for _, row in comparison.iterrows():
    print(f"  {row['scenario']}:")
    print(f"    Unemployment: {row['mean_unemployment_mean']:.4f}")
    print(f"    GDP: {row['mean_GDP_mean']:.2f}")

# Statistical test
test = comp_analyzer.statistical_comparison(
    'Baseline (0.35)', 'High MinWage (0.45)', 'mean_unemployment'
)
print(f"\nStatistical test (Baseline vs High MinWage):")
print(f"  Difference: {test['diff']:.4f}")
print(f"  Significant: {test['significant']} (p={test['pvalue']:.4f})")

# Save comparison plot
comp_analyzer.plot_scenario_comparison(
    ['Baseline (0.35)', 'Low MinWage (0.25)', 'High MinWage (0.45)'],
    ['mean_unemployment', 'mean_GDP'],
    save_path='results/quick_comparison.png'
)
print("  Saved: results/quick_comparison.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Quick Start Complete!")
print("=" * 60)
print("\nGenerated files in results/ directory:")
print("  1. quick_sensitivity.png - Parameter sensitivity heatmap")
print("  2. quick_validation.png - Beveridge curve validation")
print("  3. quick_minwage.png - Minimum wage policy effects")
print("  4. quick_skill_mismatch.png - Skill composition effects")
print("  5. quick_comparison.png - Scenario comparison")
print("\nNext steps:")
print("  - Run example_comprehensive_analysis.py for full analysis")
print("  - Run example_focused_research.py for detailed research questions")
print("  - See ANALYSIS_MODULES_README.md for complete documentation")
print("=" * 60)
