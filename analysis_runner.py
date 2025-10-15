"""
Unified Analysis Runner for Labor Market ABM

This module provides a comprehensive interface for running all analyses:
- Sensitivity analysis
- Skill mismatch analysis  
- Digitalization and regulation analysis
- Empirical validation
- Parameter estimation
"""

import numpy as np
import numpy.random as rd
from model_class import Model
from sensitivity_analysis import SensitivityAnalysis, example_sensitivity_analysis
from skill_mismatch_analysis import SkillMismatchAnalyzer, example_skill_mismatch_analysis
from digitalization_regulation import DigitalizationModel, DigitalizationAnalyzer, example_digitalization_analysis
from empirical_validation import EmpiricalValidator, example_empirical_validation
from parameter_estimation import ParameterEstimator, example_parameter_estimation
import os


class ComprehensiveAnalyzer:
    """
    Comprehensive analyzer combining all analysis modules
    """
    
    def __init__(self, output_dir='analysis_results'):
        """
        Initialize comprehensive analyzer
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
    
    def run_full_analysis(self, 
                         run_sensitivity=True,
                         run_mismatch=True, 
                         run_digitalization=True,
                         run_validation=False,
                         run_estimation=False):
        """
        Run comprehensive analysis
        
        Parameters:
        -----------
        run_sensitivity : bool
            Run sensitivity analysis
        run_mismatch : bool
            Run skill mismatch analysis
        run_digitalization : bool
            Run digitalization scenarios
        run_validation : bool
            Run empirical validation (requires empirical data)
        run_estimation : bool
            Run parameter estimation (requires empirical data)
        """
        print("=" * 80)
        print("COMPREHENSIVE LABOR MARKET ABM ANALYSIS")
        print("=" * 80)
        
        # Sensitivity Analysis
        if run_sensitivity:
            print("\n" + "-" * 80)
            print("1. SENSITIVITY ANALYSIS")
            print("-" * 80)
            
            param_ranges = {
                'sigma_m': (0.001, 0.01),
                'sigma_w': (0.001, 0.01),
                'lambda_LM': (1, 10),
                'mu_r': (0.2, 0.5),
                'gamma_nr': (0.2, 0.5),
            }
            
            base_params = {
                'H': 200, 'F': 20, 'T': 300,
                'alpha_2': 0.25, 'chi_C': 0.2,
                'sigma_delta': 0.001, 'nu': 0.1,
                'u_r': 0.08, 'beta': 1,
                'lambda_exp': 0.5, 'N_app': 4,
                'sigma': 1.5, 'nr_to_r': False,
                'a': 100, 'min_w_par': 0.3,
                'W_r': 1, 'f_max': 3
            }
            
            sa = SensitivityAnalysis(param_ranges, T=300, burnin=100)
            
            # Run OAT
            print("Running One-At-A-Time analysis...")
            oat_results = sa.oat_sensitivity(base_params, n_samples=4)
            sa.plot_oat_results(f'{self.output_dir}/sensitivity_oat.png')
            
            # Run Morris
            print("Running Morris screening...")
            morris_results = sa.morris_screening(base_params, n_trajectories=4, n_levels=3)
            sa.plot_morris_results(f'{self.output_dir}/sensitivity_morris.png')
            
            sa.save_results(f'{self.output_dir}/sensitivity_results.csv')
            
            self.results['sensitivity'] = sa
            print("✓ Sensitivity analysis complete")
        
        # Skill Mismatch Analysis
        if run_mismatch:
            print("\n" + "-" * 80)
            print("2. SKILL MISMATCH ANALYSIS")
            print("-" * 80)
            
            print("Running model for mismatch analysis...")
            rd.seed(42)
            m = Model(
                T=400, H=200, F=20,
                alpha_2=0.25, chi_C=0.2,
                lambda_LM=5, sigma_m=0.005,
                sigma_w=0.005, nu=0.1,
                u_r=0.08, beta=1,
                lambda_exp=0.5, N_app=4,
                sigma=1.5, mu_r=0.3,
                nr_to_r=True, gamma_nr=0.33,
                min_w_par=0.3
            )
            m.run()
            
            analyzer = SkillMismatchAnalyzer(m)
            
            # Current metrics
            metrics = analyzer.get_mismatch_metrics()
            wage_metrics = analyzer.get_wage_penalties()
            
            # Time evolution
            fig, df = analyzer.plot_mismatch_evolution(
                start=100, 
                save_path=f'{self.output_dir}/mismatch_evolution.png'
            )
            
            # Report
            analyzer.generate_mismatch_report(
                start=100,
                save_path=f'{self.output_dir}/mismatch_report.txt'
            )
            
            self.results['mismatch'] = {
                'analyzer': analyzer,
                'metrics': metrics,
                'wage_metrics': wage_metrics,
                'time_series': df
            }
            print("✓ Skill mismatch analysis complete")
        
        # Digitalization and Regulation Analysis
        if run_digitalization:
            print("\n" + "-" * 80)
            print("3. DIGITALIZATION & REGULATION ANALYSIS")
            print("-" * 80)
            
            base_params = {
                'H': 200, 'F': 20,
                'alpha_2': 0.25, 'chi_C': 0.2,
                'lambda_LM': 5, 'sigma_m': 0.005,
                'sigma_w': 0.005, 'nu': 0.1,
                'u_r': 0.08, 'beta': 1,
                'lambda_exp': 0.5, 'N_app': 4,
                'sigma': 1.5, 'mu_r': 0.3,
                'nr_to_r': True, 'gamma_nr': 0.33,
                'min_w_par': 0.3
            }
            
            scenarios = {
                'Baseline': {
                    **base_params,
                    'automation_rate': 0.0,
                },
                'Automation': {
                    **base_params,
                    'automation_rate': 0.02,
                    'automation_start': 100,
                    'automation_routine_impact': 0.7,
                    'automation_cost_reduction': 0.2,
                },
                'Automation + Retraining': {
                    **base_params,
                    'automation_rate': 0.02,
                    'automation_start': 100,
                    'automation_routine_impact': 0.7,
                    'automation_cost_reduction': 0.2,
                    'retraining_policy': True,
                    'training_effectiveness': 0.3,
                    'training_cost': 0.5,
                },
            }
            
            digi_analyzer = DigitalizationAnalyzer()
            results = digi_analyzer.compare_scenarios(scenarios, T=400, n_runs=2)
            
            digi_analyzer.plot_scenario_comparison(
                f'{self.output_dir}/scenario_comparison.png'
            )
            digi_analyzer.generate_policy_report(
                f'{self.output_dir}/policy_report.txt'
            )
            
            self.results['digitalization'] = digi_analyzer
            print("✓ Digitalization analysis complete")
        
        # Empirical Validation
        if run_validation:
            print("\n" + "-" * 80)
            print("4. EMPIRICAL VALIDATION")
            print("-" * 80)
            print("Note: Using example empirical data. Replace with real data.")
            
            # Run model
            rd.seed(42)
            m = Model(T=400, H=200, F=20)
            m.run()
            
            validator = EmpiricalValidator(m)
            
            # Example empirical moments
            empirical_moments = {
                'mean_unemployment': 0.06,
                'mean_real_wage': 0.95,
                'std_unemployment': 0.015,
                'wage_inequality_9010': 3.5,
            }
            
            moment_results = validator.moment_matching(empirical_moments)
            gof = validator.goodness_of_fit(empirical_moments)
            
            validator.plot_moment_comparison(
                f'{self.output_dir}/moment_comparison.png'
            )
            validator.generate_validation_report(
                f'{self.output_dir}/validation_report.txt'
            )
            
            self.results['validation'] = validator
            print("✓ Empirical validation complete")
        
        # Parameter Estimation
        if run_estimation:
            print("\n" + "-" * 80)
            print("5. PARAMETER ESTIMATION")
            print("-" * 80)
            print("Note: Using example empirical data. Replace with real data.")
            
            empirical_data = {
                'mean_unemployment': 0.06,
                'mean_real_wage': 0.95,
                'wage_inequality': 3.5,
            }
            
            param_bounds = {
                'lambda_LM': (1, 10),
                'sigma_w': (0.001, 0.01),
            }
            
            base_params = {
                'H': 200, 'F': 20,
                'alpha_2': 0.25, 'chi_C': 0.2,
                'sigma_m': 0.005, 'nu': 0.1,
                'u_r': 0.08, 'beta': 1,
                'lambda_exp': 0.5, 'N_app': 4,
                'sigma': 1.5, 'mu_r': 0.3,
                'nr_to_r': True, 'gamma_nr': 0.33,
                'min_w_par': 0.3,
            }
            
            estimator = ParameterEstimator(
                empirical_data, param_bounds,
                T=300, burnin=100
            )
            
            # Grid search
            grid_results = estimator.grid_search(base_params, n_points=3)
            
            self.results['estimation'] = estimator
            print("✓ Parameter estimation complete")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.output_dir}/")
        print("=" * 80)
        
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """
        Generate overall summary report
        """
        filepath = f'{self.output_dir}/ANALYSIS_SUMMARY.txt'
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE LABOR MARKET ABM ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("This analysis examines:\n")
            f.write("1. Parameter sensitivity and robustness\n")
            f.write("2. Skill mismatch dynamics and wage penalties\n")
            f.write("3. Digitalization effects and policy interventions\n")
            f.write("4. Model validation against empirical data\n")
            f.write("5. Parameter estimation from data\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODULES RUN\n")
            f.write("=" * 80 + "\n\n")
            
            for module_name in self.results.keys():
                f.write(f"✓ {module_name.upper()}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("=" * 80 + "\n\n")
            
            # List all files in output directory
            for filename in sorted(os.listdir(self.output_dir)):
                if filename != 'ANALYSIS_SUMMARY.txt':
                    f.write(f"  - {filename}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 80 + "\n\n")
            
            if 'mismatch' in self.results:
                metrics = self.results['mismatch']['metrics']
                f.write("SKILL MISMATCH:\n")
                f.write(f"  - Over-qualification rate: {metrics['overqualified_rate']:.2%}\n")
                f.write(f"  - Under-qualification rate: {metrics['underqualified_rate']:.2%}\n")
                f.write(f"  - Total mismatch rate: {metrics['total_mismatch_rate']:.2%}\n\n")
            
            f.write("\nFor detailed results, see individual report files.\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nSummary report saved to: {filepath}")


def run_quick_demo():
    """
    Quick demo of main functionality
    """
    analyzer = ComprehensiveAnalyzer(output_dir='demo_results')
    
    # Run subset of analyses (faster for demo)
    analyzer.run_full_analysis(
        run_sensitivity=True,
        run_mismatch=True,
        run_digitalization=True,
        run_validation=False,
        run_estimation=False
    )
    
    return analyzer


def run_full_suite():
    """
    Run complete analysis suite
    """
    analyzer = ComprehensiveAnalyzer(output_dir='full_analysis_results')
    
    analyzer.run_full_analysis(
        run_sensitivity=True,
        run_mismatch=True,
        run_digitalization=True,
        run_validation=True,
        run_estimation=False  # Time-intensive
    )
    
    return analyzer


if __name__ == '__main__':
    print("Labor Market ABM - Comprehensive Analysis\n")
    print("Choose analysis mode:")
    print("1. Quick demo (subset of analyses)")
    print("2. Full analysis suite")
    print("3. Custom (modify code)\n")
    
    choice = input("Enter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\nRunning quick demo...")
        analyzer = run_quick_demo()
    elif choice == "2":
        print("\nRunning full analysis suite (this may take a while)...")
        analyzer = run_full_suite()
    else:
        print("\nPlease modify the script for custom analysis.")
        print("See example functions for guidance.")
