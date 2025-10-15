"""
Empirical Validation Module for Labor Market ABM

This module provides tools for validating model outputs against empirical data:
1. Goodness-of-fit tests
2. Moment matching
3. Time series validation
4. Distribution comparisons
5. Statistical tests
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from model_class import Model


class EmpiricalValidator:
    """
    Validator for comparing model outputs with empirical data
    """
    
    def __init__(self, model=None):
        """
        Initialize validator
        
        Parameters:
        -----------
        model : Model
            Labor market model instance
        """
        self.model = model
        self.validation_results = {}
    
    def load_empirical_data(self, data_source):
        """
        Load empirical data for comparison
        
        Parameters:
        -----------
        data_source : dict or pd.DataFrame
            Empirical data with metrics as keys
            
        Returns:
        --------
        dict : Structured empirical data
        """
        if isinstance(data_source, pd.DataFrame):
            return data_source.to_dict('list')
        return data_source
    
    def moment_matching(self, empirical_moments, model_moments=None):
        """
        Compare empirical and model moments
        
        Parameters:
        -----------
        empirical_moments : dict
            Dictionary of empirical moments
        model_moments : dict
            Dictionary of model moments (computed if None)
            
        Returns:
        --------
        dict : Moment comparison results
        """
        if model_moments is None and self.model is not None:
            model_moments = self.compute_model_moments()
        
        results = {}
        
        for moment_name in empirical_moments.keys():
            if moment_name in model_moments:
                emp_val = empirical_moments[moment_name]
                model_val = model_moments[moment_name]
                
                # Calculate relative error
                rel_error = abs(model_val - emp_val) / abs(emp_val) if emp_val != 0 else np.nan
                
                results[moment_name] = {
                    'empirical': emp_val,
                    'model': model_val,
                    'absolute_error': model_val - emp_val,
                    'relative_error': rel_error,
                    'match_quality': 'good' if rel_error < 0.1 else 'moderate' if rel_error < 0.3 else 'poor'
                }
        
        self.validation_results['moment_matching'] = results
        return results
    
    def compute_model_moments(self, burnin=100):
        """
        Compute key moments from model output
        
        Parameters:
        -----------
        burnin : int
            Periods to exclude
            
        Returns:
        --------
        dict : Dictionary of moments
        """
        if self.model is None:
            raise ValueError("No model instance provided")
        
        m = self.model
        t_start = burnin
        
        moments = {
            # First moments (means)
            'mean_unemployment': np.mean(m.u_r_arr[t_start:]),
            'mean_routine_unemployment': np.mean(m.ur_r_arr[t_start:]),
            'mean_nonroutine_unemployment': np.mean(m.unr_r_arr[t_start:]),
            'mean_real_wage': np.mean(m.mean_w_arr[t_start:]),
            'mean_routine_wage': np.mean(m.mean_r_w_arr[t_start:]),
            'mean_nonroutine_wage': np.mean(m.mean_nr_w_arr[t_start:]),
            'mean_gdp': np.mean(m.GDP[t_start:]),
            
            # Second moments (volatilities)
            'std_unemployment': np.std(m.u_r_arr[t_start:]),
            'std_real_wage': np.std(m.mean_w_arr[t_start:]),
            'std_gdp': np.std(m.GDP[t_start:]),
            
            # Inequality measures
            'wage_inequality_9010': np.mean(m.nine_to_one[t_start:]),
            'wage_inequality_905': np.mean(m.nine_to_five[t_start:]),
            'wage_inequality_501': np.mean(m.five_to_one[t_start:]),
            'wage_variance': np.mean(m.wage_variance[t_start:]),
            
            # Labor market measures
            'skill_mismatch': np.mean(m.share_nr_in_r[t_start:]) / m.H_nr,
            'firm_default_rate': np.mean(m.share_inactive[t_start:]),
            'mean_vacancies': np.mean(m.open_vs[t_start:]),
            
            # Correlations
            'corr_gdp_unemployment': np.corrcoef(
                m.GDP[t_start:], m.u_r_arr[t_start:])[0, 1],
            'corr_wage_unemployment': np.corrcoef(
                m.mean_w_arr[t_start:], m.u_r_arr[t_start:])[0, 1],
        }
        
        return moments
    
    def distribution_comparison(self, empirical_dist, model_dist, dist_name='wage'):
        """
        Compare empirical and model distributions
        
        Parameters:
        -----------
        empirical_dist : array-like
            Empirical distribution data
        model_dist : array-like
            Model distribution data
        dist_name : str
            Name of the distribution
            
        Returns:
        --------
        dict : Comparison statistics
        """
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(empirical_dist, model_dist)
        
        # Anderson-Darling test (approximate)
        # Compare quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        emp_quantiles = np.percentile(empirical_dist, [q*100 for q in quantiles])
        model_quantiles = np.percentile(model_dist, [q*100 for q in quantiles])
        
        results = {
            'distribution': dist_name,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'ks_reject': ks_pval < 0.05,
            'empirical_mean': np.mean(empirical_dist),
            'model_mean': np.mean(model_dist),
            'empirical_std': np.std(empirical_dist),
            'model_std': np.std(model_dist),
            'empirical_quantiles': dict(zip(quantiles, emp_quantiles)),
            'model_quantiles': dict(zip(quantiles, model_quantiles)),
        }
        
        return results
    
    def goodness_of_fit(self, empirical_data, tolerance=0.1):
        """
        Overall goodness-of-fit assessment
        
        Parameters:
        -----------
        empirical_data : dict
            Empirical moments and distributions
        tolerance : float
            Acceptable relative error
            
        Returns:
        --------
        dict : Goodness-of-fit assessment
        """
        moment_results = self.moment_matching(empirical_data)
        
        # Count matches
        n_total = len(moment_results)
        n_good = sum(1 for r in moment_results.values() 
                    if r['match_quality'] == 'good')
        n_moderate = sum(1 for r in moment_results.values() 
                        if r['match_quality'] == 'moderate')
        
        fit_score = (n_good + 0.5*n_moderate) / n_total
        
        results = {
            'n_moments': n_total,
            'n_good_matches': n_good,
            'n_moderate_matches': n_moderate,
            'n_poor_matches': n_total - n_good - n_moderate,
            'fit_score': fit_score,
            'overall_assessment': (
                'excellent' if fit_score > 0.8 else
                'good' if fit_score > 0.6 else
                'moderate' if fit_score > 0.4 else
                'poor'
            )
        }
        
        self.validation_results['goodness_of_fit'] = results
        return results
    
    def time_series_validation(self, empirical_series, model_series, metric_name):
        """
        Validate time series properties
        
        Parameters:
        -----------
        empirical_series : array-like
            Empirical time series
        model_series : array-like
            Model time series
        metric_name : str
            Name of the metric
            
        Returns:
        --------
        dict : Time series validation results
        """
        # Autocorrelation comparison
        emp_acf = np.correlate(empirical_series - np.mean(empirical_series), 
                               empirical_series - np.mean(empirical_series), 
                               mode='full')
        emp_acf = emp_acf[len(emp_acf)//2:]
        emp_acf = emp_acf / emp_acf[0]
        
        model_acf = np.correlate(model_series - np.mean(model_series),
                                 model_series - np.mean(model_series),
                                 mode='full')
        model_acf = model_acf[len(model_acf)//2:]
        model_acf = model_acf / model_acf[0]
        
        # Compare first-order autocorrelation
        emp_ar1 = emp_acf[1] if len(emp_acf) > 1 else 0
        model_ar1 = model_acf[1] if len(model_acf) > 1 else 0
        
        results = {
            'metric': metric_name,
            'empirical_ar1': emp_ar1,
            'model_ar1': model_ar1,
            'ar1_error': abs(model_ar1 - emp_ar1),
            'empirical_persistence': emp_ar1 > 0.5,
            'model_persistence': model_ar1 > 0.5,
            'persistence_match': (emp_ar1 > 0.5) == (model_ar1 > 0.5)
        }
        
        return results
    
    def plot_moment_comparison(self, save_path='moment_comparison.png'):
        """
        Plot moment comparison results
        """
        if 'moment_matching' not in self.validation_results:
            print("No moment matching results. Run moment_matching first.")
            return
        
        results = self.validation_results['moment_matching']
        
        moments = list(results.keys())
        empirical_vals = [results[m]['empirical'] for m in moments]
        model_vals = [results[m]['model'] for m in moments]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1.scatter(empirical_vals, model_vals, s=100, alpha=0.6)
        
        # Add 45-degree line
        min_val = min(min(empirical_vals), min(model_vals))
        max_val = max(max(empirical_vals), max(model_vals))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax1.set_xlabel('Empirical Values')
        ax1.set_ylabel('Model Values')
        ax1.set_title('Model vs Empirical Moments')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot of relative errors
        rel_errors = [results[m]['relative_error'] for m in moments]
        colors = ['green' if results[m]['match_quality'] == 'good' else
                 'orange' if results[m]['match_quality'] == 'moderate' else 'red'
                 for m in moments]
        
        ax2.barh(range(len(moments)), rel_errors, color=colors, alpha=0.6)
        ax2.set_yticks(range(len(moments)))
        ax2.set_yticklabels(moments, fontsize=8)
        ax2.set_xlabel('Relative Error')
        ax2.set_title('Moment Matching Quality')
        ax2.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<10%)')
        ax2.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (<30%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Moment comparison plot saved to {save_path}")
        
        return fig
    
    def generate_validation_report(self, save_path='validation_report.txt'):
        """
        Generate comprehensive validation report
        """
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EMPIRICAL VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Moment matching results
            if 'moment_matching' in self.validation_results:
                f.write("MOMENT MATCHING RESULTS\n")
                f.write("-" * 70 + "\n")
                
                for moment, results in self.validation_results['moment_matching'].items():
                    f.write(f"\n{moment}:\n")
                    f.write(f"  Empirical: {results['empirical']:.4f}\n")
                    f.write(f"  Model: {results['model']:.4f}\n")
                    f.write(f"  Relative Error: {results['relative_error']:.2%}\n")
                    f.write(f"  Match Quality: {results['match_quality']}\n")
            
            # Goodness of fit
            if 'goodness_of_fit' in self.validation_results:
                f.write("\n" + "=" * 70 + "\n")
                f.write("GOODNESS-OF-FIT ASSESSMENT\n")
                f.write("-" * 70 + "\n")
                
                gof = self.validation_results['goodness_of_fit']
                f.write(f"Total moments: {gof['n_moments']}\n")
                f.write(f"Good matches: {gof['n_good_matches']}\n")
                f.write(f"Moderate matches: {gof['n_moderate_matches']}\n")
                f.write(f"Poor matches: {gof['n_poor_matches']}\n")
                f.write(f"Fit score: {gof['fit_score']:.2f}\n")
                f.write(f"Overall assessment: {gof['overall_assessment']}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Validation report saved to {save_path}")


def example_empirical_validation():
    """
    Example empirical validation
    """
    print("Running labor market model...")
    
    # Run model
    rd.seed(42)
    m = Model(T=500, H=200, F=20)
    m.run()
    
    # Create validator
    validator = EmpiricalValidator(m)
    
    # Define empirical moments (example values - replace with real data)
    empirical_moments = {
        'mean_unemployment': 0.06,
        'mean_real_wage': 0.95,
        'std_unemployment': 0.015,
        'std_real_wage': 0.08,
        'wage_inequality_9010': 3.5,
        'corr_gdp_unemployment': -0.5,
    }
    
    print("\nComputing moment matching...")
    moment_results = validator.moment_matching(empirical_moments)
    
    print("\nMoment Matching Results:")
    for moment, results in moment_results.items():
        print(f"\n{moment}:")
        print(f"  Empirical: {results['empirical']:.4f}")
        print(f"  Model: {results['model']:.4f}")
        print(f"  Relative Error: {results['relative_error']:.2%}")
        print(f"  Quality: {results['match_quality']}")
    
    # Goodness of fit
    gof = validator.goodness_of_fit(empirical_moments)
    print(f"\nGoodness of Fit Score: {gof['fit_score']:.2f}")
    print(f"Overall Assessment: {gof['overall_assessment']}")
    
    # Plot results
    validator.plot_moment_comparison('moment_comparison.png')
    
    # Generate report
    validator.generate_validation_report('validation_report.txt')
    
    print("\nEmpirical validation complete!")
    
    return validator


if __name__ == '__main__':
    validator = example_empirical_validation()
