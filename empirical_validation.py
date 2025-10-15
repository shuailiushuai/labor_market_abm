"""
Empirical Validation Module for Labor Market ABM

This module provides tools for validating the model against empirical data:
- Statistical moment comparison
- Beveridge curve validation
- Wage distribution validation
- Unemployment dynamics validation
- Time series comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from model_class import Model
import warnings
warnings.filterwarnings('ignore')


class EmpiricalValidator:
    """
    Class for validating Labor Market ABM against empirical data
    """
    
    def __init__(self, model_results=None):
        """
        Initialize empirical validator
        
        Parameters:
        -----------
        model_results : dict, optional
            Dictionary containing model simulation results
        """
        self.model_results = model_results
        self.validation_results = {}
        
    def set_model_results(self, model):
        """
        Extract relevant results from a model instance
        
        Parameters:
        -----------
        model : Model
            Trained model instance
        """
        self.model_results = {
            'unemployment_rate': model.u_r_arr,
            'GDP': model.GDP,
            'mean_wages': model.mean_w_arr,
            'mean_r_wages': model.mean_r_w_arr,
            'mean_nr_wages': model.mean_nr_w_arr,
            'wage_variance': model.wage_variance,
            'open_vacancies': model.open_vs,
            'mean_prices': model.mean_p_arr,
            'share_inactive': model.share_inactive,
            'nine_to_one': model.nine_to_one,
            'T': model.t,
            'H': model.H,
            'F': model.F,
        }
        
    def compare_moments(self, empirical_data, model_data, data_name):
        """
        Compare statistical moments between empirical and model data
        
        Parameters:
        -----------
        empirical_data : array-like
            Empirical data
        model_data : array-like
            Model simulation data
        data_name : str
            Name of the variable being compared
            
        Returns:
        --------
        dict : Dictionary of moment comparisons
        """
        moments = {}
        
        # Mean
        emp_mean = np.mean(empirical_data)
        mod_mean = np.mean(model_data)
        moments['mean_empirical'] = emp_mean
        moments['mean_model'] = mod_mean
        moments['mean_diff'] = mod_mean - emp_mean
        moments['mean_rel_error'] = (mod_mean - emp_mean) / emp_mean if emp_mean != 0 else np.nan
        
        # Standard deviation
        emp_std = np.std(empirical_data)
        mod_std = np.std(model_data)
        moments['std_empirical'] = emp_std
        moments['std_model'] = mod_std
        moments['std_diff'] = mod_std - emp_std
        moments['std_rel_error'] = (mod_std - emp_std) / emp_std if emp_std != 0 else np.nan
        
        # Skewness
        emp_skew = stats.skew(empirical_data)
        mod_skew = stats.skew(model_data)
        moments['skew_empirical'] = emp_skew
        moments['skew_model'] = mod_skew
        moments['skew_diff'] = mod_skew - emp_skew
        
        # Kurtosis
        emp_kurt = stats.kurtosis(empirical_data)
        mod_kurt = stats.kurtosis(model_data)
        moments['kurt_empirical'] = emp_kurt
        moments['kurt_model'] = mod_kurt
        moments['kurt_diff'] = mod_kurt - emp_kurt
        
        return moments
    
    def validate_wage_distribution(self, empirical_wages, model_wages=None, bins=30):
        """
        Validate wage distribution using Kolmogorov-Smirnov test and visual comparison
        
        Parameters:
        -----------
        empirical_wages : array-like
            Empirical wage data
        model_wages : array-like, optional
            Model wage data. If None, uses stored model results.
        bins : int
            Number of bins for histogram
            
        Returns:
        --------
        dict : Validation results including KS statistic and p-value
        """
        if model_wages is None:
            if self.model_results is None or 'mean_wages' not in self.model_results:
                raise ValueError("No model wage data available")
            # Use final period wages as proxy (in real validation, use full distribution)
            model_wages = self.model_results['mean_wages']
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(empirical_wages, model_wages)
        
        # Anderson-Darling test (if same size)
        try:
            ad_result = stats.anderson_ksamp([empirical_wages, model_wages])
            ad_stat = ad_result.statistic
            ad_pvalue = ad_result.pvalue if hasattr(ad_result, 'pvalue') else None
        except:
            ad_stat = None
            ad_pvalue = None
        
        results = {
            'ks_statistic': ks_stat,
            'ks_pvalue': p_value,
            'ad_statistic': ad_stat,
            'ad_pvalue': ad_pvalue,
            'passes_ks_test': p_value > 0.05,
        }
        
        # Compare moments
        moments = self.compare_moments(empirical_wages, model_wages, 'wages')
        results.update(moments)
        
        self.validation_results['wage_distribution'] = results
        return results
    
    def validate_beveridge_curve(self, emp_unemployment, emp_vacancies, 
                                 model_unemployment=None, model_vacancies=None):
        """
        Validate the Beveridge curve relationship
        
        Parameters:
        -----------
        emp_unemployment : array-like
            Empirical unemployment rates
        emp_vacancies : array-like
            Empirical vacancy rates
        model_unemployment : array-like, optional
            Model unemployment rates
        model_vacancies : array-like, optional
            Model vacancy rates
            
        Returns:
        --------
        dict : Validation results including correlation and curve fit
        """
        if model_unemployment is None or model_vacancies is None:
            if self.model_results is None:
                raise ValueError("No model results available")
            model_unemployment = self.model_results['unemployment_rate']
            model_vacancies = self.model_results['open_vacancies']
        
        # Calculate correlations
        emp_corr = np.corrcoef(emp_unemployment, emp_vacancies)[0, 1]
        mod_corr = np.corrcoef(model_unemployment, model_vacancies)[0, 1]
        
        # Fit power law: v = a * u^b
        def power_law(u, a, b):
            return a * np.power(u, b)
        
        try:
            emp_params, _ = curve_fit(power_law, emp_unemployment, emp_vacancies, 
                                     p0=[1, -1], maxfev=5000)
            mod_params, _ = curve_fit(power_law, model_unemployment, model_vacancies, 
                                     p0=[1, -1], maxfev=5000)
        except:
            emp_params = [np.nan, np.nan]
            mod_params = [np.nan, np.nan]
        
        results = {
            'emp_correlation': emp_corr,
            'model_correlation': mod_corr,
            'correlation_diff': mod_corr - emp_corr,
            'emp_power_law_a': emp_params[0],
            'emp_power_law_b': emp_params[1],
            'model_power_law_a': mod_params[0],
            'model_power_law_b': mod_params[1],
            'negative_correlation': mod_corr < 0,
        }
        
        self.validation_results['beveridge_curve'] = results
        return results
    
    def validate_unemployment_dynamics(self, emp_unemployment, model_unemployment=None, 
                                      max_lag=12):
        """
        Validate unemployment dynamics using autocorrelation
        
        Parameters:
        -----------
        emp_unemployment : array-like
            Empirical unemployment rates
        model_unemployment : array-like, optional
            Model unemployment rates
        max_lag : int
            Maximum lag for autocorrelation
            
        Returns:
        --------
        dict : Validation results including autocorrelation comparison
        """
        if model_unemployment is None:
            if self.model_results is None:
                raise ValueError("No model results available")
            model_unemployment = self.model_results['unemployment_rate']
        
        # Calculate autocorrelations
        emp_acf = [np.corrcoef(emp_unemployment[:-lag], emp_unemployment[lag:])[0, 1] 
                   for lag in range(1, min(max_lag+1, len(emp_unemployment)//2))]
        mod_acf = [np.corrcoef(model_unemployment[:-lag], model_unemployment[lag:])[0, 1] 
                   for lag in range(1, min(max_lag+1, len(model_unemployment)//2))]
        
        # Calculate mean squared error in autocorrelations
        min_len = min(len(emp_acf), len(mod_acf))
        acf_mse = np.mean([(emp_acf[i] - mod_acf[i])**2 for i in range(min_len)])
        
        results = {
            'emp_acf': emp_acf,
            'model_acf': mod_acf,
            'acf_mse': acf_mse,
            'emp_persistence': emp_acf[0] if len(emp_acf) > 0 else np.nan,
            'model_persistence': mod_acf[0] if len(mod_acf) > 0 else np.nan,
        }
        
        self.validation_results['unemployment_dynamics'] = results
        return results
    
    def validate_wage_phillips_curve(self, emp_unemployment, emp_wage_growth,
                                     model_unemployment=None, model_wage_growth=None):
        """
        Validate the wage Phillips curve relationship
        
        Parameters:
        -----------
        emp_unemployment : array-like
            Empirical unemployment rates
        emp_wage_growth : array-like
            Empirical wage growth rates
        model_unemployment : array-like, optional
            Model unemployment rates
        model_wage_growth : array-like, optional
            Model wage growth rates
            
        Returns:
        --------
        dict : Validation results
        """
        if model_unemployment is None or model_wage_growth is None:
            if self.model_results is None:
                raise ValueError("No model results available")
            model_unemployment = self.model_results['unemployment_rate']
            wages = self.model_results['mean_wages']
            model_wage_growth = np.diff(wages) / wages[:-1]
            model_unemployment = model_unemployment[1:]  # Align lengths
        
        # Calculate correlations
        emp_corr = np.corrcoef(emp_unemployment, emp_wage_growth)[0, 1]
        mod_corr = np.corrcoef(model_unemployment[:len(model_wage_growth)], 
                              model_wage_growth)[0, 1]
        
        # Fit linear relationship
        emp_slope = np.polyfit(emp_unemployment, emp_wage_growth, 1)[0]
        mod_slope = np.polyfit(model_unemployment[:len(model_wage_growth)], 
                              model_wage_growth, 1)[0]
        
        results = {
            'emp_correlation': emp_corr,
            'model_correlation': mod_corr,
            'emp_slope': emp_slope,
            'model_slope': mod_slope,
            'negative_correlation': mod_corr < 0,
            'slope_diff': mod_slope - emp_slope,
        }
        
        self.validation_results['phillips_curve'] = results
        return results
    
    def plot_validation_results(self, validation_type='beveridge_curve', 
                               emp_data=None, save_path=None):
        """
        Plot validation results
        
        Parameters:
        -----------
        validation_type : str
            Type of validation to plot
        emp_data : dict, optional
            Dictionary containing empirical data
        save_path : str, optional
            Path to save the figure
        """
        if validation_type == 'beveridge_curve':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            if emp_data:
                ax1.scatter(emp_data['unemployment'], emp_data['vacancies'], 
                          alpha=0.6, label='Empirical', s=50)
                ax1.set_xlabel('Unemployment Rate', fontsize=12)
                ax1.set_ylabel('Vacancy Rate / Open Vacancies', fontsize=12)
                ax1.set_title('Empirical Beveridge Curve', fontsize=14)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            if self.model_results:
                ax2.scatter(self.model_results['unemployment_rate'], 
                          self.model_results['open_vacancies'],
                          alpha=0.6, label='Model', color='orange', s=50)
                ax2.set_xlabel('Unemployment Rate', fontsize=12)
                ax2.set_ylabel('Open Vacancies', fontsize=12)
                ax2.set_title('Model Beveridge Curve', fontsize=14)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        elif validation_type == 'wage_distribution':
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            if emp_data and 'wages' in emp_data:
                axes[0].hist(emp_data['wages'], bins=30, alpha=0.7, 
                           label='Empirical', density=True, color='blue')
                axes[0].set_xlabel('Wage', fontsize=12)
                axes[0].set_ylabel('Density', fontsize=12)
                axes[0].set_title('Empirical Wage Distribution', fontsize=14)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            if self.model_results and 'mean_wages' in self.model_results:
                axes[1].hist(self.model_results['mean_wages'], bins=30, alpha=0.7,
                           label='Model', density=True, color='orange')
                axes[1].set_xlabel('Wage', fontsize=12)
                axes[1].set_ylabel('Density', fontsize=12)
                axes[1].set_title('Model Wage Distribution', fontsize=14)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        elif validation_type == 'unemployment_dynamics':
            if 'unemployment_dynamics' not in self.validation_results:
                print("No unemployment dynamics validation results available")
                return None
            
            results = self.validation_results['unemployment_dynamics']
            emp_acf = results['emp_acf']
            mod_acf = results['model_acf']
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            lags = range(1, len(emp_acf) + 1)
            ax.plot(lags, emp_acf, 'o-', label='Empirical', linewidth=2)
            ax.plot(lags, mod_acf[:len(emp_acf)], 's-', label='Model', linewidth=2)
            ax.set_xlabel('Lag', fontsize=12)
            ax.set_ylabel('Autocorrelation', fontsize=12)
            ax.set_title('Unemployment Rate Autocorrelation', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_validation_report(self, output_file='validation_report.txt'):
        """
        Generate comprehensive validation report
        
        Parameters:
        -----------
        output_file : str
            Path to save the report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EMPIRICAL VALIDATION REPORT\n")
            f.write("Labor Market Agent-Based Model\n")
            f.write("=" * 80 + "\n\n")
            
            for val_type, results in self.validation_results.items():
                f.write(f"\n{val_type.upper().replace('_', ' ')}\n")
                f.write("-" * 80 + "\n")
                
                for key, value in results.items():
                    if isinstance(value, (list, np.ndarray)):
                        continue  # Skip array values
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("VALIDATION SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            # Summary statistics
            if 'wage_distribution' in self.validation_results:
                wage_val = self.validation_results['wage_distribution']
                f.write(f"Wage Distribution KS Test: ")
                f.write(f"{'PASS' if wage_val.get('passes_ks_test', False) else 'FAIL'}\n")
                f.write(f"  p-value: {wage_val.get('ks_pvalue', 'N/A')}\n\n")
            
            if 'beveridge_curve' in self.validation_results:
                bev_val = self.validation_results['beveridge_curve']
                f.write(f"Beveridge Curve Negative Correlation: ")
                f.write(f"{'YES' if bev_val.get('negative_correlation', False) else 'NO'}\n")
                f.write(f"  Model correlation: {bev_val.get('model_correlation', 'N/A')}\n\n")
        
        print(f"Validation report saved to {output_file}")


# Example usage
if __name__ == "__main__":
    print("Empirical Validation Module for Labor Market ABM")
    print("=" * 50)
    
    # Run a model simulation
    print("\nRunning model simulation...")
    import numpy.random as rd
    rd.seed(42)
    
    m = Model(T=500, H=500, F=80, alpha_2=0.25, lambda_LM=5)
    m.run()
    
    # Initialize validator with model results
    validator = EmpiricalValidator()
    validator.set_model_results(m)
    
    # Generate synthetic "empirical" data for demonstration
    print("\nGenerating synthetic empirical data for demonstration...")
    burn_in = 200
    emp_unemployment = np.random.normal(0.08, 0.02, 200)
    emp_vacancies = 50 - 200 * emp_unemployment + np.random.normal(0, 5, 200)
    emp_wages = np.random.lognormal(0, 0.3, 500)
    
    # Validate Beveridge curve
    print("\nValidating Beveridge curve...")
    bev_results = validator.validate_beveridge_curve(emp_unemployment, emp_vacancies)
    print(f"Empirical correlation: {bev_results['emp_correlation']:.3f}")
    print(f"Model correlation: {bev_results['model_correlation']:.3f}")
    
    # Validate wage distribution
    print("\nValidating wage distribution...")
    wage_results = validator.validate_wage_distribution(emp_wages)
    print(f"KS statistic: {wage_results['ks_statistic']:.3f}")
    print(f"KS p-value: {wage_results['ks_pvalue']:.3f}")
    
    # Generate validation report
    validator.generate_validation_report('results/validation_report.txt')
    
    print("\nEmpirical validation completed!")
