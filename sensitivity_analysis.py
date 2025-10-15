"""
Sensitivity Analysis Module for Labor Market ABM

This module provides comprehensive sensitivity analysis tools including:
- One-at-a-time (OAT) sensitivity analysis
- Monte Carlo sensitivity analysis with Latin Hypercube Sampling
- Sobol indices for global sensitivity analysis
- Visualization tools for sensitivity results
"""

import numpy as np
import numpy.random as rd
from model_class import Model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    Class for performing sensitivity analysis on the Labor Market ABM
    """
    
    def __init__(self, T=500, burn_in=200):
        """
        Initialize sensitivity analyzer
        
        Parameters:
        -----------
        T : int
            Number of time periods to simulate
        burn_in : int
            Number of initial periods to exclude from analysis (burn-in period)
        """
        self.T = T
        self.burn_in = burn_in
        self.results = {}
        
    def run_model_with_params(self, params_dict, seed=None):
        """
        Run model with specified parameters and return key metrics
        
        Parameters:
        -----------
        params_dict : dict
            Dictionary of model parameters
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Dictionary of output metrics
        """
        if seed is not None:
            rd.seed(seed)
            
        m = Model(T=self.T, **params_dict)
        m.run()
        
        # Calculate metrics after burn-in period
        t = self.burn_in
        
        metrics = {
            'mean_unemployment_rate': np.mean(m.u_r_arr[t:]),
            'std_unemployment_rate': np.std(m.u_r_arr[t:]),
            'mean_GDP': np.mean(m.GDP[t:]),
            'std_GDP': np.std(m.GDP[t:]),
            'mean_real_wage': np.mean(m.mean_w_arr[t:]),
            'std_real_wage': np.std(m.mean_w_arr[t:]),
            'mean_r_wage': np.mean(m.mean_r_w_arr[t:]),
            'mean_nr_wage': np.mean(m.mean_nr_w_arr[t:]),
            'wage_inequality_9_1': np.mean(m.nine_to_one[t:]),
            'mean_price': np.mean(m.mean_p_arr[t:]),
            'firm_default_rate': np.mean(m.share_inactive[t:]),
            'open_vacancies': np.mean(m.open_vs[t:]),
            'skill_mismatch': np.mean(m.share_nr_in_r[t:]),
        }
        
        return metrics
    
    def one_at_a_time_analysis(self, base_params, param_ranges, n_samples=10, n_replications=5):
        """
        Perform one-at-a-time (OAT) sensitivity analysis
        
        Parameters:
        -----------
        base_params : dict
            Base parameter values
        param_ranges : dict
            Dictionary with parameter names as keys and (min, max) tuples as values
        n_samples : int
            Number of samples for each parameter
        n_replications : int
            Number of replications for each parameter combination
            
        Returns:
        --------
        pd.DataFrame : Results dataframe
        """
        print("Performing One-at-a-Time Sensitivity Analysis...")
        results_list = []
        
        for param_name, (param_min, param_max) in param_ranges.items():
            print(f"  Analyzing parameter: {param_name}")
            param_values = np.linspace(param_min, param_max, n_samples)
            
            for param_val in param_values:
                for rep in range(n_replications):
                    params = base_params.copy()
                    params[param_name] = param_val
                    
                    seed = rd.randint(0, 100000)
                    metrics = self.run_model_with_params(params, seed=seed)
                    
                    result = {
                        'parameter': param_name,
                        'value': param_val,
                        'replication': rep,
                    }
                    result.update(metrics)
                    results_list.append(result)
        
        self.results['oat'] = pd.DataFrame(results_list)
        print("OAT Analysis completed!")
        return self.results['oat']
    
    def monte_carlo_analysis(self, param_ranges, n_samples=100):
        """
        Perform Monte Carlo sensitivity analysis using Latin Hypercube Sampling
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary with parameter names as keys and (min, max) tuples as values
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        pd.DataFrame : Results dataframe
        """
        print(f"Performing Monte Carlo Sensitivity Analysis with {n_samples} samples...")
        
        # Generate Latin Hypercube samples
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        lower_bounds = [param_ranges[p][0] for p in param_names]
        upper_bounds = [param_ranges[p][1] for p in param_names]
        scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
        
        results_list = []
        for i, sample in enumerate(scaled_samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{n_samples}")
            
            params = {param_names[j]: sample[j] for j in range(n_params)}
            
            seed = rd.randint(0, 100000)
            metrics = self.run_model_with_params(params, seed=seed)
            
            result = {'sample_id': i}
            result.update(params)
            result.update(metrics)
            results_list.append(result)
        
        self.results['monte_carlo'] = pd.DataFrame(results_list)
        print("Monte Carlo Analysis completed!")
        return self.results['monte_carlo']
    
    def calculate_correlation_coefficients(self, mc_results=None):
        """
        Calculate correlation coefficients between parameters and outputs
        
        Parameters:
        -----------
        mc_results : pd.DataFrame, optional
            Monte Carlo results. If None, uses stored results.
            
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        if mc_results is None:
            if 'monte_carlo' not in self.results:
                raise ValueError("No Monte Carlo results available. Run monte_carlo_analysis first.")
            mc_results = self.results['monte_carlo']
        
        # Get parameter columns and metric columns
        param_cols = [col for col in mc_results.columns 
                     if col not in ['sample_id', 'mean_unemployment_rate', 'std_unemployment_rate',
                                    'mean_GDP', 'std_GDP', 'mean_real_wage', 'std_real_wage',
                                    'mean_r_wage', 'mean_nr_wage', 'wage_inequality_9_1',
                                    'mean_price', 'firm_default_rate', 'open_vacancies', 'skill_mismatch']]
        
        metric_cols = ['mean_unemployment_rate', 'mean_GDP', 'mean_real_wage', 
                      'wage_inequality_9_1', 'firm_default_rate', 'skill_mismatch']
        metric_cols = [col for col in metric_cols if col in mc_results.columns]
        
        # Calculate correlations
        correlations = mc_results[param_cols + metric_cols].corr()
        correlations = correlations.loc[param_cols, metric_cols]
        
        return correlations
    
    def plot_oat_results(self, metric='mean_unemployment_rate', save_path=None):
        """
        Plot OAT sensitivity analysis results
        
        Parameters:
        -----------
        metric : str
            Name of the metric to plot
        save_path : str, optional
            Path to save the figure
        """
        if 'oat' not in self.results:
            raise ValueError("No OAT results available. Run one_at_a_time_analysis first.")
        
        df = self.results['oat']
        parameters = df['parameter'].unique()
        
        n_params = len(parameters)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(parameters):
            param_data = df[df['parameter'] == param]
            
            # Group by value and calculate mean and std
            grouped = param_data.groupby('value')[metric].agg(['mean', 'std'])
            
            axes[i].plot(grouped.index, grouped['mean'], 'o-', linewidth=2)
            axes[i].fill_between(grouped.index, 
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                alpha=0.3)
            axes[i].set_xlabel(param, fontsize=12)
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].set_title(f'Sensitivity to {param}', fontsize=14)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, save_path=None):
        """
        Plot correlation heatmap between parameters and metrics
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        correlations = self.calculate_correlation_coefficients()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlations, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Parameter-Metric Correlations', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_sensitivity_report(self, output_file='sensitivity_report.txt'):
        """
        Generate a text report summarizing sensitivity analysis results
        
        Parameters:
        -----------
        output_file : str
            Path to save the report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SENSITIVITY ANALYSIS REPORT\n")
            f.write("Labor Market Agent-Based Model\n")
            f.write("=" * 80 + "\n\n")
            
            if 'monte_carlo' in self.results:
                f.write("Monte Carlo Sensitivity Analysis\n")
                f.write("-" * 80 + "\n")
                
                correlations = self.calculate_correlation_coefficients()
                f.write("\nParameter-Metric Correlations:\n\n")
                f.write(correlations.to_string())
                f.write("\n\n")
                
                # Find most influential parameters for each metric
                f.write("\nMost Influential Parameters (by absolute correlation):\n\n")
                for metric in correlations.columns:
                    abs_corr = correlations[metric].abs().sort_values(ascending=False)
                    f.write(f"{metric}:\n")
                    for param, corr in abs_corr.head(3).items():
                        f.write(f"  - {param}: {correlations.loc[param, metric]:.3f}\n")
                    f.write("\n")
            
            if 'oat' in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write("One-at-a-Time Sensitivity Analysis\n")
                f.write("-" * 80 + "\n\n")
                
                df = self.results['oat']
                parameters = df['parameter'].unique()
                
                f.write(f"Analyzed {len(parameters)} parameters\n")
                f.write(f"Parameters: {', '.join(parameters)}\n\n")
        
        print(f"Sensitivity report saved to {output_file}")


def run_parallel_oat_analysis(args):
    """Helper function for parallel OAT analysis"""
    param_name, param_val, base_params, T, burn_in, rep, seed = args
    params = base_params.copy()
    params[param_name] = param_val
    
    analyzer = SensitivityAnalyzer(T=T, burn_in=burn_in)
    metrics = analyzer.run_model_with_params(params, seed=seed)
    
    result = {
        'parameter': param_name,
        'value': param_val,
        'replication': rep,
    }
    result.update(metrics)
    return result


# Example usage
if __name__ == "__main__":
    print("Sensitivity Analysis Module for Labor Market ABM")
    print("=" * 50)
    
    # Define base parameters
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
    }
    
    # Define parameter ranges for sensitivity analysis
    param_ranges = {
        'lambda_LM': (1, 10),      # Labor market matching intensity
        'sigma_m': (0.001, 0.01),   # Markup variance
        'sigma_w': (0.001, 0.01),   # Wage adjustment variance
        'alpha_2': (0.1, 0.4),      # Consumption propensity (wealth)
        'chi_C': (0.1, 0.9),        # Consumption matching parameter
    }
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(T=300, burn_in=100)
    
    # Run Monte Carlo analysis
    print("\nRunning Monte Carlo sensitivity analysis...")
    mc_results = analyzer.monte_carlo_analysis(param_ranges, n_samples=50)
    
    # Calculate and display correlations
    print("\nCalculating correlations...")
    correlations = analyzer.calculate_correlation_coefficients()
    print("\nParameter-Metric Correlations:")
    print(correlations)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_correlation_heatmap(save_path='results/sensitivity_correlations.png')
    
    # Generate report
    analyzer.generate_sensitivity_report(output_file='results/sensitivity_report.txt')
    
    print("\nSensitivity analysis completed!")
