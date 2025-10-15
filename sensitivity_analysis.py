"""
Sensitivity Analysis Module for Labor Market ABM

This module provides tools for conducting sensitivity analysis on the labor market model:
1. Sobol sensitivity analysis (variance-based)
2. One-At-A-Time (OAT) sensitivity analysis
3. Morris screening method
4. Parameter importance ranking
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from model_class import Model
from multiprocessing import Pool
import csv
from itertools import product


class SensitivityAnalysis:
    """
    Main class for conducting sensitivity analysis on the labor market ABM
    """
    
    def __init__(self, param_ranges, output_metrics=None, T=500, burnin=200):
        """
        Initialize sensitivity analysis
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary with parameter names as keys and (min, max) tuples as values
        output_metrics : list
            List of metric names to analyze (default: unemployment, wages, GDP)
        T : int
            Total simulation periods
        burnin : int
            Burn-in periods to exclude from analysis
        """
        self.param_ranges = param_ranges
        self.T = T
        self.burnin = burnin
        
        if output_metrics is None:
            self.output_metrics = [
                'mean_unemployment',
                'unemployment_volatility', 
                'mean_real_wage',
                'wage_inequality',
                'mean_gdp',
                'gdp_volatility',
                'skill_mismatch_rate'
            ]
        else:
            self.output_metrics = output_metrics
            
        self.results = {}
        
    def run_model_with_params(self, params):
        """
        Run model with given parameter values
        
        Parameters:
        -----------
        params : dict
            Parameter values to use
            
        Returns:
        --------
        dict : Dictionary of output metrics
        """
        # Create and run model
        m = Model(T=self.T, **params)
        m.run()
        
        # Extract metrics
        t_start = self.burnin
        metrics = {}
        
        # Unemployment metrics
        metrics['mean_unemployment'] = np.mean(m.u_r_arr[t_start:])
        metrics['unemployment_volatility'] = np.std(m.u_r_arr[t_start:])
        
        # Wage metrics
        metrics['mean_real_wage'] = np.mean(m.mean_w_arr[t_start:])
        metrics['wage_inequality'] = np.mean(m.nine_to_one[t_start:])
        
        # GDP metrics
        metrics['mean_gdp'] = np.mean(m.GDP[t_start:])
        metrics['gdp_volatility'] = np.std(m.GDP[t_start:])
        
        # Skill mismatch metrics
        metrics['skill_mismatch_rate'] = np.mean(m.share_nr_in_r[t_start:]) / m.H_nr
        
        return metrics
    
    def oat_sensitivity(self, base_params, n_samples=10):
        """
        One-At-A-Time sensitivity analysis
        
        Parameters:
        -----------
        base_params : dict
            Baseline parameter values
        n_samples : int
            Number of samples for each parameter
            
        Returns:
        --------
        dict : Sensitivity results for each parameter
        """
        print("Running OAT Sensitivity Analysis...")
        results = {}
        
        for param_name, (param_min, param_max) in self.param_ranges.items():
            print(f"  Analyzing parameter: {param_name}")
            param_values = np.linspace(param_min, param_max, n_samples)
            param_results = {metric: [] for metric in self.output_metrics}
            
            for param_val in param_values:
                params = base_params.copy()
                params[param_name] = param_val
                
                metrics = self.run_model_with_params(params)
                
                for metric in self.output_metrics:
                    param_results[metric].append(metrics[metric])
            
            results[param_name] = {
                'values': param_values,
                'metrics': param_results
            }
        
        self.results['oat'] = results
        return results
    
    def morris_screening(self, base_params, n_trajectories=10, n_levels=4):
        """
        Morris screening method for identifying important parameters
        
        Parameters:
        -----------
        base_params : dict
            Baseline parameter values
        n_trajectories : int
            Number of trajectories to sample
        n_levels : int
            Number of levels for each parameter
            
        Returns:
        --------
        dict : Morris sensitivity indices (mu, sigma) for each parameter
        """
        print("Running Morris Screening Method...")
        
        param_names = list(self.param_ranges.keys())
        k = len(param_names)
        
        # Morris indices
        morris_indices = {param: {'mu': {}, 'sigma': {}} for param in param_names}
        
        for traj in range(n_trajectories):
            print(f"  Trajectory {traj+1}/{n_trajectories}")
            
            # Generate base point
            base_point = {}
            for param in param_names:
                pmin, pmax = self.param_ranges[param]
                base_point[param] = rd.uniform(pmin, pmax)
            
            # Merge with other params
            params = base_params.copy()
            params.update(base_point)
            
            # Run base model
            base_metrics = self.run_model_with_params(params)
            
            # Perturb each parameter
            for param in param_names:
                perturbed_params = params.copy()
                pmin, pmax = self.param_ranges[param]
                delta = (pmax - pmin) / (n_levels - 1)
                perturbed_params[param] += delta
                perturbed_params[param] = np.clip(perturbed_params[param], pmin, pmax)
                
                perturbed_metrics = self.run_model_with_params(perturbed_params)
                
                # Calculate elementary effects
                for metric in self.output_metrics:
                    ee = (perturbed_metrics[metric] - base_metrics[metric]) / delta
                    
                    if metric not in morris_indices[param]['mu']:
                        morris_indices[param]['mu'][metric] = []
                        morris_indices[param]['sigma'][metric] = []
                    
                    morris_indices[param]['mu'][metric].append(ee)
        
        # Calculate final statistics
        for param in param_names:
            for metric in self.output_metrics:
                ees = morris_indices[param]['mu'][metric]
                morris_indices[param]['mu'][metric] = np.mean(np.abs(ees))
                morris_indices[param]['sigma'][metric] = np.std(ees)
        
        self.results['morris'] = morris_indices
        return morris_indices
    
    def monte_carlo_analysis(self, base_params, n_samples=100):
        """
        Monte Carlo analysis for global sensitivity
        
        Parameters:
        -----------
        base_params : dict
            Baseline parameter values
        n_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        dict : Results from all samples
        """
        print(f"Running Monte Carlo Analysis with {n_samples} samples...")
        
        samples = []
        results = {metric: [] for metric in self.output_metrics}
        
        for i in range(n_samples):
            if (i+1) % 10 == 0:
                print(f"  Sample {i+1}/{n_samples}")
            
            # Sample parameters
            sample_params = base_params.copy()
            for param, (pmin, pmax) in self.param_ranges.items():
                sample_params[param] = rd.uniform(pmin, pmax)
            
            samples.append(sample_params.copy())
            
            # Run model
            metrics = self.run_model_with_params(sample_params)
            
            for metric in self.output_metrics:
                results[metric].append(metrics[metric])
        
        self.results['monte_carlo'] = {
            'samples': samples,
            'results': results
        }
        
        return results
    
    def plot_oat_results(self, save_path='sensitivity_oat.png'):
        """
        Plot OAT sensitivity analysis results
        """
        if 'oat' not in self.results:
            print("No OAT results to plot. Run oat_sensitivity first.")
            return
        
        oat_results = self.results['oat']
        n_params = len(oat_results)
        n_metrics = len(self.output_metrics)
        
        fig, axes = plt.subplots(n_metrics, n_params, figsize=(5*n_params, 4*n_metrics))
        
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(self.output_metrics):
            for j, (param_name, param_data) in enumerate(oat_results.items()):
                ax = axes[i, j]
                ax.plot(param_data['values'], param_data['metrics'][metric], 'o-')
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{metric} vs {param_name}")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"OAT plot saved to {save_path}")
        return fig
    
    def plot_morris_results(self, save_path='sensitivity_morris.png'):
        """
        Plot Morris screening results (mu* vs sigma)
        """
        if 'morris' not in self.results:
            print("No Morris results to plot. Run morris_screening first.")
            return
        
        morris_results = self.results['morris']
        n_metrics = len(self.output_metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(self.output_metrics):
            ax = axes[i]
            
            for param in morris_results.keys():
                mu = morris_results[param]['mu'][metric]
                sigma = morris_results[param]['sigma'][metric]
                ax.scatter(mu, sigma, s=100, alpha=0.6)
                ax.annotate(param, (mu, sigma), fontsize=8)
            
            ax.set_xlabel('μ* (mean of absolute elementary effects)')
            ax.set_ylabel('σ (std of elementary effects)')
            ax.set_title(f'Morris Screening: {metric}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Morris plot saved to {save_path}")
        return fig
    
    def save_results(self, filename='sensitivity_results.csv'):
        """
        Save sensitivity analysis results to CSV
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write OAT results
            if 'oat' in self.results:
                writer.writerow(['OAT Sensitivity Analysis'])
                writer.writerow([])
                
                for param_name, param_data in self.results['oat'].items():
                    writer.writerow([f'Parameter: {param_name}'])
                    writer.writerow(['Value'] + self.output_metrics)
                    
                    for i, val in enumerate(param_data['values']):
                        row = [val]
                        for metric in self.output_metrics:
                            row.append(param_data['metrics'][metric][i])
                        writer.writerow(row)
                    writer.writerow([])
            
            # Write Morris results
            if 'morris' in self.results:
                writer.writerow(['Morris Screening Results'])
                writer.writerow([])
                writer.writerow(['Parameter', 'Metric', 'μ*', 'σ'])
                
                for param, indices in self.results['morris'].items():
                    for metric in self.output_metrics:
                        writer.writerow([
                            param, 
                            metric,
                            indices['mu'][metric],
                            indices['sigma'][metric]
                        ])
                writer.writerow([])
        
        print(f"Results saved to {filename}")


def example_sensitivity_analysis():
    """
    Example usage of sensitivity analysis
    """
    # Define parameter ranges to test
    param_ranges = {
        'sigma_m': (0.001, 0.01),      # Price adjustment volatility
        'sigma_w': (0.001, 0.01),      # Wage adjustment volatility  
        'lambda_LM': (1, 10),          # Labor market matching intensity
        'mu_r': (0.2, 0.5),            # Routine labor share
        'gamma_nr': (0.2, 0.5),        # Non-routine worker share
    }
    
    # Base parameters
    base_params = {
        'H': 200,
        'F': 20, 
        'T': 500,
        'alpha_2': 0.25,
        'chi_C': 0.2,
        'sigma_delta': 0.001,
        'nu': 0.1,
        'u_r': 0.08,
        'beta': 1,
        'lambda_exp': 0.5,
        'N_app': 4,
        'sigma': 1.5,
        'nr_to_r': False,
        'a': 100,
        'min_w_par': 0.3,
        'W_r': 1,
        'f_max': 3
    }
    
    # Create analyzer
    sa = SensitivityAnalysis(param_ranges, T=500, burnin=200)
    
    # Run OAT analysis
    oat_results = sa.oat_sensitivity(base_params, n_samples=5)
    sa.plot_oat_results('sensitivity_oat.png')
    
    # Run Morris screening  
    morris_results = sa.morris_screening(base_params, n_trajectories=5, n_levels=4)
    sa.plot_morris_results('sensitivity_morris.png')
    
    # Save results
    sa.save_results('sensitivity_results.csv')
    
    print("\nSensitivity analysis complete!")
    return sa


if __name__ == '__main__':
    # Run example
    sa = example_sensitivity_analysis()
