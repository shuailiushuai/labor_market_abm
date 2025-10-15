"""
Parameter Estimation Module for Labor Market ABM

This module provides methods for estimating model parameters:
1. Maximum Likelihood Estimation (MLE)
2. Method of Simulated Moments (MSM)  
3. Approximate Bayesian Computation (ABC)
4. Grid search and optimization
"""

import numpy as np
import numpy.random as rd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from model_class import Model
import pandas as pd


class ParameterEstimator:
    """
    Parameter estimator for the labor market ABM
    """
    
    def __init__(self, empirical_data, param_bounds, T=500, burnin=100):
        """
        Initialize estimator
        
        Parameters:
        -----------
        empirical_data : dict
            Empirical moments or data to match
        param_bounds : dict
            Dictionary with parameter names as keys and (min, max) tuples
        T : int
            Simulation periods
        burnin : int
            Burn-in periods
        """
        self.empirical_data = empirical_data
        self.param_bounds = param_bounds
        self.T = T
        self.burnin = burnin
        
        self.best_params = None
        self.estimation_results = {}
    
    def simulate_moments(self, params, base_params):
        """
        Simulate model and compute moments
        
        Parameters:
        -----------
        params : dict or array
            Parameter values to test
        base_params : dict
            Other fixed parameters
            
        Returns:
        --------
        dict : Simulated moments
        """
        # Combine params
        if isinstance(params, np.ndarray):
            # Convert array to dict
            param_dict = {}
            for i, (name, _) in enumerate(self.param_bounds.items()):
                param_dict[name] = params[i]
        else:
            param_dict = params
        
        full_params = {**base_params, **param_dict}
        
        # Run model
        try:
            m = Model(T=self.T, **full_params)
            m.run()
            
            # Compute moments
            t_start = self.burnin
            moments = {
                'mean_unemployment': np.mean(m.u_r_arr[t_start:]),
                'std_unemployment': np.std(m.u_r_arr[t_start:]),
                'mean_real_wage': np.mean(m.mean_w_arr[t_start:]),
                'std_real_wage': np.std(m.mean_w_arr[t_start:]),
                'wage_inequality': np.mean(m.nine_to_one[t_start:]),
                'skill_mismatch': np.mean(m.share_nr_in_r[t_start:]) / m.H_nr,
                'corr_gdp_unemp': np.corrcoef(m.GDP[t_start:], m.u_r_arr[t_start:])[0, 1],
            }
            
            return moments
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            # Return large penalty
            return {k: np.nan for k in self.empirical_data.keys()}
    
    def msm_objective(self, params, base_params, weights=None):
        """
        Method of Simulated Moments objective function
        
        Parameters:
        -----------
        params : array
            Parameter values
        base_params : dict
            Fixed parameters
        weights : dict
            Weights for each moment
            
        Returns:
        --------
        float : Weighted squared distance
        """
        simulated = self.simulate_moments(params, base_params)
        
        if weights is None:
            weights = {k: 1.0 for k in self.empirical_data.keys()}
        
        # Calculate weighted squared distance
        distance = 0
        for moment_name in self.empirical_data.keys():
            if moment_name in simulated:
                emp_val = self.empirical_data[moment_name]
                sim_val = simulated[moment_name]
                
                if not np.isnan(sim_val) and emp_val != 0:
                    error = (sim_val - emp_val) / abs(emp_val)
                    distance += weights.get(moment_name, 1.0) * error**2
                else:
                    distance += 1e6  # Large penalty for failed simulation
        
        return distance
    
    def estimate_msm(self, base_params, weights=None, method='differential_evolution'):
        """
        Estimate parameters using Method of Simulated Moments
        
        Parameters:
        -----------
        base_params : dict
            Fixed parameter values
        weights : dict
            Moment weights
        method : str
            Optimization method ('differential_evolution' or 'minimize')
            
        Returns:
        --------
        dict : Estimation results
        """
        print("Starting MSM estimation...")
        
        # Get bounds as list
        bounds = [self.param_bounds[name] for name in self.param_bounds.keys()]
        
        # Define objective
        def objective(params_array):
            return self.msm_objective(params_array, base_params, weights)
        
        # Optimize
        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42,
                disp=True
            )
        else:
            # Initial guess (midpoint of bounds)
            x0 = np.array([(b[0] + b[1])/2 for b in bounds])
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        # Extract results
        estimated_params = {}
        for i, name in enumerate(self.param_bounds.keys()):
            estimated_params[name] = result.x[i]
        
        self.best_params = estimated_params
        
        # Compute fit
        simulated_moments = self.simulate_moments(result.x, base_params)
        
        results = {
            'estimated_params': estimated_params,
            'objective_value': result.fun,
            'simulated_moments': simulated_moments,
            'empirical_moments': self.empirical_data,
            'success': result.success if hasattr(result, 'success') else True,
            'method': method
        }
        
        self.estimation_results['msm'] = results
        
        print("\nMSM Estimation Complete!")
        print(f"Objective value: {result.fun:.6f}")
        print("\nEstimated parameters:")
        for name, val in estimated_params.items():
            print(f"  {name}: {val:.6f}")
        
        return results
    
    def grid_search(self, base_params, n_points=5):
        """
        Grid search over parameter space
        
        Parameters:
        -----------
        base_params : dict
            Fixed parameters
        n_points : int
            Number of points per parameter
            
        Returns:
        --------
        pd.DataFrame : Grid search results
        """
        print(f"Starting grid search with {n_points} points per parameter...")
        
        # Create grid
        param_names = list(self.param_bounds.keys())
        grid_points = []
        
        for name in param_names:
            pmin, pmax = self.param_bounds[name]
            grid_points.append(np.linspace(pmin, pmax, n_points))
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*grid_points))
        
        print(f"Total combinations: {len(combinations)}")
        
        results = []
        
        for i, combo in enumerate(combinations):
            if (i+1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(combinations)} combinations")
            
            param_dict = dict(zip(param_names, combo))
            
            # Calculate objective
            obj_val = self.msm_objective(np.array(combo), base_params)
            
            result_row = {**param_dict, 'objective': obj_val}
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        
        # Find best
        best_idx = results_df['objective'].idxmin()
        best_row = results_df.loc[best_idx]
        
        print("\nGrid Search Complete!")
        print(f"Best objective: {best_row['objective']:.6f}")
        print("\nBest parameters:")
        for name in param_names:
            print(f"  {name}: {best_row[name]:.6f}")
        
        self.estimation_results['grid_search'] = results_df
        self.best_params = {name: best_row[name] for name in param_names}
        
        return results_df
    
    def abc_estimation(self, base_params, n_samples=1000, tolerance=0.1, n_keep=100):
        """
        Approximate Bayesian Computation
        
        Parameters:
        -----------
        base_params : dict
            Fixed parameters
        n_samples : int
            Number of samples to draw
        tolerance : float
            Acceptance threshold
        n_keep : int
            Number of samples to keep
            
        Returns:
        --------
        dict : ABC results
        """
        print(f"Starting ABC with {n_samples} samples...")
        
        accepted_params = []
        accepted_distances = []
        
        for i in range(n_samples):
            if (i+1) % 100 == 0:
                print(f"  Sample {i+1}/{n_samples}, accepted: {len(accepted_params)}")
            
            # Sample parameters from priors (uniform)
            sample = []
            for name, (pmin, pmax) in self.param_bounds.items():
                sample.append(rd.uniform(pmin, pmax))
            
            # Compute distance
            distance = self.msm_objective(np.array(sample), base_params)
            
            # Accept if below tolerance
            if distance < tolerance:
                param_dict = {}
                for j, name in enumerate(self.param_bounds.keys()):
                    param_dict[name] = sample[j]
                accepted_params.append(param_dict)
                accepted_distances.append(distance)
        
        # Keep best n_keep
        if len(accepted_params) > n_keep:
            sorted_idx = np.argsort(accepted_distances)[:n_keep]
            accepted_params = [accepted_params[i] for i in sorted_idx]
            accepted_distances = [accepted_distances[i] for i in sorted_idx]
        
        print(f"\nABC Complete! Accepted {len(accepted_params)} samples")
        
        # Compute posterior means
        if len(accepted_params) > 0:
            posterior_means = {}
            for name in self.param_bounds.keys():
                vals = [p[name] for p in accepted_params]
                posterior_means[name] = np.mean(vals)
            
            self.best_params = posterior_means
            
            print("\nPosterior means:")
            for name, val in posterior_means.items():
                print(f"  {name}: {val:.6f}")
        
        results = {
            'accepted_params': accepted_params,
            'accepted_distances': accepted_distances,
            'posterior_means': posterior_means if len(accepted_params) > 0 else None,
            'acceptance_rate': len(accepted_params) / n_samples
        }
        
        self.estimation_results['abc'] = results
        
        return results
    
    def plot_estimation_results(self, method='msm', save_path='estimation_results.png'):
        """
        Plot estimation results
        """
        if method not in self.estimation_results:
            print(f"No results for method {method}")
            return
        
        results = self.estimation_results[method]
        
        if method == 'msm':
            # Plot moment comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            moments = list(self.empirical_data.keys())
            emp_vals = [self.empirical_data[m] for m in moments]
            sim_vals = [results['simulated_moments'][m] for m in moments]
            
            x = np.arange(len(moments))
            width = 0.35
            
            ax.bar(x - width/2, emp_vals, width, label='Empirical', alpha=0.7)
            ax.bar(x + width/2, sim_vals, width, label='Simulated', alpha=0.7)
            
            ax.set_xlabel('Moment')
            ax.set_ylabel('Value')
            ax.set_title('MSM: Empirical vs Simulated Moments')
            ax.set_xticks(x)
            ax.set_xticklabels(moments, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Estimation plot saved to {save_path}")
        
        elif method == 'abc' and 'accepted_params' in results:
            # Plot posterior distributions
            accepted = results['accepted_params']
            if len(accepted) == 0:
                print("No accepted samples to plot")
                return
            
            param_names = list(self.param_bounds.keys())
            n_params = len(param_names)
            
            fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
            if n_params == 1:
                axes = [axes]
            
            for i, name in enumerate(param_names):
                vals = [p[name] for p in accepted]
                axes[i].hist(vals, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(name)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Posterior: {name}')
                axes[i].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ABC posterior plot saved to {save_path}")
        
        return fig


def example_parameter_estimation():
    """
    Example parameter estimation
    """
    # Define empirical data to match
    empirical_data = {
        'mean_unemployment': 0.06,
        'std_unemployment': 0.015,
        'mean_real_wage': 0.95,
        'wage_inequality': 3.5,
        'skill_mismatch': 0.15,
    }
    
    # Parameters to estimate
    param_bounds = {
        'lambda_LM': (1, 10),
        'sigma_w': (0.001, 0.01),
        'mu_r': (0.2, 0.5),
    }
    
    # Fixed parameters
    base_params = {
        'H': 200,
        'F': 20,
        'alpha_2': 0.25,
        'chi_C': 0.2,
        'sigma_m': 0.005,
        'nu': 0.1,
        'u_r': 0.08,
        'beta': 1,
        'lambda_exp': 0.5,
        'N_app': 4,
        'sigma': 1.5,
        'nr_to_r': True,
        'gamma_nr': 0.33,
        'min_w_par': 0.3,
    }
    
    # Create estimator
    estimator = ParameterEstimator(empirical_data, param_bounds, T=300, burnin=100)
    
    # Method 1: Grid search (fast, coarse)
    print("="*70)
    print("METHOD 1: GRID SEARCH")
    print("="*70)
    grid_results = estimator.grid_search(base_params, n_points=3)
    
    # Method 2: MSM with optimization (more precise)
    print("\n" + "="*70)
    print("METHOD 2: MSM OPTIMIZATION")
    print("="*70)
    msm_results = estimator.estimate_msm(base_params, method='differential_evolution')
    estimator.plot_estimation_results('msm', 'msm_results.png')
    
    print("\nParameter estimation complete!")
    
    return estimator


if __name__ == '__main__':
    estimator = example_parameter_estimation()
