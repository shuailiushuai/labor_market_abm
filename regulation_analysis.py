"""
Labor Market Regulation and Skill Mismatch Analysis Module

This module extends the base model to analyze:
1. Labor market regulations (minimum wage, firing costs, hiring subsidies)
2. Skill mismatch effects and dynamics
3. Retraining and skill upgrade policies
4. Labor market friction analysis
"""

import numpy as np
import numpy.random as rd
from model_class import Model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


class RegulationAnalyzer:
    """
    Analyzer for labor market regulation and skill mismatch studies
    """
    
    def __init__(self, T=500, burn_in=200):
        """
        Initialize regulation analyzer
        
        Parameters:
        -----------
        T : int
            Number of time periods to simulate
        burn_in : int
            Number of initial periods to exclude from analysis
        """
        self.T = T
        self.burn_in = burn_in
        self.scenarios = {}
        self.results = {}
        
    def run_baseline_scenario(self, base_params, n_replications=5, seed=42):
        """
        Run baseline scenario without regulations
        
        Parameters:
        -----------
        base_params : dict
            Base model parameters
        n_replications : int
            Number of simulation replications
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Baseline results
        """
        print("Running baseline scenario...")
        rd.seed(seed)
        
        results_list = []
        for rep in range(n_replications):
            m = Model(T=self.T, **base_params)
            m.run()
            
            metrics = self._extract_metrics(m)
            metrics['replication'] = rep
            results_list.append(metrics)
        
        self.scenarios['baseline'] = {
            'params': base_params,
            'results': pd.DataFrame(results_list)
        }
        
        print("Baseline scenario completed!")
        return self.scenarios['baseline']
    
    def analyze_minimum_wage_policy(self, base_params, min_wage_levels, 
                                   n_replications=5, seed=42):
        """
        Analyze effects of different minimum wage levels
        
        Parameters:
        -----------
        base_params : dict
            Base model parameters
        min_wage_levels : list
            List of minimum wage parameters to test (min_w_par values)
        n_replications : int
            Number of simulation replications per policy
        seed : int
            Random seed
            
        Returns:
        --------
        pd.DataFrame : Results across different minimum wage policies
        """
        print("\nAnalyzing minimum wage policies...")
        rd.seed(seed)
        
        results_list = []
        for min_w in min_wage_levels:
            print(f"  Testing minimum wage parameter: {min_w:.2f}")
            
            for rep in range(n_replications):
                params = base_params.copy()
                params['min_w_par'] = min_w
                
                m = Model(T=self.T, **params)
                m.run()
                
                metrics = self._extract_metrics(m)
                metrics['min_wage_par'] = min_w
                metrics['replication'] = rep
                results_list.append(metrics)
        
        self.results['minimum_wage'] = pd.DataFrame(results_list)
        print("Minimum wage analysis completed!")
        return self.results['minimum_wage']
    
    def analyze_skill_mismatch(self, base_params, skill_ratios, 
                              n_replications=5, seed=42):
        """
        Analyze effects of different skill compositions (routine vs non-routine)
        
        Parameters:
        -----------
        base_params : dict
            Base model parameters
        skill_ratios : list
            List of gamma_nr values (proportion of non-routine workers)
        n_replications : int
            Number of simulation replications
        seed : int
            Random seed
            
        Returns:
        --------
        pd.DataFrame : Results showing skill mismatch effects
        """
        print("\nAnalyzing skill mismatch scenarios...")
        rd.seed(seed)
        
        results_list = []
        for gamma_nr in skill_ratios:
            print(f"  Testing skill ratio (gamma_nr): {gamma_nr:.2f}")
            
            for rep in range(n_replications):
                params = base_params.copy()
                params['gamma_nr'] = gamma_nr
                
                m = Model(T=self.T, **params)
                m.run()
                
                metrics = self._extract_metrics(m)
                metrics['gamma_nr'] = gamma_nr
                metrics['replication'] = rep
                
                # Additional skill mismatch metrics
                metrics['routine_unemployment'] = np.mean(m.ur_r_arr[self.burn_in:])
                metrics['nonroutine_unemployment'] = np.mean(m.unr_r_arr[self.burn_in:])
                metrics['skill_mismatch_rate'] = np.mean(m.share_nr_in_r[self.burn_in:])
                
                results_list.append(metrics)
        
        self.results['skill_mismatch'] = pd.DataFrame(results_list)
        print("Skill mismatch analysis completed!")
        return self.results['skill_mismatch']
    
    def analyze_firing_costs(self, base_params, firing_cost_levels,
                            n_replications=5, seed=42):
        """
        Analyze effects of firing costs (modeled through wage stickiness)
        
        Parameters:
        -----------
        base_params : dict
            Base model parameters
        firing_cost_levels : list
            List of sigma_w values (wage adjustment costs)
        n_replications : int
            Number of replications
        seed : int
            Random seed
            
        Returns:
        --------
        pd.DataFrame : Results showing firing cost effects
        """
        print("\nAnalyzing firing cost effects...")
        rd.seed(seed)
        
        results_list = []
        for firing_cost in firing_cost_levels:
            print(f"  Testing firing cost (sigma_w): {firing_cost:.4f}")
            
            for rep in range(n_replications):
                params = base_params.copy()
                params['sigma_w'] = firing_cost
                
                m = Model(T=self.T, **params)
                m.run()
                
                metrics = self._extract_metrics(m)
                metrics['firing_cost'] = firing_cost
                metrics['replication'] = rep
                results_list.append(metrics)
        
        self.results['firing_costs'] = pd.DataFrame(results_list)
        print("Firing cost analysis completed!")
        return self.results['firing_costs']
    
    def analyze_labor_market_efficiency(self, base_params, lambda_LM_values,
                                       n_replications=5, seed=42):
        """
        Analyze labor market matching efficiency
        
        Parameters:
        -----------
        base_params : dict
            Base model parameters
        lambda_LM_values : list
            List of lambda_LM values (labor market matching intensity)
        n_replications : int
            Number of replications
        seed : int
            Random seed
            
        Returns:
        --------
        pd.DataFrame : Results showing matching efficiency effects
        """
        print("\nAnalyzing labor market matching efficiency...")
        rd.seed(seed)
        
        results_list = []
        for lambda_LM in lambda_LM_values:
            print(f"  Testing matching efficiency (lambda_LM): {lambda_LM:.1f}")
            
            for rep in range(n_replications):
                params = base_params.copy()
                params['lambda_LM'] = lambda_LM
                
                m = Model(T=self.T, **params)
                m.run()
                
                metrics = self._extract_metrics(m)
                metrics['lambda_LM'] = lambda_LM
                metrics['replication'] = rep
                results_list.append(metrics)
        
        self.results['matching_efficiency'] = pd.DataFrame(results_list)
        print("Matching efficiency analysis completed!")
        return self.results['matching_efficiency']
    
    def analyze_combined_policies(self, base_params, policy_combinations,
                                 n_replications=5, seed=42):
        """
        Analyze combinations of policies
        
        Parameters:
        -----------
        base_params : dict
            Base model parameters
        policy_combinations : list of dict
            List of parameter dictionaries representing policy combinations
        n_replications : int
            Number of replications
        seed : int
            Random seed
            
        Returns:
        --------
        pd.DataFrame : Results for policy combinations
        """
        print("\nAnalyzing combined policy scenarios...")
        rd.seed(seed)
        
        results_list = []
        for i, policy in enumerate(policy_combinations):
            print(f"  Testing policy combination {i+1}/{len(policy_combinations)}")
            
            for rep in range(n_replications):
                params = base_params.copy()
                params.update(policy)
                
                m = Model(T=self.T, **params)
                m.run()
                
                metrics = self._extract_metrics(m)
                metrics['policy_id'] = i
                metrics['replication'] = rep
                
                # Add policy parameters to results
                for key, val in policy.items():
                    metrics[f'policy_{key}'] = val
                
                results_list.append(metrics)
        
        self.results['combined_policies'] = pd.DataFrame(results_list)
        print("Combined policy analysis completed!")
        return self.results['combined_policies']
    
    def _extract_metrics(self, model):
        """
        Extract key metrics from a model run
        
        Parameters:
        -----------
        model : Model
            Completed model instance
            
        Returns:
        --------
        dict : Dictionary of metrics
        """
        t = self.burn_in
        
        metrics = {
            # Labor market outcomes
            'mean_unemployment': np.mean(model.u_r_arr[t:]),
            'std_unemployment': np.std(model.u_r_arr[t:]),
            'mean_vacancies': np.mean(model.open_vs[t:]),
            
            # Wage outcomes
            'mean_real_wage': np.mean(model.mean_w_arr[t:]),
            'std_real_wage': np.std(model.mean_w_arr[t:]),
            'mean_routine_wage': np.mean(model.mean_r_w_arr[t:]),
            'mean_nonroutine_wage': np.mean(model.mean_nr_w_arr[t:]),
            'wage_gap': np.mean(model.mean_nr_w_arr[t:]) - np.mean(model.mean_r_w_arr[t:]),
            
            # Inequality
            'wage_inequality_9_1': np.mean(model.nine_to_one[t:]),
            'wage_variance': np.mean(model.wage_variance[t:]),
            
            # Production and prices
            'mean_GDP': np.mean(model.GDP[t:]),
            'std_GDP': np.std(model.GDP[t:]),
            'mean_price': np.mean(model.mean_p_arr[t:]),
            
            # Firm dynamics
            'firm_default_rate': np.mean(model.share_inactive[t:]),
            
            # Skill mismatch
            'skill_mismatch': np.mean(model.share_nr_in_r[t:]),
        }
        
        return metrics
    
    def plot_policy_comparison(self, analysis_type, param_name, metric='mean_unemployment',
                              save_path=None):
        """
        Plot comparison of policy effects
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis (e.g., 'minimum_wage', 'skill_mismatch')
        param_name : str
            Name of the parameter being varied
        metric : str
            Metric to plot
        save_path : str, optional
            Path to save the figure
        """
        if analysis_type not in self.results:
            raise ValueError(f"No results for {analysis_type}")
        
        df = self.results[analysis_type]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by parameter and calculate mean and std
        grouped = df.groupby(param_name)[metric].agg(['mean', 'std'])
        
        ax.plot(grouped.index, grouped['mean'], 'o-', linewidth=2, markersize=8)
        ax.fill_between(grouped.index,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.3)
        
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f'{metric} vs {param_name}', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multi_metric_comparison(self, analysis_type, param_name,
                                    metrics=None, save_path=None):
        """
        Plot multiple metrics for policy comparison
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis
        param_name : str
            Name of the parameter being varied
        metrics : list, optional
            List of metrics to plot
        save_path : str, optional
            Path to save the figure
        """
        if metrics is None:
            metrics = ['mean_unemployment', 'mean_GDP', 'mean_real_wage', 'wage_inequality_9_1']
        
        if analysis_type not in self.results:
            raise ValueError(f"No results for {analysis_type}")
        
        df = self.results[analysis_type]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            grouped = df.groupby(param_name)[metric].agg(['mean', 'std'])
            
            axes[i].plot(grouped.index, grouped['mean'], 'o-', linewidth=2, markersize=6)
            axes[i].fill_between(grouped.index,
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                alpha=0.3)
            
            axes[i].set_xlabel(param_name, fontsize=12)
            axes[i].set_ylabel(metric, fontsize=12)
            axes[i].set_title(metric.replace('_', ' ').title(), fontsize=14)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_policy_report(self, output_file='regulation_analysis_report.txt'):
        """
        Generate comprehensive policy analysis report
        
        Parameters:
        -----------
        output_file : str
            Path to save the report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LABOR MARKET REGULATION AND SKILL MISMATCH ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Baseline scenario
            if 'baseline' in self.scenarios:
                f.write("BASELINE SCENARIO\n")
                f.write("-" * 80 + "\n")
                baseline = self.scenarios['baseline']['results']
                for col in baseline.columns:
                    if col != 'replication':
                        f.write(f"{col}: {baseline[col].mean():.4f} ± {baseline[col].std():.4f}\n")
                f.write("\n")
            
            # Policy analysis results
            for analysis_type, results_df in self.results.items():
                f.write(f"\n{analysis_type.upper().replace('_', ' ')}\n")
                f.write("-" * 80 + "\n\n")
                
                # Identify the policy parameter
                param_cols = [col for col in results_df.columns 
                            if col not in ['replication', 'mean_unemployment', 'std_unemployment',
                                          'mean_vacancies', 'mean_real_wage', 'std_real_wage',
                                          'mean_routine_wage', 'mean_nonroutine_wage', 'wage_gap',
                                          'wage_inequality_9_1', 'wage_variance', 'mean_GDP', 
                                          'std_GDP', 'mean_price', 'firm_default_rate', 
                                          'skill_mismatch', 'routine_unemployment', 
                                          'nonroutine_unemployment', 'skill_mismatch_rate']]
                
                if param_cols:
                    for param in param_cols:
                        f.write(f"Results by {param}:\n")
                        grouped = results_df.groupby(param).agg(['mean', 'std'])
                        f.write(grouped.to_string())
                        f.write("\n\n")
                
                # Key findings
                f.write("Key Findings:\n")
                
                # Unemployment effects
                if 'mean_unemployment' in results_df.columns:
                    best_unemployment = results_df.groupby(param_cols[0] if param_cols else 'replication')['mean_unemployment'].mean()
                    f.write(f"  - Lowest unemployment: {best_unemployment.min():.4f} ")
                    f.write(f"at {param_cols[0] if param_cols else 'baseline'} = {best_unemployment.idxmin()}\n")
                
                # GDP effects
                if 'mean_GDP' in results_df.columns:
                    best_GDP = results_df.groupby(param_cols[0] if param_cols else 'replication')['mean_GDP'].mean()
                    f.write(f"  - Highest GDP: {best_GDP.max():.4f} ")
                    f.write(f"at {param_cols[0] if param_cols else 'baseline'} = {best_GDP.idxmax()}\n")
                
                f.write("\n")
        
        print(f"Policy analysis report saved to {output_file}")


# Example usage
if __name__ == "__main__":
    print("Labor Market Regulation and Skill Mismatch Analysis Module")
    print("=" * 60)
    
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
        'gamma_nr': 0.33,
        'min_w_par': 0.3,
    }
    
    # Initialize analyzer
    analyzer = RegulationAnalyzer(T=300, burn_in=100)
    
    # Run baseline
    print("\n" + "=" * 60)
    baseline = analyzer.run_baseline_scenario(base_params, n_replications=3)
    
    # Analyze minimum wage policies
    print("\n" + "=" * 60)
    min_wage_results = analyzer.analyze_minimum_wage_policy(
        base_params,
        min_wage_levels=[0.2, 0.3, 0.4, 0.5, 0.6],
        n_replications=3
    )
    
    # Analyze skill mismatch
    print("\n" + "=" * 60)
    skill_results = analyzer.analyze_skill_mismatch(
        base_params,
        skill_ratios=[0.2, 0.33, 0.4, 0.5, 0.6],
        n_replications=3
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_multi_metric_comparison(
        'minimum_wage',
        'min_wage_par',
        save_path='results/minimum_wage_effects.png'
    )
    
    analyzer.plot_multi_metric_comparison(
        'skill_mismatch',
        'gamma_nr',
        save_path='results/skill_mismatch_effects.png'
    )
    
    # Generate report
    analyzer.generate_policy_report('results/regulation_analysis_report.txt')
    
    print("\nRegulation and skill mismatch analysis completed!")
