"""
Digitalization and Labor Market Regulation Module

This module extends the labor market ABM to study:
1. Automation and digitalization effects on routine jobs
2. Labor market regulations (minimum wage, hiring/firing costs, etc.)
3. Skill upgrading and retraining policies
4. Platform economy and gig work effects
5. Policy interventions and their impacts on skill mismatch
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from model_class import Model
import pandas as pd
from copy import deepcopy


class DigitalizationModel(Model):
    """
    Extended model with digitalization and regulation features
    """
    
    def __init__(self, 
                 # Digitalization parameters
                 automation_rate=0.0,
                 automation_start=0,
                 automation_routine_impact=0.8,
                 automation_cost_reduction=0.2,
                 # Regulation parameters
                 firing_cost=0.0,
                 hiring_subsidy=0.0,
                 training_effectiveness=0.0,
                 training_cost=0.0,
                 # Policy parameters
                 min_wage_policy=None,
                 retraining_policy=False,
                 **kwargs):
        """
        Initialize extended model
        
        Parameters:
        -----------
        automation_rate : float
            Rate of automation per period (0 to 1)
        automation_start : int
            Period when automation begins
        automation_routine_impact : float
            Share of routine jobs affected by automation
        automation_cost_reduction : float
            Cost reduction from automation
        firing_cost : float
            Cost to fire a worker (as fraction of wage)
        hiring_subsidy : float
            Subsidy for hiring workers
        training_effectiveness : float
            Probability that training converts R worker to NR
        training_cost : float
            Cost of training program
        min_wage_policy : dict or None
            Minimum wage policy {'start': t, 'level': w}
        retraining_policy : bool
            Whether retraining programs are active
        """
        super().__init__(**kwargs)
        
        # Digitalization parameters
        self.automation_rate = automation_rate
        self.automation_start = automation_start
        self.automation_routine_impact = automation_routine_impact
        self.automation_cost_reduction = automation_cost_reduction
        
        # Track automation level over time
        self.automation_level = np.zeros(self.T)
        self.automated_jobs = np.zeros(self.T)
        
        # Regulation parameters
        self.firing_cost = firing_cost
        self.hiring_subsidy = hiring_subsidy
        self.training_effectiveness = training_effectiveness
        self.training_cost = training_cost
        
        # Policy parameters
        self.min_wage_policy = min_wage_policy
        self.retraining_policy = retraining_policy
        
        # Track worker transitions
        self.workers_retrained = np.zeros(self.T)
        self.workers_displaced = np.zeros(self.T)
        
        # Track policy costs
        self.total_firing_costs = np.zeros(self.T)
        self.total_hiring_subsidies = np.zeros(self.T)
        self.total_training_costs = np.zeros(self.T)
    
    def apply_automation(self):
        """
        Apply automation effects to routine jobs
        """
        if self.t < self.automation_start:
            return
        
        # Update automation level
        progress = min(1.0, (self.t - self.automation_start) * self.automation_rate)
        self.automation_level[self.t] = progress
        
        # Firms adjust production function with automation
        for f in self.f_arr[self.active_fs]:
            # Automation reduces need for routine workers
            automation_factor = 1 - progress * self.automation_routine_impact
            f.automated_efficiency = automation_factor
            
            # Automation reduces costs
            cost_factor = 1 - progress * self.automation_cost_reduction
            f.automation_cost_factor = cost_factor
        
        # Count jobs automated
        routine_demand = sum([f.d_Nr for f in self.f_arr[self.active_fs]])
        self.automated_jobs[self.t] = routine_demand * progress * self.automation_routine_impact
    
    def apply_firing_costs(self, n_fired, avg_wage):
        """
        Apply firing costs to firm
        
        Returns:
        --------
        float : Total firing cost
        """
        return n_fired * avg_wage * self.firing_cost
    
    def apply_hiring_subsidy(self, n_hired):
        """
        Apply hiring subsidy
        
        Returns:
        --------
        float : Total subsidy
        """
        return n_hired * self.hiring_subsidy
    
    def offer_retraining(self):
        """
        Offer retraining to unemployed routine workers
        """
        if not self.retraining_policy:
            return
        
        n_retrained = 0
        
        # Find unemployed routine workers
        unemployed_routine = []
        for h in self.h_arr[self.routine_arr]:
            if h.u[self.t] == 1:
                unemployed_routine.append(h)
        
        # Offer retraining
        for h in unemployed_routine:
            if rd.random() < self.training_effectiveness:
                # Worker successfully retrains
                h.routine = False
                h.nr_job = False  # Will look for NR job
                n_retrained += 1
        
        self.workers_retrained[self.t] = n_retrained
        self.total_training_costs[self.t] = n_retrained * self.training_cost
    
    def apply_min_wage_policy(self):
        """
        Apply minimum wage policy
        """
        if self.min_wage_policy is None:
            return
        
        if self.t >= self.min_wage_policy.get('start', 0):
            policy_min_wage = self.min_wage_policy.get('level', self.min_w)
            self.min_w = max(self.min_w, policy_min_wage)
    
    def step_function(self):
        """
        Extended step function with digitalization and regulation
        """
        # Apply automation
        self.apply_automation()
        
        # Apply minimum wage policy
        self.apply_min_wage_policy()
        
        # Offer retraining
        self.offer_retraining()
        
        # Call parent step function
        super().step_function()


class DigitalizationAnalyzer:
    """
    Analyzer for digitalization and regulation effects
    """
    
    def __init__(self):
        """
        Initialize analyzer
        """
        self.results = {}
    
    def compare_scenarios(self, scenarios, T=500, n_runs=5):
        """
        Compare different policy scenarios
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary of scenario names and parameters
        T : int
            Simulation periods
        n_runs : int
            Number of replications per scenario
            
        Returns:
        --------
        dict : Results for each scenario
        """
        results = {}
        
        for scenario_name, params in scenarios.items():
            print(f"\nRunning scenario: {scenario_name}")
            
            scenario_results = {
                'unemployment': [],
                'mismatch': [],
                'mean_wage': [],
                'inequality': [],
                'gdp': []
            }
            
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                
                # Set random seed for reproducibility
                rd.seed(run)
                
                # Create and run model
                m = DigitalizationModel(T=T, **params)
                m.run()
                
                # Collect metrics (excluding burn-in)
                burnin = 100
                scenario_results['unemployment'].append(
                    np.mean(m.u_r_arr[burnin:]))
                scenario_results['mismatch'].append(
                    np.mean(m.share_nr_in_r[burnin:]) / m.H_nr)
                scenario_results['mean_wage'].append(
                    np.mean(m.mean_w_arr[burnin:]))
                scenario_results['inequality'].append(
                    np.mean(m.nine_to_one[burnin:]))
                scenario_results['gdp'].append(
                    np.mean(m.GDP[burnin:]))
            
            # Calculate means and stds
            results[scenario_name] = {
                'unemployment_mean': np.mean(scenario_results['unemployment']),
                'unemployment_std': np.std(scenario_results['unemployment']),
                'mismatch_mean': np.mean(scenario_results['mismatch']),
                'mismatch_std': np.std(scenario_results['mismatch']),
                'wage_mean': np.mean(scenario_results['mean_wage']),
                'wage_std': np.std(scenario_results['mean_wage']),
                'inequality_mean': np.mean(scenario_results['inequality']),
                'inequality_std': np.std(scenario_results['inequality']),
                'gdp_mean': np.mean(scenario_results['gdp']),
                'gdp_std': np.std(scenario_results['gdp'])
            }
        
        self.results = results
        return results
    
    def plot_scenario_comparison(self, save_path='scenario_comparison.png'):
        """
        Plot comparison of scenarios
        """
        if not self.results:
            print("No results to plot. Run compare_scenarios first.")
            return
        
        scenarios = list(self.results.keys())
        metrics = ['unemployment_mean', 'mismatch_mean', 
                   'wage_mean', 'inequality_mean', 'gdp_mean']
        metric_labels = ['Unemployment', 'Skill Mismatch', 
                        'Real Wage', 'Inequality (90/10)', 'GDP']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            values = [self.results[s][metric] for s in scenarios]
            errors = [self.results[s][metric.replace('_mean', '_std')] 
                     for s in scenarios]
            
            x_pos = np.arange(len(scenarios))
            ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.set_ylabel(label)
            ax.set_title(f'{label} by Scenario')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scenario comparison saved to {save_path}")
        
        return fig
    
    def generate_policy_report(self, save_path='policy_report.txt'):
        """
        Generate policy analysis report
        """
        if not self.results:
            print("No results to report. Run compare_scenarios first.")
            return
        
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DIGITALIZATION AND POLICY ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for scenario_name, results in self.results.items():
                f.write(f"\nSCENARIO: {scenario_name}\n")
                f.write("-" * 70 + "\n")
                
                f.write(f"Unemployment Rate: {results['unemployment_mean']:.4f} ")
                f.write(f"(±{results['unemployment_std']:.4f})\n")
                
                f.write(f"Skill Mismatch Rate: {results['mismatch_mean']:.4f} ")
                f.write(f"(±{results['mismatch_std']:.4f})\n")
                
                f.write(f"Mean Real Wage: {results['wage_mean']:.4f} ")
                f.write(f"(±{results['wage_std']:.4f})\n")
                
                f.write(f"Inequality (90/10): {results['inequality_mean']:.4f} ")
                f.write(f"(±{results['inequality_std']:.4f})\n")
                
                f.write(f"Mean GDP: {results['gdp_mean']:.2f} ")
                f.write(f"(±{results['gdp_std']:.2f})\n")
            
            # Compare scenarios
            f.write("\n" + "=" * 70 + "\n")
            f.write("SCENARIO COMPARISONS\n")
            f.write("=" * 70 + "\n\n")
            
            baseline = list(self.results.keys())[0]
            f.write(f"Baseline scenario: {baseline}\n\n")
            
            for scenario_name in list(self.results.keys())[1:]:
                f.write(f"\n{scenario_name} vs {baseline}:\n")
                f.write("-" * 70 + "\n")
                
                baseline_unemp = self.results[baseline]['unemployment_mean']
                scenario_unemp = self.results[scenario_name]['unemployment_mean']
                unemp_change = (scenario_unemp - baseline_unemp) / baseline_unemp * 100
                f.write(f"Unemployment change: {unemp_change:+.2f}%\n")
                
                baseline_mismatch = self.results[baseline]['mismatch_mean']
                scenario_mismatch = self.results[scenario_name]['mismatch_mean']
                mismatch_change = (scenario_mismatch - baseline_mismatch) / baseline_mismatch * 100
                f.write(f"Mismatch change: {mismatch_change:+.2f}%\n")
                
                baseline_wage = self.results[baseline]['wage_mean']
                scenario_wage = self.results[scenario_name]['wage_mean']
                wage_change = (scenario_wage - baseline_wage) / baseline_wage * 100
                f.write(f"Real wage change: {wage_change:+.2f}%\n")
                
                baseline_gdp = self.results[baseline]['gdp_mean']
                scenario_gdp = self.results[scenario_name]['gdp_mean']
                gdp_change = (scenario_gdp - baseline_gdp) / baseline_gdp * 100
                f.write(f"GDP change: {gdp_change:+.2f}%\n")
        
        print(f"Policy report saved to {save_path}")


def example_digitalization_analysis():
    """
    Example analysis of digitalization and policy effects
    """
    # Define base parameters
    base_params = {
        'H': 200,
        'F': 20,
        'alpha_2': 0.25,
        'chi_C': 0.2,
        'lambda_LM': 5,
        'sigma_m': 0.005,
        'sigma_w': 0.005,
        'nu': 0.1,
        'u_r': 0.08,
        'beta': 1,
        'lambda_exp': 0.5,
        'N_app': 4,
        'sigma': 1.5,
        'mu_r': 0.3,
        'nr_to_r': True,
        'gamma_nr': 0.33,
        'min_w_par': 0.3
    }
    
    # Define scenarios
    scenarios = {
        'Baseline': {
            **base_params,
            'automation_rate': 0.0,
        },
        
        'Moderate Automation': {
            **base_params,
            'automation_rate': 0.01,
            'automation_start': 100,
            'automation_routine_impact': 0.5,
            'automation_cost_reduction': 0.15,
        },
        
        'High Automation': {
            **base_params,
            'automation_rate': 0.02,
            'automation_start': 100,
            'automation_routine_impact': 0.8,
            'automation_cost_reduction': 0.3,
        },
        
        'Automation + Retraining': {
            **base_params,
            'automation_rate': 0.02,
            'automation_start': 100,
            'automation_routine_impact': 0.8,
            'automation_cost_reduction': 0.3,
            'retraining_policy': True,
            'training_effectiveness': 0.3,
            'training_cost': 0.5,
        },
        
        'Automation + Higher Min Wage': {
            **base_params,
            'automation_rate': 0.02,
            'automation_start': 100,
            'automation_routine_impact': 0.8,
            'automation_cost_reduction': 0.3,
            'min_wage_policy': {'start': 150, 'level': 0.6},
        },
    }
    
    # Run analysis
    analyzer = DigitalizationAnalyzer()
    results = analyzer.compare_scenarios(scenarios, T=500, n_runs=3)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for scenario, metrics in results.items():
        print(f"\n{scenario}:")
        print(f"  Unemployment: {metrics['unemployment_mean']:.4f} (±{metrics['unemployment_std']:.4f})")
        print(f"  Mismatch: {metrics['mismatch_mean']:.4f} (±{metrics['mismatch_std']:.4f})")
        print(f"  Real Wage: {metrics['wage_mean']:.4f} (±{metrics['wage_std']:.4f})")
    
    # Plot results
    analyzer.plot_scenario_comparison('scenario_comparison.png')
    
    # Generate report
    analyzer.generate_policy_report('policy_report.txt')
    
    print("\nDigitalization analysis complete!")
    
    return analyzer


if __name__ == '__main__':
    analyzer = example_digitalization_analysis()
