"""
Comparative Analysis Module

This module provides tools for comparing multiple scenarios, policies,
and counterfactual analyses for the labor market ABM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ComparativeAnalyzer:
    """
    Class for comparing multiple model scenarios and policy experiments
    """
    
    def __init__(self):
        """Initialize comparative analyzer"""
        self.scenarios = {}
        self.comparisons = {}
        
    def add_scenario(self, name, results_df, description=""):
        """
        Add a scenario for comparison
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        results_df : pd.DataFrame
            Results dataframe from analysis
        description : str, optional
            Description of the scenario
        """
        self.scenarios[name] = {
            'data': results_df,
            'description': description
        }
        print(f"Added scenario: {name}")
    
    def compare_scenarios(self, scenario_names, metrics=None):
        """
        Compare multiple scenarios across specified metrics
        
        Parameters:
        -----------
        scenario_names : list
            List of scenario names to compare
        metrics : list, optional
            List of metrics to compare. If None, uses all common metrics.
            
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        if metrics is None:
            # Get common metrics across all scenarios
            all_cols = set(self.scenarios[scenario_names[0]]['data'].columns)
            for name in scenario_names[1:]:
                all_cols = all_cols.intersection(
                    set(self.scenarios[name]['data'].columns)
                )
            metrics = [col for col in all_cols if col != 'replication']
        
        comparison_data = []
        for name in scenario_names:
            scenario_data = self.scenarios[name]['data']
            
            row = {'scenario': name}
            for metric in metrics:
                if metric in scenario_data.columns:
                    row[f'{metric}_mean'] = scenario_data[metric].mean()
                    row[f'{metric}_std'] = scenario_data[metric].std()
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        self.comparisons['scenarios'] = comparison_df
        
        return comparison_df
    
    def statistical_comparison(self, scenario1, scenario2, metric, test='ttest'):
        """
        Perform statistical test comparing two scenarios
        
        Parameters:
        -----------
        scenario1 : str
            Name of first scenario
        scenario2 : str
            Name of second scenario
        metric : str
            Metric to compare
        test : str
            Statistical test to use ('ttest', 'mannwhitney', 'ks')
            
        Returns:
        --------
        dict : Test results including statistic and p-value
        """
        data1 = self.scenarios[scenario1]['data'][metric]
        data2 = self.scenarios[scenario2]['data'][metric]
        
        if test == 'ttest':
            statistic, pvalue = stats.ttest_ind(data1, data2)
            test_name = "T-test"
        elif test == 'mannwhitney':
            statistic, pvalue = stats.mannwhitneyu(data1, data2)
            test_name = "Mann-Whitney U test"
        elif test == 'ks':
            statistic, pvalue = stats.ks_2samp(data1, data2)
            test_name = "Kolmogorov-Smirnov test"
        else:
            raise ValueError(f"Unknown test: {test}")
        
        results = {
            'test': test_name,
            'metric': metric,
            'scenario1': scenario1,
            'scenario2': scenario2,
            'mean1': data1.mean(),
            'mean2': data2.mean(),
            'diff': data2.mean() - data1.mean(),
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < 0.05,
        }
        
        return results
    
    def rank_scenarios(self, scenarios, metric, ascending=True):
        """
        Rank scenarios by a specific metric
        
        Parameters:
        -----------
        scenarios : list
            List of scenario names
        metric : str
            Metric to rank by
        ascending : bool
            If True, lower values rank higher
            
        Returns:
        --------
        pd.DataFrame : Ranked scenarios
        """
        rankings = []
        for name in scenarios:
            mean_value = self.scenarios[name]['data'][metric].mean()
            std_value = self.scenarios[name]['data'][metric].std()
            rankings.append({
                'scenario': name,
                'mean': mean_value,
                'std': std_value
            })
        
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('mean', ascending=ascending)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        return rankings_df
    
    def plot_scenario_comparison(self, scenario_names, metrics, save_path=None):
        """
        Create comparison plots for multiple scenarios
        
        Parameters:
        -----------
        scenario_names : list
            List of scenarios to compare
        metrics : list
            List of metrics to plot
        save_path : str, optional
            Path to save the figure
        """
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            data_to_plot = []
            labels = []
            for name in scenario_names:
                data_to_plot.append(self.scenarios[name]['data'][metric].values)
                labels.append(name)
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_names)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_radar_comparison(self, scenario_names, metrics, save_path=None):
        """
        Create radar chart comparing scenarios across multiple metrics
        
        Parameters:
        -----------
        scenario_names : list
            List of scenarios to compare
        metrics : list
            List of metrics for comparison
        save_path : str, optional
            Path to save the figure
        """
        # Normalize metrics to 0-1 range for radar chart
        normalized_data = {}
        for name in scenario_names:
            scenario_data = self.scenarios[name]['data']
            normalized_data[name] = []
            
            for metric in metrics:
                values = [self.scenarios[n]['data'][metric].mean() 
                         for n in scenario_names]
                min_val, max_val = min(values), max(values)
                
                if max_val > min_val:
                    norm_val = (scenario_data[metric].mean() - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                
                normalized_data[name].append(norm_val)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_names)))
        
        for i, name in enumerate(scenario_names):
            values = normalized_data[name]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=8)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Scenario Comparison (Normalized)', size=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_comparison_report(self, output_file='comparison_report.txt'):
        """
        Generate comprehensive comparison report
        
        Parameters:
        -----------
        output_file : str
            Path to save the report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPARATIVE ANALYSIS REPORT\n")
            f.write("Labor Market Agent-Based Model\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Number of scenarios: {len(self.scenarios)}\n")
            f.write(f"Scenarios: {', '.join(self.scenarios.keys())}\n\n")
            
            # Scenario descriptions
            f.write("SCENARIO DESCRIPTIONS\n")
            f.write("-" * 80 + "\n")
            for name, data in self.scenarios.items():
                f.write(f"\n{name}:\n")
                f.write(f"  {data['description']}\n")
                f.write(f"  Number of replications: {len(data['data'])}\n")
            
            f.write("\n\n")
            
            # Comparison results
            if 'scenarios' in self.comparisons:
                f.write("SCENARIO COMPARISON\n")
                f.write("-" * 80 + "\n\n")
                f.write(self.comparisons['scenarios'].to_string())
                f.write("\n\n")
            
            # Recommendations
            f.write("=" * 80 + "\n")
            f.write("SUMMARY AND RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")
            
            if 'scenarios' in self.comparisons:
                comp_df = self.comparisons['scenarios']
                
                # Find best scenarios for key metrics
                key_metrics = [col for col in comp_df.columns 
                              if col.endswith('_mean') and 'unemployment' in col]
                if key_metrics:
                    metric = key_metrics[0]
                    best_idx = comp_df[metric].idxmin()
                    f.write(f"Best scenario for low unemployment: ")
                    f.write(f"{comp_df.loc[best_idx, 'scenario']}\n")
                
                key_metrics = [col for col in comp_df.columns 
                              if col.endswith('_mean') and 'GDP' in col]
                if key_metrics:
                    metric = key_metrics[0]
                    best_idx = comp_df[metric].idxmax()
                    f.write(f"Best scenario for high GDP: ")
                    f.write(f"{comp_df.loc[best_idx, 'scenario']}\n")
        
        print(f"Comparison report saved to {output_file}")


# Example usage
if __name__ == "__main__":
    print("Comparative Analysis Module for Labor Market ABM")
    print("=" * 60)
    
    # This would typically use results from regulation_analysis.py
    # or sensitivity_analysis.py
    
    # Create sample data for demonstration
    np.random.seed(42)
    
    scenario1_data = pd.DataFrame({
        'mean_unemployment': np.random.normal(0.08, 0.01, 10),
        'mean_GDP': np.random.normal(1000, 50, 10),
        'mean_real_wage': np.random.normal(1.5, 0.1, 10),
        'replication': range(10)
    })
    
    scenario2_data = pd.DataFrame({
        'mean_unemployment': np.random.normal(0.06, 0.01, 10),
        'mean_GDP': np.random.normal(1050, 50, 10),
        'mean_real_wage': np.random.normal(1.6, 0.1, 10),
        'replication': range(10)
    })
    
    scenario3_data = pd.DataFrame({
        'mean_unemployment': np.random.normal(0.10, 0.015, 10),
        'mean_GDP': np.random.normal(950, 60, 10),
        'mean_real_wage': np.random.normal(1.4, 0.12, 10),
        'replication': range(10)
    })
    
    # Initialize analyzer
    analyzer = ComparativeAnalyzer()
    
    # Add scenarios
    analyzer.add_scenario('Baseline', scenario1_data, 
                         "Standard parameter configuration")
    analyzer.add_scenario('High Minimum Wage', scenario2_data,
                         "Minimum wage = 0.5")
    analyzer.add_scenario('Low Matching Efficiency', scenario3_data,
                         "lambda_LM = 2")
    
    # Compare scenarios
    print("\nComparing scenarios...")
    comparison = analyzer.compare_scenarios(
        ['Baseline', 'High Minimum Wage', 'Low Matching Efficiency']
    )
    print("\nComparison Results:")
    print(comparison)
    
    # Statistical comparison
    print("\nStatistical comparison (Baseline vs High Minimum Wage):")
    stat_result = analyzer.statistical_comparison(
        'Baseline', 'High Minimum Wage', 'mean_unemployment'
    )
    print(f"Mean difference: {stat_result['diff']:.4f}")
    print(f"P-value: {stat_result['pvalue']:.4f}")
    print(f"Significant: {stat_result['significant']}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_scenario_comparison(
        ['Baseline', 'High Minimum Wage', 'Low Matching Efficiency'],
        ['mean_unemployment', 'mean_GDP', 'mean_real_wage'],
        save_path='results/scenario_comparison.png'
    )
    
    analyzer.plot_radar_comparison(
        ['Baseline', 'High Minimum Wage', 'Low Matching Efficiency'],
        ['mean_unemployment', 'mean_GDP', 'mean_real_wage'],
        save_path='results/radar_comparison.png'
    )
    
    # Generate report
    analyzer.generate_comparison_report('results/comparison_report.txt')
    
    print("\nComparative analysis completed!")
