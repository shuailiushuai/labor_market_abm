"""
Skill Mismatch Analysis Module for Labor Market ABM

This module provides comprehensive analysis of skill mismatch in the labor market:
1. Over-qualification (NR workers in R jobs)
2. Under-qualification (R workers attempting NR jobs)
3. Skill mismatch indices
4. Wage penalties and premiums
5. Job quality measures
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from model_class import Model
import pandas as pd


class SkillMismatchAnalyzer:
    """
    Analyzer for skill mismatch in the labor market model
    """
    
    def __init__(self, model):
        """
        Initialize analyzer with a model instance
        
        Parameters:
        -----------
        model : Model
            An instance of the labor market model
        """
        self.model = model
        
    def get_mismatch_metrics(self, t=None):
        """
        Calculate skill mismatch metrics at time t
        
        Parameters:
        -----------
        t : int
            Time period (if None, uses current time)
            
        Returns:
        --------
        dict : Dictionary of mismatch metrics
        """
        if t is None:
            t = self.model.t - 1
        
        m = self.model
        metrics = {}
        
        # Over-qualification: NR workers in R jobs
        nr_in_r_jobs = np.sum(m.emp_matrix[:, np.logical_and(
            np.invert(m.nr_job_arr), m.non_routine_arr)])
        nr_employed = np.sum(m.emp_matrix[:, m.non_routine_arr])
        
        metrics['overqualified_count'] = nr_in_r_jobs
        metrics['overqualified_rate'] = nr_in_r_jobs / m.H_nr if m.H_nr > 0 else 0
        metrics['overqualified_share_employed'] = (nr_in_r_jobs / nr_employed 
                                                    if nr_employed > 0 else 0)
        
        # Under-qualification: R workers in NR jobs  
        r_in_nr_jobs = np.sum(m.emp_matrix[:, np.logical_and(
            m.nr_job_arr, m.routine_arr)])
        r_employed = np.sum(m.emp_matrix[:, m.routine_arr])
        
        metrics['underqualified_count'] = r_in_nr_jobs
        metrics['underqualified_rate'] = r_in_nr_jobs / m.H_r if m.H_r > 0 else 0
        metrics['underqualified_share_employed'] = (r_in_nr_jobs / r_employed 
                                                     if r_employed > 0 else 0)
        
        # Total mismatch
        total_employed = np.sum(m.emp_matrix)
        total_mismatch = nr_in_r_jobs + r_in_nr_jobs
        
        metrics['total_mismatch_count'] = total_mismatch
        metrics['total_mismatch_rate'] = (total_mismatch / total_employed 
                                          if total_employed > 0 else 0)
        
        # Unemployment by skill type
        metrics['routine_unemployment'] = m.ur_r_arr[t]
        metrics['non_routine_unemployment'] = m.unr_r_arr[t]
        metrics['unemployment_gap'] = m.unr_r_arr[t] - m.ur_r_arr[t]
        
        return metrics
    
    def get_wage_penalties(self, t=None):
        """
        Calculate wage penalties/premiums due to skill mismatch
        
        Returns:
        --------
        dict : Dictionary of wage metrics
        """
        if t is None:
            t = self.model.t - 1
            
        m = self.model
        metrics = {}
        
        # Get wages by worker type and job type
        # NR workers in NR jobs (well-matched)
        nr_in_nr = np.logical_and(m.nr_job_arr, m.non_routine_arr)
        employed_nr_in_nr = np.logical_and(nr_in_nr, 
                                           np.sum(m.emp_matrix, axis=0) > 0)
        wages_nr_in_nr = np.array([h.w for h in m.h_arr[employed_nr_in_nr]])
        wages_nr_in_nr = wages_nr_in_nr[wages_nr_in_nr > 0]
        
        # NR workers in R jobs (over-qualified)
        nr_in_r = np.logical_and(np.invert(m.nr_job_arr), m.non_routine_arr)
        employed_nr_in_r = np.logical_and(nr_in_r,
                                          np.sum(m.emp_matrix, axis=0) > 0)
        wages_nr_in_r = np.array([h.w for h in m.h_arr[employed_nr_in_r]])
        wages_nr_in_r = wages_nr_in_r[wages_nr_in_r > 0]
        
        # R workers in R jobs (well-matched)
        r_in_r = np.logical_and(np.invert(m.nr_job_arr), m.routine_arr)
        employed_r_in_r = np.logical_and(r_in_r,
                                         np.sum(m.emp_matrix, axis=0) > 0)
        wages_r_in_r = np.array([h.w for h in m.h_arr[employed_r_in_r]])
        wages_r_in_r = wages_r_in_r[wages_r_in_r > 0]
        
        # R workers in NR jobs (under-qualified)
        r_in_nr = np.logical_and(m.nr_job_arr, m.routine_arr)
        employed_r_in_nr = np.logical_and(r_in_nr,
                                          np.sum(m.emp_matrix, axis=0) > 0)
        wages_r_in_nr = np.array([h.w for h in m.h_arr[employed_r_in_nr]])
        wages_r_in_nr = wages_r_in_nr[wages_r_in_nr > 0]
        
        # Calculate means and penalties
        metrics['wage_nr_in_nr_mean'] = np.mean(wages_nr_in_nr) if len(wages_nr_in_nr) > 0 else 0
        metrics['wage_nr_in_r_mean'] = np.mean(wages_nr_in_r) if len(wages_nr_in_r) > 0 else 0
        metrics['wage_r_in_r_mean'] = np.mean(wages_r_in_r) if len(wages_r_in_r) > 0 else 0
        metrics['wage_r_in_nr_mean'] = np.mean(wages_r_in_nr) if len(wages_r_in_nr) > 0 else 0
        
        # Over-qualification penalty (NR worker wage loss from being in R job)
        if len(wages_nr_in_nr) > 0 and len(wages_nr_in_r) > 0:
            metrics['overqualification_penalty'] = (
                (metrics['wage_nr_in_nr_mean'] - metrics['wage_nr_in_r_mean']) / 
                metrics['wage_nr_in_nr_mean']
            )
        else:
            metrics['overqualification_penalty'] = 0
        
        # Under-qualification premium (R worker wage gain from being in NR job)
        if len(wages_r_in_r) > 0 and len(wages_r_in_nr) > 0:
            metrics['underqualification_premium'] = (
                (metrics['wage_r_in_nr_mean'] - metrics['wage_r_in_r_mean']) /
                metrics['wage_r_in_r_mean']
            )
        else:
            metrics['underqualification_premium'] = 0
        
        return metrics
    
    def track_mismatch_over_time(self, start=0, end=None):
        """
        Track skill mismatch metrics over time
        
        Parameters:
        -----------
        start : int
            Starting period
        end : int
            Ending period (if None, uses model.t)
            
        Returns:
        --------
        pd.DataFrame : Time series of mismatch metrics
        """
        if end is None:
            end = self.model.t
        
        results = []
        
        for t in range(start, end):
            # Temporarily set model time
            original_t = self.model.t
            self.model.t = t + 1
            
            metrics = self.get_mismatch_metrics(t)
            metrics['time'] = t
            results.append(metrics)
            
            self.model.t = original_t
        
        return pd.DataFrame(results)
    
    def plot_mismatch_evolution(self, start=0, end=None, save_path='mismatch_evolution.png'):
        """
        Plot evolution of skill mismatch over time
        """
        df = self.track_mismatch_over_time(start, end)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Mismatch rates
        ax = axes[0, 0]
        ax.plot(df['time'], df['overqualified_rate'], label='Over-qualified (NR in R)', linewidth=2)
        ax.plot(df['time'], df['underqualified_rate'], label='Under-qualified (R in NR)', linewidth=2)
        ax.plot(df['time'], df['total_mismatch_rate'], 
                label='Total Mismatch', linewidth=2, linestyle='--', color='red')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Mismatch Rate')
        ax.set_title('Skill Mismatch Rates Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mismatch counts
        ax = axes[0, 1]
        ax.plot(df['time'], df['overqualified_count'], label='Over-qualified', linewidth=2)
        ax.plot(df['time'], df['underqualified_count'], label='Under-qualified', linewidth=2)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Number of Mismatched Workers')
        ax.set_title('Mismatch Counts Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Unemployment by skill type
        ax = axes[1, 0]
        ax.plot(df['time'], df['routine_unemployment'], 
                label='Routine Unemployment', linewidth=2)
        ax.plot(df['time'], df['non_routine_unemployment'], 
                label='Non-routine Unemployment', linewidth=2)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Unemployment Rate')
        ax.set_title('Unemployment by Skill Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Unemployment gap
        ax = axes[1, 1]
        ax.plot(df['time'], df['unemployment_gap'], 
                linewidth=2, color='purple')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Unemployment Gap (NR - R)')
        ax.set_title('Skill-Type Unemployment Gap')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mismatch evolution plot saved to {save_path}")
        
        return fig, df
    
    def analyze_mismatch_determinants(self, df):
        """
        Analyze what determines skill mismatch
        
        Parameters:
        -----------
        df : pd.DataFrame
            Time series data from track_mismatch_over_time
            
        Returns:
        --------
        dict : Correlation analysis results
        """
        results = {}
        
        # Correlation between unemployment gap and mismatch
        results['corr_unemp_gap_mismatch'] = np.corrcoef(
            df['unemployment_gap'], df['total_mismatch_rate'])[0, 1]
        
        # Correlation between NR unemployment and over-qualification
        results['corr_nr_unemp_overqual'] = np.corrcoef(
            df['non_routine_unemployment'], df['overqualified_rate'])[0, 1]
        
        # Correlation between R unemployment and under-qualification  
        results['corr_r_unemp_underqual'] = np.corrcoef(
            df['routine_unemployment'], df['underqualified_rate'])[0, 1]
        
        return results
    
    def generate_mismatch_report(self, start=0, end=None, save_path='mismatch_report.txt'):
        """
        Generate comprehensive skill mismatch report
        """
        df = self.track_mismatch_over_time(start, end)
        
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SKILL MISMATCH ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average over-qualification rate: {df['overqualified_rate'].mean():.4f}\n")
            f.write(f"Average under-qualification rate: {df['underqualified_rate'].mean():.4f}\n")
            f.write(f"Average total mismatch rate: {df['total_mismatch_rate'].mean():.4f}\n")
            f.write(f"\n")
            
            f.write(f"Max over-qualification rate: {df['overqualified_rate'].max():.4f}\n")
            f.write(f"Max under-qualification rate: {df['underqualified_rate'].max():.4f}\n")
            f.write(f"Max total mismatch rate: {df['total_mismatch_rate'].max():.4f}\n")
            f.write(f"\n")
            
            # Volatility
            f.write("VOLATILITY MEASURES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Std over-qualification rate: {df['overqualified_rate'].std():.4f}\n")
            f.write(f"Std under-qualification rate: {df['underqualified_rate'].std():.4f}\n")
            f.write(f"Std total mismatch rate: {df['total_mismatch_rate'].std():.4f}\n")
            f.write(f"\n")
            
            # Unemployment analysis
            f.write("UNEMPLOYMENT BY SKILL TYPE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average routine unemployment: {df['routine_unemployment'].mean():.4f}\n")
            f.write(f"Average non-routine unemployment: {df['non_routine_unemployment'].mean():.4f}\n")
            f.write(f"Average unemployment gap: {df['unemployment_gap'].mean():.4f}\n")
            f.write(f"\n")
            
            # Determinants
            f.write("MISMATCH DETERMINANTS\n")
            f.write("-" * 70 + "\n")
            determinants = self.analyze_mismatch_determinants(df)
            for key, val in determinants.items():
                f.write(f"{key}: {val:.4f}\n")
            f.write(f"\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"Mismatch report saved to {save_path}")
        return df


def example_skill_mismatch_analysis():
    """
    Example usage of skill mismatch analyzer
    """
    print("Running labor market model...")
    
    # Create and run model
    rd.seed(42)
    m = Model(
        T=500, 
        H=200, 
        F=20,
        alpha_2=0.25, 
        chi_C=0.2, 
        lambda_LM=5, 
        sigma_m=0.005,
        sigma_w=0.005,
        nu=0.1,
        u_r=0.08,
        beta=1,
        lambda_exp=0.5,
        N_app=4,
        sigma=1.5,
        mu_r=0.3,
        nr_to_r=True,  # Allow NR workers to take R jobs
        gamma_nr=0.33,
        min_w_par=0.3
    )
    
    m.run()
    
    print("\nAnalyzing skill mismatch...")
    
    # Create analyzer
    analyzer = SkillMismatchAnalyzer(m)
    
    # Get current mismatch metrics
    metrics = analyzer.get_mismatch_metrics()
    print("\nCurrent Mismatch Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # Get wage penalties
    wage_metrics = analyzer.get_wage_penalties()
    print("\nWage Penalties/Premiums:")
    for key, val in wage_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # Plot evolution
    fig, df = analyzer.plot_mismatch_evolution(start=100, save_path='mismatch_evolution.png')
    
    # Generate report
    analyzer.generate_mismatch_report(start=100, save_path='mismatch_report.txt')
    
    print("\nSkill mismatch analysis complete!")
    
    return analyzer, df


if __name__ == '__main__':
    analyzer, df = example_skill_mismatch_analysis()
