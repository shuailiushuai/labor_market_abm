"""
Main Model Class for Labor Market ABM

This module defines the core Model class that orchestrates the entire
labor market agent-based simulation:

Key Features:
- Initializes agents (firms and households)
- Calibrates parameters to steady-state
- Manages simulation time loop
- Coordinates market processes (labor and goods markets)
- Collects and stores simulation data
- Handles firm defaults and refinancing

The Model class is the main entry point for running simulations and
analyzing labor market dynamics.

Author: Labor Market ABM Project
"""

from agent_class import *
from default_tools import *
from calibration import calibrate_model
from hire_fire_routine import *
from hire_fire_non_routine import *
import numpy as np
from goods_market import gm_matching
from stepfunction_methods import *
from sys import exit


class Model:
    """
    Main labor market agent-based model class.
    
    This class encapsulates the entire simulation, including:
    - Agent initialization (firms and households)
    - Parameter calibration
    - Time loop execution
    - Data collection
    - Market clearing processes
    
    Attributes
    ----------
    H : int
        Number of households
    F : int
        Number of firms
    T : int
        Number of simulation periods
    h_arr : np.ndarray
        Array of household agents
    f_arr : np.ndarray
        Array of firm agents
    emp_matrix : np.ndarray
        Employment matrix (F x H)
    t : int
        Current time period
    """

    def __init__(self,
                 # Exogenously chosen steady state parameters
                 H=200, F=20, Ah=1, u_r=0.08, mu_r=0.3, W_r=1, gamma_nr=0.33,
                 m=0.1, sigma=0.5, delta=1, alpha_2=0.25,
                 # Exogenous model parameters
                 lambda_LM=0.5, lambda_exp=0.25, beta=1, nu=0.1, sigma_m=0.001, sigma_w=0.005,
                 sigma_delta=0.001, chi_C=0.2, T=500,
                 tol=1e-10, N_app=4, nr_to_r=False, a=100, min_w_par=0.3, f_max=3):
        """
        Initialize the labor market model.
        
        Parameters
        ----------
        H : int
            Number of households
        F : int
            Number of firms
        Ah : float
            Initial household wealth
        u_r : float
            Target unemployment rate
        mu_r : float
            Routine labor productivity parameter
        W_r : float
            Routine wage (normalized)
        gamma_nr : float
            Share of non-routine workers
        m : float
            Markup rate
        sigma : float
            Elasticity of substitution
        delta : float
            Dividend payout rate
        alpha_2 : float
            Marginal propensity to consume from wealth
        lambda_LM : float
            Labor market friction parameter
        lambda_exp : float
            Expectation formation speed
        beta : float
            Experience premium parameter
        nu : float
            Inventory share parameter
        sigma_m : float
            Markup volatility
        sigma_w : float
            Wage volatility
        sigma_delta : float
            Dividend rate volatility
        chi_C : float
            Goods market search parameter
        T : int
            Number of simulation periods
        tol : float
            Numerical tolerance
        N_app : int
            Number of applications per household
        nr_to_r : bool
            Whether non-routine workers can take routine jobs
        a : float
            Total factor productivity
        min_w_par : float
            Minimum wage as fraction of median wage
        f_max : int
            Maximum time fired before becoming unemployed
        """


        # exogenous parameters
        self.sigma_m, self.sigma_w, self.chi_C = sigma_m, sigma_w, chi_C
        self.N_app, self.nr_to_r = N_app, nr_to_r
        self.sigma_delta = sigma_delta
        self.lambda_LM = lambda_LM
        self.lambda_exp = lambda_exp
        self.beta = beta
        self.nu = nu

        self.min_w_par = min_w_par
        self.min_w = min_w_par*W_r

        self.T, self.t = T, 0
        self.tol = tol

        # exogenously chosen steady state parameters
        self.H, self.F, self.Ah = H, F, Ah
        self.mu_r, self.u_r, self.W_r, self.gamma_nr  = a*mu_r, u_r, W_r, gamma_nr
        self.m, self.sigma, self.delta, self.alpha_2 = m, sigma, delta, alpha_2

        # steady state calibration
        calibration = calibrate_model(a = a, H=H, F=F, Ah=W_r, u_r=u_r, mu_r=self.mu_r, W_r=W_r, gamma_nr=gamma_nr,
                                      m=m, sigma=sigma, delta=delta, alpha_2=alpha_2)

        # parameters derived from steady state model

        mu_nr, W_nr, Af, uc, p, y_f, pi_f, div_h, div_f, c, alpha_1, Nr, Nnr = calibration

        self.mu_nr, self.W_nr, self.Af, self.uc , self.p, self.y = mu_nr, W_nr, Af, uc, p, y_f
        self.pi_f, self.div_h, self.div_f, self.c = pi_f, div_h, div_f, c
        self.alpha_1, self.Nr, self.Nnr = alpha_1, Nr, Nnr

        # Number of routine resp. non-routine households
        self.H_r = int(np.round(self.H*(1-self.gamma_nr)))
        self.H_nr = int(H - self.H_r)

        routine = True
        non_routine = False

        # create firms
        self.f_arr = np.array([Firm(j, self.Af, T, self.y, self.nu, self.W_r, self.W_nr,
                                    self.delta, self.p, self.m, self.pi_f, self.div_f, self.uc) for j in range(F)])

        # create households
        self.h_arr = np.array([Household(j, self.Ah, T,
                                         routine, self.W_r, self.p, self.c, self.div_h, f_max) for j in range(self.H_r)])

        self.h_arr = np.append(self.h_arr, np.array([Household(j, self.Ah, T,
                                                               non_routine, self.W_nr, self.p, self.c,
                                                               self.div_h, f_max)
                                                     for j in range(self.H_r, self.H_r + self.H_nr)]))

        # select routine resp. non routine workers
        self.routine_arr = np.array([h.routine for h in self.h_arr])
        self.non_routine_arr = np.array([not h.routine for h in self.h_arr])

        # create employment and application matrices
        self.emp_matrix = init_emp_mat(F, H, u_r, self.h_arr)

        employed = np.sum(self.emp_matrix, axis=0) > 0
        self.nr_job_arr = np.logical_and(employed, self.non_routine_arr)
        update_N(self.f_arr, self.emp_matrix, self.nr_job_arr)

        self.app_matrix = np.zeros((F, H))

        # Data

        # mean wages
        mean_w = (1 - gamma_nr)*W_r + gamma_nr*W_nr
        self.mean_w, self.u_n = mean_w, 0
        self.mean_r_w, self.mean_nr_w, self.mean_w_arr = W_r, W_nr, np.zeros(T)
        self.median_w_arr = np.zeros(T)

        # unemployment
        self.mean_p_arr = np.zeros(T)
        self.mean_p_arr[-1] = np.mean([f.p for f in self.f_arr])
        self.u_r_arr, self.mean_r_w_arr, self.mean_nr_w_arr = np.zeros(T), np.zeros(T), np.zeros(T)
        self.mean_nominal_w_arr = np.zeros(T)
        self.mean_r_w_arr[-1], self.mean_nr_w_arr[-1] = W_r, W_nr
        self.ur_r_arr, self.unr_r_arr = np.zeros(T), np.zeros(T)
        self.u_n = 0

        self.SAVINGS = np.array(T)

        # GDP and open vacancies
        self.GDP, self.open_vs = np.zeros(T), np.zeros(T)

        # produced
        self.Y_arr = np.zeros(T)
        self.DC_arr, self.C_arr, self.DY_arr = np.zeros(T), np.zeros(T), np.zeros(T)
        self.INV_arr = np.zeros(T)


        # surviving and default firm masks
        self.default_fs, self.active_fs = np.full(F, False, dtype=bool), np.full(F, True, dtype=bool)

        # share of nr in r jobs, share of inactive firms
        self.share_nr_in_r = np.zeros(T)
        self.share_inactive = np.zeros(T)

        self.n_refinanced = np.zeros(T)

        # decile ratios
        self.nine_to_five, self.five_to_one, self.nine_to_one = np.zeros(T), np.zeros(T), np.zeros(T)

        self.wage_variance = np.zeros(T)

    def data_collector(self):
        """
        Collect and store simulation data for the current period.
        
        This method computes and stores key economic indicators:
        - Unemployment rates (overall, routine, non-routine)
        - Wages (mean, median, by worker type)
        - GDP and output
        - Firm activity metrics
        - Skill mismatch indicators
        - Income inequality measures
        """

        ur_n = self.H_r - np.sum(self.emp_matrix[:, self.routine_arr])
        unr_n = self.H_nr - np.sum(self.emp_matrix[:, self.non_routine_arr])
        u_n = self.H - np.sum(self.emp_matrix)

        self.share_nr_in_r[self.t] = np.sum(self.emp_matrix[:, np.logical_and(np.invert(self.nr_job_arr), self.non_routine_arr)])

        self.u_n = u_n

        self.u_r_arr[self.t] = u_n/self.H
        self.ur_r_arr[self.t] = ur_n/self.H_r
        self.unr_r_arr[self.t] = unr_n / self.H_nr

        self.Y_arr[self.t] = np.sum(np.array([f.y + f.inv for f in self.f_arr]))
        self.mean_p_arr[self.t] = np.sum(np.array([f.p * (f.y + f.inv) for f in self.f_arr])) / self.Y_arr[self.t]

        self.DC_arr[self.t] = np.sum([h.d_c for h in self.h_arr])
        self.C_arr[self.t] = np.sum([h.c for h in self.h_arr])
        self.DY_arr[self.t] = np.sum([f.d_y for f in self.f_arr])
        self.INV_arr[self.t] = np.sum([f.inv for f in self.f_arr])

        # get mean wages
        wages = np.array([h.w for h in self.h_arr])
        wages = wages[wages > 0]
        self.mean_w_arr[self.t] = (np.sum(wages)/(self.H-u_n))/self.mean_p_arr[self.t]
        self.median_w_arr[self.t] = np.median(wages) / self.mean_p_arr[self.t]
        self.mean_nominal_w_arr[self.t] = (np.sum(wages) / (self.H - u_n))

        r_wages = np.array([h.w for h in self.h_arr[self.routine_arr]])
        r_wages = r_wages[r_wages > 0]
        self.mean_r_w = np.sum(r_wages)/(self.H_r-ur_n)
        self.mean_r_w_arr[self.t] = self.mean_r_w/self.mean_p_arr[self.t]

        nr_wages = np.array([h.w for h in self.h_arr[self.non_routine_arr]])
        nr_wages = nr_wages[nr_wages > 0]
        self.mean_nr_w = np.sum(nr_wages)/(self.H_nr-unr_n)
        self.mean_nr_w_arr[self.t] = self.mean_nr_w/self.mean_p_arr[self.t]

        # get GDP
        self.GDP[self.t] = np.sum(np.array([f.y*f.p for f in self.f_arr]))

        # open vacancies
        self.open_vs[self.t] = np.sum(np.array([(f.v_r>0)*f.v_r + (f.v_nr>0)*f.v_nr for f in self.f_arr ]))

        # share of default firms
        n_def = np.sum(self.default_fs)
        self.share_inactive[self.t] = n_def/self.F

        # decile ratios
        nine_to_five = np.percentile(wages, 90)/np.percentile(wages, 50)
        five_to_one = np.percentile(wages, 50)/np.percentile(wages, 10)
        nine_to_one = np.percentile(wages, 90)/np.percentile(wages, 10)

        # wage variance
        self.wage_variance[self.t] = np.var(wages)

        self.nine_to_five[self.t], self.five_to_one[self.t] = nine_to_five, five_to_one
        self.nine_to_one[self.t] = nine_to_one

    def step_function(self):
        """
        Execute one simulation period.
        
        This is the main time-step function that coordinates all agent
        decisions and market processes in the correct sequence:
        
        1. Process fired workers
        2. Wage decisions
        3. Household consumption decisions
        4. Firm production decisions
        5. Labor market matching (firing and hiring)
        6. Goods market matching
        7. Profit calculation and dividend distribution
        8. Firm refinancing
        9. Wage payments and unemployment updates
        10. Data collection
        11. Minimum wage adjustment
        """

        if self.t % 50 == 0:
            print("Period: {}".format(self.t))

        # count unemployed households
        count_fired_time(self.h_arr)
        if self.t > 0:
            count_unemployed_hs(self.h_arr, self.t)

        # fired workers loose job
        fired_workers_loose_job(self.h_arr, self.f_arr, self.emp_matrix, self.nr_job_arr, self.t)

        wage_decisions(self)

        household_decisions(self)
        firm_decisions(self)

        run_labor_market(self)
        run_goods_market(self)

        firm_profits_and_dividends(self)
        hh_refin_firms(self)

        # defaulted firms, pay remaining wage bills
        unemp_arr = firms_pay_employees(self.f_arr, self.h_arr, self.default_fs, self.emp_matrix)
        set_W_fs(self.f_arr, self.emp_matrix, self.nr_job_arr, self.h_arr)

        update_Af(self.f_arr, self.tol)
        update_Ah(self.h_arr)

        for h in self.h_arr[unemp_arr]:
            get_unemployed(h, self.nr_job_arr, self.emp_matrix, self.t)

        self.data_collector()

        if self.t % 1 == 0:
            wages = np.array([h.w for h in self.h_arr])
            median_w = np.median(wages)
            self.min_w = self.min_w_par*median_w

        # firms loose employees
        for f in self.f_arr[self.default_fs]:
            f.n_r_fired, f.n_nr_fired = 0, 0

        # bug-check
        bug_check = np.sum(self.emp_matrix[self.default_fs, :]) > 0
        if bug_check:
            print("STOP! Here is a bug, defaulted firm still have employees!")
            print("You should check 'default_firms()' and 'hh_refin_firms()'")

        # Debug check - only if household 66 exists
        if len(self.h_arr) > 66 and self.h_arr[66].d_w == 0:
            print(self.h_arr[66].d_w)
        self.t += 1

    def run(self):
        """
        Run the complete simulation.
        
        Initializes employment relationships and executes the step_function
        for T periods, collecting data at each step.
        """
        # initialize employment
        set_W_fs(self.f_arr, self.emp_matrix, self.nr_job_arr, self.h_arr)

        for t in range(self.T):
            self.step_function()


