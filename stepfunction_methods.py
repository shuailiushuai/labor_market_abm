"""
Step Function Methods for Labor Market ABM

This module implements the main simulation step functions that coordinate
agent decisions and market processes:
- Wage decisions by households and firms
- Household consumption decisions
- Firm production and pricing decisions
- Labor market matching
- Goods market matching
- Profit distribution and firm refinancing

These functions are called sequentially in each simulation period.

Author: Labor Market ABM Project
"""

import numpy as np
from hire_fire_routine import *
from hire_fire_non_routine import *
from default_tools import *
from goods_market import gm_matching


def wage_decisions(m):
    """
    Process wage-related decisions for all agents.
    
    This function coordinates:
    1. Enforce minimum wage for employed workers
    2. Firms update average wage observations
    3. Firms update employment counts
    4. Firms form wage expectations
    5. Households update desired wages
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """
    # wage decisions
    update_w(m.h_arr, m.emp_matrix, m.min_w)
    set_W_fs(m.f_arr[m.active_fs], m.emp_matrix, m.nr_job_arr, m.h_arr)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr[m.active_fs], m.emp_matrix, m.nr_job_arr, m.h_arr)  # firms measure average wages paid to employees
    update_Wr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)  # firms build wage expectations
    update_Wnr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)
    update_d_w(m.h_arr, m.emp_matrix, m.sigma_w, m.min_w, m.t)  # households decide for desired wages


def household_decisions(m):
    """
    Process household decisions for the current period.
    
    This function coordinates:
    1. Update work experience
    2. Update price expectations
    3. Make consumption decisions
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """

    # households update work experience
    update_exp(m.h_arr, m.t, 4)

    # households make consumption decision
    update_p_e(m.h_arr, m.lambda_exp)
    update_d_c(m.h_arr, m.alpha_1, m.alpha_2)


def firm_decisions(m):
    """
    Process firm decisions for the current period.
    
    This function coordinates:
    1. Decide desired production levels
    2. Decide labor demand (routine and non-routine)
    3. Determine vacancies or firing needs
    4. Set markups and prices
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """
    # Firms decide for desired production
    update_d_y(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # Firms decide for labor demand
    update_d_N(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # Firms choose whether to hire or fire
    update_v(m.f_arr)

    # price decisions
    update_m(m.f_arr[m.active_fs], m.sigma_m)  # choose markup
    update_uc_arr(m.f_arr, m.t)
    update_p(m.f_arr[m.active_fs], m.t)


def run_labor_market(m):
    """
    Execute labor market matching for the current period.
    
    This function coordinates:
    1. Fire workers (routine and non-routine)
    2. Update employment counts
    3. Households send job applications
    4. Firms hire new workers (non-routine then routine)
    5. Update employment matrix
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """

    ## Labor market matching
    # get vacancies Fx2 matrix
    v_mat = np.array([(f.id, f.v_r, f.v_nr) for f in m.f_arr[m.active_fs]])
    firms_fire_r_workers(v_mat, m.h_arr, m.f_arr, m.emp_matrix, m.nr_job_arr)
    firms_fire_nr_workers(v_mat, m.h_arr, m.f_arr, m.emp_matrix, m.nr_job_arr)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)

    # households apply
    # hs_send_nr_apps(m.f_arr, m.h_arr[m.non_routine_arr], m.chi_L, m.H_nr, m.H_r, m.beta)
    # hs_send_r_apps(m.f_arr, m.h_arr[m.routine_arr], m.chi_L, m.H_r, m.beta)
    h_send_apps(m.app_matrix, m.H, m.F, m.N_app)

    # firms hire
    update_v(m.f_arr)
    firms_employ_nr_applicants(m)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr, m.emp_matrix, m.nr_job_arr, m.h_arr)

    firms_employ_r_applicants(m)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr, m.emp_matrix, m.nr_job_arr, m.h_arr)
    # m.app_matrix = np.zeros((m.F, m.H))


def run_goods_market(m):
    """
    Execute goods market matching for the current period.
    
    This function coordinates:
    1. Firms produce goods
    2. Clear sales and household expenditure from previous period
    3. Match households with firms for consumption
    4. Firms update sales expectations
    5. Firms update inventories
    6. Households update price observations
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """
    # firms produce goods
    firms_produce(m.f_arr, m.mu_r, m.mu_nr, m.sigma)

    # firms sell goods
    clear_s(m.f_arr)
    clear_expenditure(m.h_arr)
    gm_matching(m.f_arr, m.h_arr, m.chi_C, m.tol)

    # firms update sales expectations
    update_s_e(m.f_arr[m.active_fs], m.lambda_exp)
    update_inv(m.f_arr[m.active_fs])
    update_delta(m.f_arr[m.active_fs], m.sigma_delta)

    # households update mean prices
    update_h_p(m.h_arr)


def firm_profits_and_dividends(m):
    """
    Calculate firm profits and distribute dividends to households.
    
    This function:
    1. Calculates firm profits
    2. Determines dividends based on profit and payout rate
    3. Distributes dividends to households
    4. Households update dividend expectations
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """

    # firms calculate profits
    update_pi(m.f_arr)
    update_div_f(m.f_arr[m.active_fs])

    # surviving firms pay dividends
    distribute_dividends(m.h_arr, m.f_arr)


def hh_refin_firms(m):
    """
    Process firm defaults and household-financed refinancing.
    
    This function:
    1. Identifies defaulted and surviving firms
    2. Refinance selected defaulted firms using household wealth
    3. Update firm status arrays
    
    Parameters
    ----------
    m : Model
        The model instance containing all agents and parameters
    """
    # households refinance firms
    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)

    # Refinance defaulted firms
    refin_firms(m.f_arr[m.default_fs], m.f_arr[m.active_fs], m.h_arr, m.n_refinanced,
                m.tol, m.t)

    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)