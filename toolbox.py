"""
Utility Functions for Labor Market ABM

This module provides core utility functions for the labor market agent-based model:
- Employment management functions
- Wage and price update functions  
- Expectation formation
- Household decision functions
- Firm decision functions
- Probability and random sampling utilities

Author: Labor Market ABM Project
"""

import numpy as np
import numpy.random as rd


def delete_from_old_r_job2(h, f_arr):
    """
    Remove a household from their old routine job.
    
    Parameters
    ----------
    h : Household
        Household agent to be removed
    f_arr : np.ndarray
        Array of firm agents
    """
    f_id = h.employer_id  # Get ID of old employer
    emp_arr = f_arr[f_id].r_employees
    h_i = np.where(emp_arr == h.id)[0][0]  # Get index in employee array
    f_arr[f_id].r_employees = np.delete(emp_arr, h_i)
    f_arr[f_id].n_r_fired -= 1


def delete_from_old_nr_job2(h, f_arr):
    """
    Remove a household from their old non-routine job.
    
    Parameters
    ----------
    h : Household
        Household agent to be removed
    f_arr : np.ndarray
        Array of firm agents
    """
    f_id = h.employer_id  # Get ID of old employer
    emp_arr = f_arr[f_id].nr_employees
    h_i = np.where(emp_arr == h.id)[0][0]  # Get index in employee array
    f_arr[f_id].nr_employees = np.delete(emp_arr, h_i)
    f_arr[f_id].n_nr_fired -= 1


def update_wr_wnr_bar(w_bar, mean_w, phi_w):
    """
    Update average wage using exponential smoothing.
    
    Parameters
    ----------
    w_bar : float
        Current average wage
    mean_w : float
        Current period mean wage
    phi_w : float
        Smoothing parameter (0 to 1)
        
    Returns
    -------
    float
        Updated average wage
    """
    w_bar = phi_w*mean_w + (1-phi_w)*w_bar
    return w_bar


def expectation(z, z_e, lambda_exp):
    """
    Adaptive expectations formation for a generic variable.
    
    This function implements adaptive expectations where agents update
    their expectations based on the previous period's forecast error.
    
    Parameters
    ----------
    z : float
        Previous period actual observation
    z_e : float
        Previous period expectation
    lambda_exp : float
        Adjustment/learning parameter (0 to 1)
        Higher values = faster adjustment to errors
        
    Returns
    -------
    float
        Updated expectation for current period
    """
    error = z - z_e
    return z_e + lambda_exp*error


def draw_one(P):
    """
    Draw binary outcome with given probability.
    
    Parameters
    ----------
    P : float
        Probability of drawing 1 (between 0 and 1)
        
    Returns
    -------
    int
        1 with probability P, 0 with probability (1-P)
    """
    num = rd.uniform()
    return 0*(num >= P) + 1*(num < P)


def get_N_sub(chi, N):
    """
    Calculate subsample size for labor market search.
    
    Parameters
    ----------
    chi : float
        Sampling fraction (0 to 1)
    N : int
        Total population size
        
    Returns
    -------
    int
        Subsample size (at least 1 if chi*N > 0, otherwise 0)
    """
    if chi * N >= 1:
        return int(chi*N)
    elif chi * N > 0:
        return 1
    else:
        return 0


def init_emp_mat(F, H, u_r, h_arr):
    """
    Initialize employment matrix with random initial employment.
    
    Parameters
    ----------
    F : int
        Number of firms
    H : int
        Number of households
    u_r : float
        Initial unemployment rate (0 to 1)
    h_arr : np.ndarray
        Array of household agents
        
    Returns
    -------
    np.ndarray
        Employment matrix of shape (F, H) where entry (i,j) = 1
        if household j is employed by firm i, 0 otherwise
    """
    N = np.int32(H*(1-u_r))  # Number of initially employed
    emp_matrix = np.zeros((F, H))
    rand_f_ids = rd.permutation(N)
    rand_h_ids = rd.choice(np.arange(H), N, replace=False)

    for h_id, perm_num in zip(rand_h_ids, rand_f_ids):
        f_id = perm_num % F  # Distribute evenly across firms
        emp_matrix[f_id, h_id] = 1
        h_arr[h_id].u[0] = 0  # Mark as employed
    return emp_matrix


# Test expectation function (for verification)
z = 4
z_e = 6
lambda_exp = 0.5
expectation(z, z_e, lambda_exp)


###############################################################################
# HOUSEHOLD FUNCTIONS
###############################################################################

def h_send_apps(app_mat, H, F, N_app):
    """
    Households send job applications to randomly selected firms.
    
    Parameters
    ----------
    app_mat : np.ndarray
        Application matrix of shape (F, H) to be filled
        Entry (i,j) = 1 if household j applies to firm i
    H : int
        Number of households
    F : int
        Number of firms
    N_app : int
        Number of applications each household sends
    """
    f_ids = np.arange(F)
    # Function to draw N_app firm IDs without replacement
    sub_f_ids = lambda: rd.choice(f_ids, N_app, replace=False)

    for i in range(H):
        app_mat[sub_f_ids(), i] = 1


def get_unemployed(h, nr_job_arr, emp_mat, t):
    """
    Set household to unemployed status.
    
    Parameters
    ----------
    h : Household
        Household agent becoming unemployed
    nr_job_arr : np.ndarray
        Array tracking non-routine job status
    emp_mat : np.ndarray
        Employment matrix (F x H)
    t : int
        Current time period
    """
    emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))  # Clear employment
    h.u[t] = 1  # Set unemployed
    nr_job_arr[h.id] = False  # Clear job type
    h.w = 0  # No wage when unemployed
    h.fired = False  # Clear fired status
    h.fired_time = 0  # Reset fired timer


def count_unemployed_hs(h_arr, t):
    """
    Update unemployment status for all households.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    t : int
        Current time period
    """
    for h in h_arr:
        unemp = (h.u[t-1] == 1)  # Check if was unemployed last period
        h.u[t] = 1*unemp  # Carry forward unemployment status


def Pr_LM(w_old, w_new, lambda_LM):
    """
    Calculate probability of accepting a job offer based on wage change.
    
    Uses an exponential function to model job acceptance probability.
    Workers are more likely to accept jobs with higher wages.
    
    Parameters
    ----------
    w_old : float
        Current/previous wage
    w_new : float
        Offered wage
    lambda_LM : float
        Labor market friction parameter
        
    Returns
    -------
    float
        Probability of accepting job (0 to 1)
    """
    diff = (w_old - w_new)/w_old
    if diff < 0:  # New wage is higher
        return 1 - np.exp(lambda_LM*diff)
    else:  # New wage is lower or equal
        return 0


def update_exp(h_arr, t, diff):
    """
    Update work experience for all households.
    
    Experience is calculated as the sum of employed periods
    in the recent past (last 'diff' periods).
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    t : int
        Current time period
    diff : int
        Number of periods to look back for experience calculation
    """
    if t < diff:
        # Use all available history if t < diff
        for h in h_arr:
            h.exp = np.sum(1 - h.u[0: t + 1])
    else:
        # Use last 'diff' periods
        for h in h_arr:
            h.exp = np.sum(1 - h.u[t - 3: t + 1])


def update_w(h_arr, emp_mat, min_w):
    """
    Enforce minimum wage for employed households.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    emp_mat : np.ndarray
        Employment matrix (F x H)
    min_w : float
        Minimum wage level
    """
    emp_m = np.sum(emp_mat, axis=0) > 0  # Find employed households
    for h in h_arr[emp_m]:
        h.w = np.maximum(h.w, min_w)  # Apply minimum wage


def update_d_w(h_arr, emp_mat, sigma_chi, min_w, t):
    """
    Update desired wages for all households.
    
    Employed households with job offers increase desired wages,
    while unemployed/rejected households decrease desired wages.
    Uses chi-squared distribution for stochastic variation.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    emp_mat : np.ndarray
        Employment matrix (F x H)
    sigma_chi : float
        Wage adjustment volatility parameter
    min_w : float
        Minimum wage floor
    t : int
        Current time period
    """
    emp_arr = np.sum(emp_mat, axis=0)
    for h in h_arr:
        # If unemployed or no job offer, decrease desired wage
        if emp_arr[h.id] == 0 or h.job_offer[t-1] == 0:
            h.d_w = h.d_w*(1 - rd.chisquare(1)*sigma_chi)
        else:  # If employed with job offer, increase desired wage
            h.d_w = h.d_w * (1 + rd.chisquare(1) * sigma_chi)
        # Enforce minimum and positive wage
        h.d_w = np.max([h.d_w, min_w, 0.1])


def update_w_e(h_arr, lambda_exp):
    """
    Update wage expectations using adaptive expectations.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    lambda_exp : float
        Expectation adjustment parameter
    """
    for h in h_arr:
        h.w_e = expectation(h.w, h.w_e, lambda_exp)


def update_Ah(h_arr):
    """
    Update household assets with income.
    
    Households receive wage income (adjusted by payment rate in case
    of firm default) and dividend income.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    """
    for h in h_arr:
        h.A += (h.par * h.w + h.div)


def update_d_c(h_arr, alpha_1, alpha_2):
    """
    Update desired consumption for all households.
    
    Consumption depends on current income (wages + dividends)
    and accumulated wealth.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    alpha_1 : float
        Marginal propensity to consume from income
    alpha_2 : float
        Marginal propensity to consume from wealth
    """
    for h in h_arr:
        # Consumption = alpha_1 * (income/price) + alpha_2 * (wealth/price)
        h.d_c = alpha_1*(np.maximum((h.w + h.div - h.refin), 0)/h.p_e) + alpha_2*(h.A/h.p_e)


def clear_expenditure(h_arr):
    for h in h_arr:
        h.expenditure = 0
        h.c = 0


def update_h_p(h_arr):
    for h in h_arr:
        if (h.expenditure > 0) and (h.c>0):
            h.p = h.expenditure/h.c


def get_Ah_weights(h_arr):
    wealth_arr = np.array([h.A for h in h_arr])
    wealth_tot = np.sum(wealth_arr)
    if wealth_tot > 0:
        return wealth_arr/wealth_tot
    else:
        return np.ones(len(h_arr))*(1/len(h_arr))


def distribute_dividends(h_arr, f_arr):
    DIV = np.sum(np.array([f.div for f in f_arr]))
    weights = get_Ah_weights(h_arr)
    div = weights*DIV
    for h in h_arr:
        h.div = div[h.id]


def update_div_h_e(h_arr, lambda_exp):
    for h in h_arr:
        if h.div > 0:
            h.div_e = expectation(h.div, h.div_e, lambda_exp)


def count_fired_time(h_arr):
    for h in h_arr:
        if h.fired:
            h.fired_time += 1


def fired_workers_loose_job(h_arr, f_arr, emp_mat, nr_job_arr, t):
    for h in h_arr:
        if h.fired and (h.fired_time == h.fired_time_max):
            # delete from old employer
            f_id = np.nonzero(emp_mat[:, h.id])[0][0]
            get_unemployed(h, nr_job_arr, emp_mat, t)
            f_arr[f_id].n_nr_fired -= 1


def update_p_e(h_arr, lambda_exp):
    for h in h_arr:
        h.p_e = expectation(h.p, h.p_e, lambda_exp)


#########################################################################################################
###################### TOOLS FOR FIRMS ##################################################################
#########################################################################################################


def get_employee_IDs(N_arr):
    return np.nonzero(N_arr)[0]


def update_d_N(f_arr, mu_r, mu_nr, sigma):
    for f in f_arr:
        # 1. Case
        Omega = get_Omega(f.Wr_e, f.Wnr_e, mu_r, mu_nr, sigma)
        f.d_Nnr = np.round(get_d_Nnr_non_binding(f.d_y, Omega, mu_r, mu_nr, sigma))
        f.d_Nr = np.round(get_d_Nr(f.d_Nnr, Omega))

        # check if feasible
        C = get_total_costs(f.Wr_e, f.Wnr_e, f.d_Nr, f.d_Nnr)
        if C > f.A + f.p*f.s_e: # check if feasible, IDEA: consider expected net earnings (after dividends)
            f.d_Nnr = np.round(get_d_Nnr_binding(f.A, f.Wr_e, f.Wnr_e, Omega))
            f.d_Nr = np.round(get_d_Nr(f.d_Nnr, Omega))

def update_N(f_arr, emp_matrix, nr_job_arr):
    Nr_arr = np.sum(emp_matrix[:, np.invert(nr_job_arr)], axis=1)
    Nnr_arr = np.sum(emp_matrix[:, nr_job_arr], axis=1)
    for f in f_arr:
        f.Nr = Nr_arr[f.id]
        f.Nnr = Nnr_arr[f.id]


def update_v(f_arr):
    for f in f_arr:
        f.v_r = f.d_Nr - f.Nr
        f.v_nr = f.d_Nnr - f.Nnr


def update_s_e(f_arr, lambda_exp):
    for f in f_arr:
        f.s_e = expectation(f.s, f.s_e, lambda_exp)


def set_W_fs(f_arr, emp_mat, nr_job_arr, h_arr):
    h_inds = np.arange(len(h_arr))
    for f in f_arr:
        emp_mask = emp_mat[f.id, :] > 0
        if f.Nr > 0:
            r_emps = h_inds[np.logical_and(emp_mask, np.invert(nr_job_arr))]
            wages_r = np.array([h.w for h in h_arr[r_emps]])
            f.Wr_tot = np.sum(wages_r)
            f.Wr = f.Wr_tot / f.Nr
        else:
            f.Wr_tot = 0

        if f.Nnr > 0:
            nr_emps = h_inds[np.logical_and(emp_mask, nr_job_arr)]
            wages_nr = np.array([h.w for h in h_arr[nr_emps]])
            f.Wnr_tot = np.sum(wages_nr)
            f.Wnr = f.Wnr_tot / f.Nnr
        else:
            f.Wnr_tot = 0


def update_Wr_e(f_arr, min_w, lambda_exp):
    for f in f_arr:
        if f.Wr > 0:
            f.Wr_e = expectation(f.Wr, f.Wr_e, lambda_exp)
            f.Wr_e = np.maximum(f.Wr_e, min_w)


def update_Wnr_e(f_arr, min_w, lambda_exp):
    for f in f_arr:
        if f.Wr > 0:
            f.Wnr_e = expectation(f.Wnr, f.Wnr_e, lambda_exp)
            f.Wnr_e = np.maximum(f.Wnr_e, min_w)


def update_m(f_arr, sigma_chi):
    for f in f_arr:
        if f.inv < f.nu*f.s:
            f.m = f.m*(1+rd.chisquare(1)*sigma_chi)
        elif f.inv > f.nu*f.s:
            f.m = f.m*(1-rd.chisquare(1)*sigma_chi)
        f.m = np.maximum(0.01, f.m)


def update_m2(f_arr, param):
    for f in f_arr:
        if f.inv < f.nu*f.s:
            f.m = f.m*(1+param*rd.uniform())
        elif f.inv > f.nu*f.s:
            f.m = f.m*(1-param*rd.uniform())


def update_uc_arr(f_arr, t):
    for f in f_arr:
        C = (f.Wr_e*f.d_Nr + f.Wnr_e*f.d_Nnr)
        f.uc_arr[t] = C/f.d_y
        if f.uc_arr[t] == 0:
            f.uc_arr[t] = f.uc_arr[t-1]


def update_p(f_arr, t):
    for f in f_arr:
        f.p = f.uc_arr[t]*(1+f.m)


def update_d_y(f_arr, mu_r, mu_nr, sigma):
    for f in f_arr:
        if not f.default:
            min_Nnr = 1
            Omega = np.round(get_Omega(f.Wr_e, f.Wnr_e, mu_r, mu_nr, sigma))
            min_Nr = Omega * min_Nnr
            min_d_y = CES_production(min_Nr, min_Nnr, mu_r, mu_nr, sigma)
            d_y = np.maximum(f.s_e * (1 + f.nu) - f.inv, min_d_y)
            f.d_y_diff = d_y - f.d_y
            f.d_y = d_y
        else:
            f.d_y = 0

###### Production Decision ##########


def firms_produce(f_arr, mu_r, mu_nr, sigma):
    for f in f_arr:
        f.y = CES_production(f.Nr, f.Nnr, mu_r, mu_nr, sigma)


def CES_production(Nr, Nnr, mu_r, mu_nr, sigma):
    rho = (sigma - 1)/sigma
    bool_1, bool_2 = Nr > 0, Nnr > 0
    if bool_1:
        X_1 = (mu_r*Nr)**rho
    else:
        X_1 = 0

    if bool_2:
        X_2 = (mu_nr*Nnr)**rho
    else:
        X_2 = 0

    if X_1 + X_2 > 0:
        CES = (X_1 + X_2)**(1/rho)
    else:
        CES = 0

    return CES

# Omega is the ratio of R to NR workers that's optimal
def get_Omega(Wr_e, Wnr_e, mu_r, mu_nr, sigma):
    rho = (sigma - 1)/sigma
    X_1, X_2, X_3 = Wr_e / mu_r, Wnr_e / mu_nr, mu_nr / mu_r
    Omega = ((X_1/X_2)**(1/(rho-1)))*X_3
    return Omega


def get_d_Nr(Nnr, Omega):
    return  Nnr * Omega

# 1. Case: Budget constraint is non-binding
def get_d_Nnr_non_binding(d_y, Omega, mu_r, mu_nr, sigma):
    rho = (sigma - 1)/sigma
    X_1, X_2 = (d_y/mu_nr), ((mu_r/mu_nr)*Omega)**rho
    Nnr = X_1*((1 + X_2)**(-1/rho))
    return Nnr

#  check whether the 1. Case is feasible
def get_total_costs(Wr_e, Wnr_e, Nr, Nnr):
    return Wr_e*Nr + Wnr_e*Nnr


# 2. Case: Budget constraint is binding, B is the firm f's budget
def get_d_Nnr_binding(B, Wr_e, Wnr_e, Omega):
    X_1 = Wr_e*Omega
    Nnr = B*((X_1 + Wnr_e)**(-1))
    return Nnr


def update_pi(f_arr):
    for f in f_arr:
        f.pi = f.p*f.s - f.Wr_tot - f.Wnr_tot


def update_div_f(f_arr):
    for f in f_arr:
        if f.pi > 0:
            f.div = f.delta*f.pi
        else:
            f.div = 0


def update_delta(f_arr, sigma_chi):
    for f in f_arr:
        if (f.s_e * (1 + f.nu) - f.inv) > f.d_y:
            f.delta = f.delta * (1 - rd.chisquare(1) * sigma_chi)
        elif (f.s_e * (1 + f.nu) - f.inv) < f.d_y:
            f.delta = f.delta * (1 + rd.chisquare(1) * sigma_chi)

        f.delta = np.maximum(np.minimum(1, f.delta), 0)


def update_f_default(f_arr):
    for f in f_arr:
        default = (f.A + f.pi) <= 0
        f.default = default


def update_Af(f_arr, tol):
    for f in f_arr:
        f.A += f.s*f.p - f.par * (f.Wr_tot + f.Wnr_tot) - f.div
        if np.abs(f.A) <= tol:
            f.A = 0


def clear_s(f_arr):
    for f in f_arr:
        f.s = 0


def update_inv(f_arr):
    for f in f_arr:
        f.inv += f.y - f.s

