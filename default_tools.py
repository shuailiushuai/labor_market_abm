"""
Firm Default and Refinancing Tools for Labor Market ABM

This module handles firm financial distress, defaults, and refinancing:
- Identifying defaulted and surviving firms
- Processing wage payments during default
- Household-financed firm refinancing
- Cost distribution among households

Author: Labor Market ABM Project
"""

import numpy as np
import numpy.random as rd


def default_firms(f_arr):
    """
    Identify firms that have defaulted.
    
    A firm defaults if either:
    1. Assets + profits <= 0 (negative net worth)
    2. Production == 0 (no output)
    
    Parameters
    ----------
    f_arr : np.ndarray
        Array of firm agents
        
    Returns
    -------
    np.ndarray
        Boolean array where True indicates defaulted firms
    """
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr <= 0, y_arr == 0)


def surviving_firms(f_arr):
    """
    Identify firms that are still active (not defaulted).
    
    A firm survives if it has positive net worth AND positive production.
    
    Parameters
    ----------
    f_arr : np.ndarray
        Array of firm agents
        
    Returns
    -------
    np.ndarray
        Boolean array where True indicates surviving firms
    """
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr > 0, y_arr > 0)


def firms_pay_employees(f_arr, h_arr, default_fs, emp_mat):
    """
    Process wage payments from firms to employees.
    
    Firms pay wages to employees. Defaulted firms may only pay partial
    wages based on available resources. Employees of defaulted firms
    become unemployed.
    
    Parameters
    ----------
    f_arr : np.ndarray
        Array of firm agents
    h_arr : np.ndarray
        Array of household agents
    default_fs : np.ndarray
        Boolean array indicating defaulted firms
    emp_mat : np.ndarray
        Employment matrix (F x H)
        
    Returns
    -------
    np.ndarray
        Array of household IDs that become unemployed
    """
    unemp_arr = np.array([], dtype=np.int64)
    for f in f_arr:
        employees = np.nonzero(emp_mat[f.id, :])[0]
        if len(employees) > 0:
            # Calculate percentage of wage bills that can be paid
            par = np.maximum(f.A + f.s*f.p, 0) / (f.Wr_tot + f.Wnr_tot)
            f.par = np.minimum(1, par)
            for h in h_arr[employees]:
                h.par = f.par
                if default_fs[f.id]:
                    unemp_arr = np.append(unemp_arr, h.id)
    return unemp_arr


def pay_refin_cost(h_arr, weights, tol, tot_refin_cost):
    """
    Distribute firm refinancing costs among households.
    
    Households pay for firm bailouts proportionally to their wealth.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    weights : np.ndarray
        Wealth-based weight for each household
    tol : float
        Numerical tolerance for setting values to zero
    tot_refin_cost : float
        Total refinancing cost to distribute
    """
    for h in h_arr:
        cost = weights[h.id]*tot_refin_cost
        h.A -= cost
        h.refin = cost
        if np.abs(h.A) < tol:
            h.A = 0


def refin_firms(def_firms, surviving_firms, h_arr, n_refin, tol, t):
    """
    Refinance defaulted firms using household wealth.
    
    Defaulted firms are randomly selected for potential refinancing.
    Refinancing costs are distributed among households based on wealth.
    Refinanced firms receive new initial conditions drawn from
    surviving firm distributions.
    
    Parameters
    ----------
    def_firms : np.ndarray
        Array of defaulted firm agents
    surviving_firms : np.ndarray
        Array of surviving firm agents
    h_arr : np.ndarray
        Array of household agents
    n_refin : np.ndarray
        Counter array for number of refinanced firms per period
    tol : float
        Numerical tolerance
    t : int
        Current time period
    """
    wealth_arr = np.array([np.maximum(h.A, 0) for h in h_arr])
    wealth_tot = np.sum(wealth_arr)
    weights = wealth_arr / wealth_tot

    # Get distribution parameters from surviving firms
    Af_arr = np.array([f.A for f in surviving_firms])
    Af_init = np.percentile(Af_arr, 50)
    sigma_A = np.std(Af_arr)

    netA_arr = np.array([f.A + f.pi for f in def_firms])
    ids = np.arange(len(netA_arr))
    rand_ids = rd.choice(ids, len(netA_arr), replace=False)

    s_e_arr = np.array([f.s_e for f in surviving_firms])
    s_e_init = np.percentile(s_e_arr, 50)
    sigma_s_e = np.std(s_e_arr)

    Wr_e_arr = np.array([f.Wr_e for f in surviving_firms])
    Wnr_e_arr = np.array([f.Wnr_e for f in surviving_firms])
    Wr_e_init = np.percentile(Wr_e_arr, 50)
    Wnr_e_init = np.percentile(Wnr_e_arr, 50)
    sigma_Wr_e = np.std(Wr_e_arr)
    sigma_Wnr_e = np.std(Wnr_e_arr)

    m_arr = np.array([f.m for f in surviving_firms])
    m_init = np.percentile(m_arr, 50)
    sigma_m = np.std(m_arr)

    for id in rand_ids:
        refin = np.maximum((-1)*netA_arr[id] + np.maximum(Af_init + rd.randn()*sigma_A, 0), 0)
        wealth_tot -= refin
        if wealth_tot >= 0:
            def_firms[id].A += refin
            def_firms[id].default = False
            def_firms[id].s_e = np.maximum(s_e_init + rd.randn()*sigma_s_e, 1)
            def_firms[id].m = np.maximum(m_init + rd.randn()*sigma_m, 0.01)
            def_firms[id].Wr_e = np.maximum(Wr_e_init + rd.randn() * sigma_Wr_e, 0.1)
            def_firms[id].Wnr_e = np.maximum(Wnr_e_init + rd.randn() * sigma_Wnr_e, 0.1)
            n_refin[t] += 1

            pay_refin_cost(h_arr, weights, tol, refin)
