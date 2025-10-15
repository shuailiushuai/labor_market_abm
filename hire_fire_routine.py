"""
Hiring and Firing Module for Routine Workers

This module implements labor market mechanisms for routine jobs:
- Worker applications for routine positions
- Firm hiring decisions for routine workers
- Firm firing decisions for routine workers
- Wage negotiations and job matching

Routine workers are those with lower skills who perform
standardized tasks that can potentially be automated.

Author: Labor Market ABM Project
"""

from toolbox import *
from sys import exit


###############################################################################
# ROUTINE WORKER HIRING AND FIRING
###############################################################################


###############################################################################
# APPLICATION FUNCTIONS
###############################################################################

def firms_sort_r_applications(f_arr):
    """
    Sort routine job applications by desired wage.
    
    Firms sort applications in ascending order of desired wages,
    preferring candidates with lower wage demands.
    
    Parameters
    ----------
    f_arr : np.ndarray
        Array of firm agents
    """
    for f in f_arr:
        f.apps_r = f.apps_r.reshape(int(len(f.apps_r) / 2), 2)
        # Firms sort by desired wages (ascending)
        wages = f.apps_r[:, 1]
        sorted_ids = np.argsort(wages)
        f.apps_r = f.apps_r[sorted_ids]


def hs_send_r_apps(f_arr, h_arr, chi, H, beta):
    """
    Households send applications for routine jobs.
    
    Both routine and non-routine type households can apply for routine jobs.
    However, non-routine workers who currently have non-routine jobs do not
    apply (to avoid skill mismatch). Desired wages are adjusted by work
    experience using the beta parameter.
    
    Parameters
    ----------
    f_arr : np.ndarray
        Array of firm agents
    h_arr : np.ndarray
        Array of household agents
    chi : float
        Fraction of firms observed by each household (0 to 1)
    H : int
        Total number of households
    beta : float
        Experience premium parameter (wage multiplier)
    """
    h_indices = np.array([h.id for h in h_arr])
    f_ids = np.arange(len(f_arr))
    h_arr_shuffled = h_arr[rd.choice(h_indices, H, replace=False)]
    subN = get_N_sub(chi, len(f_ids))
    for h in h_arr_shuffled:
        boolean = (not h.routine) and h.nr_job # non routine worker who has a non routine job
        if not boolean: # only apply if not a nr worker with an nr job
            # write job application
            application = (h.id, (beta**h.exp)*h.d_w)
            # oberve  random subset of firms
            f_arr_shuffled = f_arr[rd.choice(f_ids, subN, replace=False)]
            # send application to observed firms
            for f in f_arr_shuffled:
                f.apps_r = np.insert(f.apps_r, 0, application)
    firms_sort_r_applications(f_arr[f_ids])


###############################################################################
# FIRING FUNCTIONS
###############################################################################


def get_r_fired_ids(f, emp_ids):
    """
    Get household IDs of routine workers to be fired.
    
    Parameters
    ----------
    f : Firm
        Firm agent
    emp_ids : np.ndarray
        Indices of employees in firm's routine worker array
        
    Returns
    -------
    np.ndarray
        Household IDs of workers to be fired
    """
    return f.r_employees[emp_ids].astype(int)


def f_fires_r_workers(h_arr, fired_ids, f):
    """
    Mark routine workers as fired.
    
    Parameters
    ----------
    h_arr : np.ndarray
        Array of household agents
    fired_ids : np.ndarray
        IDs of households being fired
    f : Firm
        Firm agent doing the firing
    """
    for h in h_arr[fired_ids]:
        h.fired = True
        f.n_r_fired += 1


def firms_fire_r_workers(v_mat, h_arr, f_arr, emp_mat, nr_job_arr):
    """
    Fire routine workers from firms with negative vacancies.
    
    Firms with v_r < 0 need to fire workers. Firms fire the routine
    workers with the highest wages first (last-in-first-out by wage).
    
    Parameters
    ----------
    v_mat : np.ndarray
        Vacancy matrix with columns [firm_id, v_r, v_nr]
    h_arr : np.ndarray
        Array of household agents
    f_arr : np.ndarray
        Array of firm agents
    emp_mat : np.ndarray
        Employment matrix (F x H)
    nr_job_arr : np.ndarray
        Boolean array indicating non-routine jobs
    """
    f_mask = v_mat[:, 1] < 0  # Firms that need to fire routine workers
    val = np.sum(f_mask)
    h_inds = np.arange(len(h_arr))
    if val > 0:
        fire_arr = v_mat[f_mask]
        ids = fire_arr[:, 0]
        n_fire_arr = fire_arr[:, 1] * (-1)
        for i in range(len(ids)):
            # Get firm ID and number of workers to fire
            f_id, n = int(ids[i]), int(n_fire_arr[i])
            emp_mask = emp_mat[f_id, :] > 0
            # Get routine employees
            emp_ids = h_inds[np.logical_and(emp_mask, np.invert(nr_job_arr))]
            # Look at wages of the employees
            wages = np.array([h.w for h in h_arr[emp_ids]])
            # Fire employees with highest wages
            mask = np.argsort(wages)[-n:]
            fired_ids = emp_ids[mask]
            f_fires_r_workers(h_arr, fired_ids, f_arr[f_id])


###############################################################################
# HIRING FUNCTIONS
###############################################################################


def remove_r_apps_from_queues(f_arr, chosen_apps):
    """
    Remove accepted applicants from all firms' application queues.
    
    When a worker is hired, their applications to other firms
    are withdrawn.
    
    Parameters
    ----------
    f_arr : np.ndarray
        Array of firm agents
    chosen_apps : list or np.ndarray
        IDs of households that have been hired
    """
    for f in f_arr:
        f_h_app_ids = f.apps_r[:, 0].astype(int)
        if len(f_h_app_ids) > 0:
            bool_arr = np.array([not app_id in chosen_apps for app_id in f_h_app_ids])
            f.apps_r = f.apps_r[bool_arr]


def employ_r_apps(h_arr, emp_mat, app_mat, nr_job_arr, f, lambda_LM, min_w, t):

    for h in h_arr:
        h.job_offer[t] = 1
        # delete all applications
        app_mat[:, h.id] = np.zeros(len(app_mat[:, h.id]))
        if np.sum(emp_mat[:, h.id]) > 0:
            Pr = Pr_LM(h.w, h.d_w, lambda_LM)
            switch = bool(draw_one(Pr))
            if switch:
                f.v_r -= 1
                # delete from old r job
                emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))
                # household gets employed
                emp_mat[f.id, h.id] = 1
                h.w = np.maximum(h.d_w, min_w)
                # update_desired wage in case of a minimum wage
                h.d_w = h.w
                h.fired_time = 0
                h.fired = False
                nr_job_arr[h.id] = False
        else:
            f.v_r -= 1
            h.u[t] = 0
            # household gets employed
            emp_mat[f.id, h.id] = 1
            h.w = np.maximum(h.d_w, min_w)
            # update_desired wage in case of a minimum wage
            h.d_w = h.w
            nr_job_arr[h.id] = False


def delete_from_old_r_job(h, f_arr):
    f_id = h.employer_id # get id of old employer
    emp_arr = f_arr[f_id].r_employees
    h_i = np.where(emp_arr == h.id)[0][0] # get index in emp_arr
    f_arr[f_id].r_employees = np.delete(emp_arr, h_i)


def firms_employ_r_applicants(m):
    f_arr, h_arr, lambda_LM, min_w, t = m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t
    emp_matrix = m.emp_matrix
    app_matrix = m.app_matrix

    # determine whether nr workers can apply for r jobs or not
    if m.nr_to_r:
        bool_arr = np.ones(m.H) > 0  # everyone
    else:
        bool_arr = m.routine_arr

    # 1. get vacancies
    # v_arr = np.array([f.v for f in f_arr])

    # 2. get ids of demand side
    f_ids = np.arange(len(f_arr))

    # 3. shuffle ids
    rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)

    # 4. shuffle vacancies using shuffled ids
    rand_v_arr = np.array([f.v_r for f in f_arr[rand_f_ids]])

    # 5. extract positive vacancy numbers

    # get demanded wages
    d_wages = np.asarray([h.d_w for h in m.h_arr])
    h_ids = np.arange(len(h_arr))

    val = True

    while val:
        # print("in 'firms_employ_r_applicants'")
        v_arr = np.array([f.v_r if f.v_r > 0 else 0 for f in f_arr])
        # change this if you want to include nr workers
        # val_arr = v_arr @ app_matrix[:, bool_arr]
        # print("val_arr: {}".format(val_arr))
        val = np.sum(v_arr @ app_matrix[:, bool_arr])

        for i in range(len(f_ids)):
            id, v = int(rand_f_ids[i]), int(rand_v_arr[i])
            if np.sum(app_matrix[id, bool_arr]) * (v > 0) > 0:
                # get app ids
                applied = app_matrix[id, :] > 0  # look at all applicants
                # take ids of workers that have applied AND DON'T HAVE A NR JOB
                mask = np.logical_and(applied, bool_arr)
                h_app_ids = h_ids[mask]
                # sort app ids from lowest to highest wrt to wages
                sorted_app_ids = np.argsort(d_wages[h_app_ids])
                # choose cheapest ids
                sorted_h_ids = h_app_ids[sorted_app_ids]  # sort applicant ids
                chosen_apps = sorted_h_ids[0:v]
                employ_r_apps(h_arr[chosen_apps], emp_matrix, app_matrix, m.nr_job_arr,
                              f_arr[id], lambda_LM, min_w, t)

        update_N(f_arr, emp_matrix, m.nr_job_arr)
        update_v(f_arr)
        rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)
        rand_v_arr = np.array([f.v_r for f in f_arr[rand_f_ids]])
