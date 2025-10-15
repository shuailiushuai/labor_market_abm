import numpy as np
import numpy.random as rd
from sys import exit
import matplotlib.pyplot as plt
import seaborn as sns

class Agent(object):

    def __init__(self, _id, A_init, T):

        self.id = _id
        self.T = T
        self.A = A_init
        self.cash = A_init

        # taxes that agent has to pay
        self.income_tax = 0
        self.wealth_tax = 0

        # tolerance level for setting values to zero
        self.tol = 10**(-10)

        # percentage of wages paid respectively received
        self.par = 1

    def change_cash(self, c):
        self.cash += c
        if np.abs(self.cash) < self.tol:
            self.cash = 0

class Firm(Agent):

    def __init__(self, _id, A_init, T, y, nu, Wr_f, Wnr_f, delta, p, m, pi, div, uc):
        super(Firm, self).__init__(_id, A_init, T)

        # employees, number of employees n_f, total wage bill
        self.r_employees, self.nr_employees = np.array([]), np.array([])

        # sold goods, expected sales
        self.s, self.s_e = y, y

        # number of employees and desired number of umployees
        self.Nr, self.d_Nr = 0, 0
        self.Nnr, self.d_Nnr = 0, 0
        self.n_r_fired, self.n_nr_fired = 0, 0

        # average wage to pay
        self.Wr, self.Wnr = Wr_f, Wnr_f
        self.Wr_e, self.Wnr_e = Wr_f, Wnr_f

        # total wage bill
        self.Wr_tot, self.Wnr_tot = 0, 0

        self.apps_r, self.apps_nr = np.array([]), np.array([])

        # desired production, production and price
        self.d_y, self.y, self.p = y, y, p
        self.d_y_diff = 0
        self.uc_arr = np.zeros(T)
        self.uc_arr[-1] = uc

        # number of vacancies
        self.v_r, self.v_nr = 0, 0

        # inventories share, inventories and mark up
        self.nu, self.m = nu, m
        self.inv = nu*y

        # profits, profits after dividends and dividend rate
        self.pi, self.div, self.delta = pi, div, delta

        # boolean for default and
        # parameter for share of wage bills paid if default
        self.default, self.par = False, 1


class Household(Agent):

    def __init__(self, _id, A_init, T, routine, w, p, c, div, f_max):
        super(Household, self).__init__(_id, A_init, T)

        # dummy if TRUE -> routine type else -> non-routine
        self.routine = routine
        self.public_worker = False

        # Dummy: if True -> current job is non routine,
        #        else -> current job is routine
        self.nr_job = False

        # Dummy -> got job offer  at time t -> 1
        #       -> got no job offer at time t -> 0
        self.job_offer = np.zeros(T)

        self.fired = False
        self.fired_time = 0
        self.fired_time_max = f_max

        # unemployed dummy,
        self.u = np.zeros(T)
        self.u[0] = 1
        self.exp, self.employer_id = 0, None

        # desired wage and actual wage
        self.d_w, self.w = w, w
        self.w_e = w
        self.last_w = w

        # average price paid
        self.p, self.p_e = p, p

        # desired consumption, consumption
        self.d_c, self.c = c, c

        # consumption expenditure
        self.expenditure = c*p

        # dividend income
        self.div = div
        self.div_e = div

        self.refin = 0


def default_firms(f_arr):
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr <= 0, y_arr == 0)


def surviving_firms(f_arr):
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr > 0, y_arr > 0)


def firms_pay_employees(f_arr, h_arr, default_fs, emp_mat):
    unemp_arr = np.array([], dtype=np.int64)
    for f in f_arr:
        employees = np.nonzero(emp_mat[f.id, :])[0]
        if len(employees) > 0:
            par = np.maximum(f.A + f.s*f.p, 0) / (f.Wr_tot + f.Wnr_tot) # percentage of wage bills paid
            f.par = np.minimum(1, par)
            for h in h_arr[employees]:
                h.par = f.par
                if default_fs[f.id]:
                    unemp_arr = np.append(unemp_arr, h.id)
    return unemp_arr


def pay_refin_cost(h_arr, weights, tol, tot_refin_cost):
    for h in h_arr:
        cost = weights[h.id]*tot_refin_cost
        h.A -= cost
        h.refin = cost
        if np.abs(h.A) < tol:
            h.A = 0


#########################################################################################################
###################### DEFAULT TOOLS ####################################################################
#########################################################################################################

def refin_firms(def_firms, surviving_firms, h_arr, n_refin, tol, t):

    wealth_arr = np.array([np.maximum(h.A, 0) for h in h_arr])
    wealth_tot = np.sum(wealth_arr)
    weights = wealth_arr / wealth_tot

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
            
# H = 200, F = 20, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33,
#                  m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25

def calibrate_model(a = 100, H = 200, F = 20, Ah = 1, u_r = 0.08, mu_r = 0.3, W_r = 1, gamma_nr = 0.33, m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.1):

    # get elasticity parameter
    rho = (sigma-1)/sigma

    if mu_r < 0:
        print("\nSorry, but mu_r has to be between 0 and 1.")
        exit()

    mu_nr = a - mu_r

    # 1. get Omega, mu_nr, W_nr
    Omega = (1-gamma_nr)/gamma_nr
    X_1 = mu_nr/mu_r
    W_nr = (X_1**rho)*(Omega**(1-rho))*W_r

    # 2. get Nr, Nnr
    Nnr = np.round(H*(1-u_r)/(1+Omega))
    Nr = np.round(Omega*Nnr)

    # 3. get y
    Y = (2**(1/rho))*Omega*mu_r*Nnr
    y_f = Y/F

    # 4. get uc, p
    uc = (Nr*W_r + Nnr*W_nr)/Y
    p = (1+m)*uc

    AF = uc * Y
    Af = AF / F

    # 5. get pi, DIV
    Pi = m*(Nr*W_r + Nnr*W_nr)
    pi_f = Pi/F
    DIV = delta*Pi

    div_f = DIV/F
    div_h = DIV/H

    # get I, c, C, alpha_1, AF
    AH = H*Ah
    I = Nr*W_r + Nnr*W_nr + DIV
    alpha_1 = (Y * p - alpha_2 * AH) / I

    c = Y/H # individual steady state consumption

    print("rho: {}, mu_nr: {}, W_nr: {}, Ah: {}, Af: {}, uc: {}, p: {}, y_f: {}".format(rho, mu_nr, W_nr, Ah, Af, uc, p, y_f))
    print("pi_f: {}, div_h: {}, div_f: {}, c: {}, alpha_1: {}, Nr: {}, Nnr: {}".format(pi_f, div_h, div_f, c, alpha_1, Nr, Nnr))

    return mu_nr, W_nr, Af, uc, p, y_f, pi_f, div_h, div_f, c, alpha_1, Nr, Nnr

def delete_from_old_r_job2(h, f_arr):
    f_id = h.employer_id # get id of old employer
    emp_arr = f_arr[f_id].r_employees
    h_i = np.where(emp_arr == h.id)[0][0] # get index in emp_arr
    f_arr[f_id].r_employees = np.delete(emp_arr, h_i)
    f_arr[f_id].n_r_fired -= 1

def delete_from_old_nr_job2(h, f_arr):
    f_id = h.employer_id  # get id of old employer
    emp_arr = f_arr[f_id].nr_employees
    h_i = np.where(emp_arr == h.id)[0][0]  # get index in emp_arr
    f_arr[f_id].nr_employees = np.delete(emp_arr, h_i)
    f_arr[f_id].n_nr_fired -= 1

# expectation function for a generic variable z

def update_wr_wnr_bar(w_bar, mean_w, phi_w):
    w_bar = phi_w*mean_w + (1-phi_w)*w_bar
    return w_bar

def expectation(z, z_e, lambda_exp):
    """
    this function returns the expected value for the
    variable z for the current period.

    :param z: previous period observation
    :param z_e: previous period expectation
    :param lambda_exp: adjustment parameter
    :return: current period observation
    """
    error = z - z_e
    return z_e + lambda_exp*error

# draw 1 with porbability P and 0 with prob. (1-P)

def draw_one(P):
    num = rd.uniform()
    return 0*(num >= P) + 1*(num<P)

# demand function


def get_N_sub(chi, N):
    if chi * N >= 1:
        return int(chi*N)
    elif chi * N > 0:
        return 1
    else:
        return 0


def init_emp_mat(F, H, u_r, h_arr):
    N = np.int32(H*(1-u_r))
    emp_matrix = np.zeros((F, H))
    rand_f_ids = rd.permutation(N)
    rand_h_ids = rd.choice(np.arange(H), N, replace = False)

    for h_id, perm_num in zip(rand_h_ids, rand_f_ids):
        f_id = perm_num % F
        emp_matrix[f_id, h_id] = 1
        h_arr[h_id].u[0] = 0
    return emp_matrix


# test expectation function
z = 4
z_e = 6
lambda_exp = 0.5
expectation(z, z_e, lambda_exp)

#########################################################################################################
###################### TOOLS FOR HOUSEHOLDS #############################################################
#########################################################################################################

def h_send_apps(app_mat, H, F, N_app):
    # on axis 0 are applicant ids and on axis 1 are firm ids
    f_ids = np.arange(F)
    # draw N_app firm ids from f_ids without replacing
    sub_f_ids = lambda : rd.choice(f_ids, N_app, replace=False)

    for i in range(H):
        app_mat[sub_f_ids(), i] = 1

def get_unemployed(h, nr_job_arr, emp_mat, t):
    emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))
    h.u[t] = 1
    nr_job_arr[h.id] = False
    h.w = 0
    h.fired = False
    h.fired_time = 0


def count_unemployed_hs(h_arr, t):
    for h in h_arr:
        unemp = (h.u[t-1] == 1)
        h.u[t] = 1*unemp


def Pr_LM(w_old, w_new, lambda_LM):
    diff = (w_old - w_new)/w_old
    if diff < 0:
        return 1 - np.exp(lambda_LM*diff)
    else:
        return 0


def update_exp(h_arr, t, diff):

    if t < diff:
        for h in h_arr:
            h.exp = np.sum(1 - h.u[0: t + 1])
    else:
        for h in h_arr:
            h.exp = np.sum(1 - h.u[t - 3: t + 1])


def update_w(h_arr, emp_mat, min_w):
    emp_m = np.sum(emp_mat, axis=0) > 0
    for h in h_arr[emp_m]:
        h.w = np.maximum(h.w, min_w)


def update_d_w(h_arr, emp_mat, sigma_chi, min_w, t):
    emp_arr = np.sum(emp_mat, axis=0)
    for h in h_arr:

        if emp_arr[h.id]==0 or h.job_offer[t-1]==0:
            h.d_w = h.d_w*(1-rd.chisquare(1)*sigma_chi)
        else:
            h.d_w = h.d_w * (1 + rd.chisquare(1) * sigma_chi)
        h.d_w = np.max([h.d_w, min_w, 0.1])


def update_w_e(h_arr, lambda_exp):
    for h in h_arr:
        h.w_e = expectation(h.w, h.w_e, lambda_exp)


def update_Ah(h_arr):
    for h in h_arr:
        h.A += (h.par * h.w + h.div)


def update_d_c(h_arr, alpha_1, alpha_2):
    for h in h_arr:
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
        


##################################################################################
########### functions for the hiring, firing and application mechanism ###########
##################################################################################
################################### ROUTINE ######################################
##################################################################################



################################ application #####################################
##################################################################################

def firms_sort_r_applications(f_arr):
    for f in f_arr:
        f.apps_r = f.apps_r.reshape(int(len(f.apps_r) / 2), 2)
        # firms sort for desired wages
        wages = f.apps_r[:,1]
        sorted_ids = np.argsort(wages)
        f.apps_r = f.apps_r[sorted_ids]


def hs_send_r_apps(f_arr, h_arr, chi, H, beta):
    """
    All households send applications to firms that want to hire
    workers for routine jobs. Both, routine and non-routine type
    households can apply for this kind of jobs.
    :param f_arr: List of all firm objects
    :param h_arr: List of all households
    :param chi: Size of observed subset of firms (between 0 and 1)
    :param H: Number of households
    :param ids: Ids of firms that want to hire routine workers
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




################################## firing ####################################
##############################################################################


def get_r_fired_ids(f, emp_ids):
    return f.r_employees[emp_ids].astype(int)


def f_fires_r_workers(h_arr, fired_ids, f):
    for h in h_arr[fired_ids]:
        h.fired = True
        f.n_r_fired += 1


def firms_fire_r_workers(v_mat, h_arr, f_arr, emp_mat, nr_job_arr):

    f_mask = v_mat[:, 1] < 0
    val = np.sum(f_mask)
    h_inds = np.arange(len(h_arr))
    if val > 0:
        fire_arr = v_mat[f_mask]
        ids = fire_arr[:, 0]
        n_fire_arr = fire_arr[:, 1] * (-1)
        for i in range(len(ids)):
            # get id of the firm, and number of workers it wants to fire
            f_id, n = int(ids[i]), int(n_fire_arr[i])
            emp_mask = emp_mat[f_id, :] > 0
            # get employees as object
            emp_ids = h_inds[np.logical_and(emp_mask, np.invert(nr_job_arr))]
            # look at wages of the employee
            wages = np.array([h.w for h in h_arr[emp_ids]])
            # take indices of employees with highest wages
            mask = np.argsort(wages)[-n:] # indices in emp array
            fired_ids = emp_ids[mask]
            f_fires_r_workers(h_arr, fired_ids, f_arr[f_id])




################################## hiring ####################################
##############################################################################


def remove_r_apps_from_queues(f_arr, chosen_apps):
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

##################################################################################
########### functions for the hiring, firing and application mechanism ###########
##################################################################################
################################# NON-ROUTINE ####################################
##################################################################################



################################# application#####################################
##################################################################################

def firms_sort_nr_applications(f_arr):
    for f in f_arr:
        f.apps_nr = f.apps_nr.reshape(int(len(f.apps_nr) / 2), 2)
        # firms sort for desired wages
        wages = f.apps_nr[:,1]
        sorted_ids = np.argsort(wages)
        f.apps_nr = f.apps_nr[sorted_ids]


def hs_send_nr_apps(f_arr, nr_h_arr, chi, H_nr, H_r, beta):
    """
    Non-routine households send applications to firms that want to hire
    workers for non-routine jobs. Only non-routine type households can
    apply for this kind of jobs.
    :param f_arr: List of all firm objects
    :param nr_h_arr: List of non-routine households
    :param chi: Size of observed subset of firms (between 0 and 1)
    :param H_nr: Number of non-routine households
    :param ids: Ids of firms that want to hire non routine workers
    """
    h_indices = np.array([h.id for h in nr_h_arr])
    f_ids = np.arange(len(f_arr))
    h_arr_shuffled = nr_h_arr[rd.choice(h_indices, H_nr, replace=False) - H_r]
    subN = get_N_sub(chi, len(f_ids))
    for h in h_arr_shuffled:
        # write job application
        application = (h.id, (beta**h.exp)*h.d_w)
        # oberve  random subset of firms
        f_arr_shuffled = f_arr[rd.choice(f_ids, subN, replace=False)]
        # send application to observed firms
        for f in f_arr_shuffled:
            f.apps_nr = np.insert(f.apps_nr, 0, application)
    firms_sort_nr_applications(f_arr[f_ids])




################################## firing ####################################
##############################################################################


def get_nr_fired_ids(f, emp_ids):
    return f.nr_employees[emp_ids].astype(int)


def f_fires_nr_workers(h_arr, fired_ids, f):
    for h in h_arr[fired_ids]:
        h.fired = True
        f.n_nr_fired += 1


def firms_fire_nr_workers(v_mat, h_arr, f_arr, emp_mat, nr_job_arr):

    f_mask = v_mat[:, 2] < 0
    val = np.sum(f_mask)
    h_inds = np.arange(len(h_arr))
    if val > 0:
        fire_arr = v_mat[f_mask]
        ids = fire_arr[:, 0]
        n_fire_arr = fire_arr[:, 2] * (-1)
        for i in range(len(ids)):
            # get id of the firm, and number of workers it wants to fire
            f_id, n = int(ids[i]), int(n_fire_arr[i])
            emp_mask = emp_mat[f_id, :] > 0
            # get employees as object
            emp_ids = h_inds[np.logical_and(emp_mask, nr_job_arr)]
            # look at wages of the employee
            wages = np.array([h.w for h in h_arr[emp_ids]])
            # take indices of employees with highest wages
            mask = np.argsort(wages)[-n:] # indices in emp array
            fired_ids = emp_ids[mask]
            f_fires_nr_workers(h_arr, fired_ids, f_arr[f_id])




################################## hiring ####################################
##############################################################################


def remove_nr_apps_from_queues(f_arr, chosen_apps, emp_ids):
    """
    Removes non-routine job applicants that were chosen by a firm for employment
    from other firms' application queues. Note that in this case we also remove from r queues.
    :param f_arr: List of firms that want to hire
    :param chosen_apps: List of household ids chosen for employment
    """
    for f in f_arr:

        if len(f.apps_nr) > 0:
            nr_f_h_app_ids = f.apps_nr[:, 0].astype(int)
            bool_arr = np.array([not app_id in chosen_apps for app_id in nr_f_h_app_ids])
            # Keep application that are not in "chosen_apps"
            f.apps_nr = f.apps_nr[bool_arr]

        # delete non-routine workers only from routine application queues
        # if they got a routine job first -> use "not app_id in emp_ids"
        if len(f.apps_r) > 0:
            r_f_h_app_ids = f.apps_r[:, 0].astype(int)
            bool_arr = np.array([not app_id in emp_ids for app_id in r_f_h_app_ids])
            f.apps_r = f.apps_r[bool_arr]


def employ_nr_apps(h_arr, emp_mat, app_mat, nr_job_arr, f, lambda_LM, min_w, t):
    for h in h_arr:
        h.job_offer[t] = 1
        # delete all applications
        app_mat[:, h.id] = np.zeros(len(app_mat[:, h.id]))
        if np.sum(emp_mat[:, h.id]) > 0:
            Pr = Pr_LM(h.w, h.d_w, lambda_LM)
            switch = bool(draw_one(Pr))
            if switch:
                f.v_nr -= 1
                # delete from old nr job
                emp_mat[:, h.id] = np.zeros(len(emp_mat[:, h.id]))
                # household gets employed
                emp_mat[f.id, h.id] = 1
                h.w = np.maximum(h.d_w, min_w)
                # update_desired wage in case of a minimum wage
                h.d_w = h.w
                h.fired_time = 0
                h.fired = False
                nr_job_arr[h.id] = True
        else:
            f.v_nr -= 1
            h.u[t] = 0
            # household gets employed
            emp_mat[f.id, h.id] = 1
            h.w = np.maximum(h.d_w, min_w)
            # update_desired wage in case of a minimum wage
            h.d_w = h.w
            nr_job_arr[h.id] = True


def delete_from_old_nr_job(h, emp_mat):
    emp_mat[:, h.id] = np.zeros(len(emp_mat[:, 0]))


def firms_employ_nr_applicants(m):
    f_arr, h_arr, lambda_LM, min_w, t = m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t
    emp_matrix, routine_arr, nr_job_arr = m.emp_matrix, m.routine_arr, m.nr_job_arr
    app_matrix = m.app_matrix
    # 1. get vacancies
    # v_arr = np.array([f.v for f in f_arr])

    # 2. get ids of demand side
    f_ids = np.arange(len(f_arr))

    # 3. shuffle ids
    rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)

    # 4. shuffle vacancies using shuffled ids
    rand_v_arr = np.array([f.v_nr for f in f_arr[rand_f_ids]])

    # get demanded wages
    d_wages = np.asarray([h.d_w for h in m.h_arr])
    h_ids = np.arange(len(h_arr))

    val = True
    while val:
        # print("in 'firms_employ_nr_applicants'")
        v_arr = np.array([f.v_nr if f.v_nr > 0 else 0 for f in f_arr])
        val_arr = v_arr @ app_matrix[:, m.non_routine_arr]
        # some_arr = np.array(
        #     [(f.id, f.v_nr, f.n_nr_fired, f.Nnr, f.d_Nnr) for f in m.f_arr]
        # )
        # print("val_arr: {}".format(some_arr))
        val = np.sum(v_arr @ app_matrix[:, m.non_routine_arr])

        for i in range(len(f_ids)):

            id, v = int(rand_f_ids[i]), int(rand_v_arr[i])
            if np.sum(app_matrix[id, m.non_routine_arr]) * (v > 0) > 0:
                # get app ids
                applied = app_matrix[id, :] > 0 # look at all applicants
                # take ids of workers that have applied and are non-routine type
                mask = np.logical_and(applied, m.non_routine_arr)
                h_app_ids = h_ids[mask]
                # sort app ids from lowest to highest wrt to wages
                sorted_app_ids = np.argsort(d_wages[h_app_ids])
                # choose cheapest ids
                sorted_h_ids = h_app_ids[sorted_app_ids] # sort applicant ids
                chosen_apps = sorted_h_ids[0:v]
                # set only column app_mat[:, h_id] to zero if worker h_id rejects
                employ_nr_apps(h_arr[chosen_apps], emp_matrix, app_matrix,
                               nr_job_arr, f_arr[id], lambda_LM, min_w, t)

        update_N(f_arr, emp_matrix, nr_job_arr)
        update_v(f_arr)
        rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)
        rand_v_arr = np.array([f.v_nr for f in f_arr[rand_f_ids]])
        
        
################################## GOOD MARKET ###############################
##############################################################################
def gm_matching(f_arr, h_arr, chi, tol):
    h_indices = np.array([h.id for h in h_arr])
    f_indices = np.array([f.id for f in f_arr])

    demand = np.array([h.d_c for h in h_arr])

    supply = np.array([(f.y + f.inv)*(not f.default) for f in f_arr])

    while np.sum(demand > 0) and np.sum(supply > 0):
        # print("supply: {}".format(np.sum(supply)))
        # print("demand: {}".format(np.sum(demand)))
        # print("help")
        # print(np.sum(demand > 0), np.sum(supply > 0))

        rand_inds = rd.choice(h_indices, len(h_arr), replace=False).astype(int)
        h_arr_shuffled = h_arr[rand_inds]
        d_m = demand[rand_inds] > 0

        for h in h_arr_shuffled[d_m]:
            # observe firms
            subN = get_N_sub(chi, np.sum(supply>0))
            if subN > 0:
                f_arr_shuffled = f_arr[rd.choice(f_indices[supply>0], subN, replace=False)]
                # take the one with lowest price
                prices = np.array([f.p for f in f_arr_shuffled])
                ind = np.argsort(prices)[0]
                f = f_arr_shuffled[ind]
                # buy consumption good
                c = np.min([h.d_c/1, demand[int(h.id)], supply[int(f.id)]])
                # print(current_rw, demand[h.id], supply[f.id])
                expenditure = c*f.p
                if expenditure <= h.A:
                    h.expenditure += expenditure
                    h.A -= expenditure
                    h.c += c
                    f.s += c
                    demand[h.id] -= c
                    supply[f.id] -= c
                else:
                    demand[h.id] = 0

                if (np.abs(demand[h.id]) < tol) or (np.abs(h.A) < tol):
                    demand[h.id] = 0
                if np.abs(supply[f.id]) < tol:
                    supply[f.id] = 0
                    

################################## STEP FUNCTION ###############################
##############################################################################
def wage_decisions(m):

    # wage decisions
    update_w(m.h_arr, m.emp_matrix, m.min_w)
    set_W_fs(m.f_arr[m.active_fs], m.emp_matrix, m.nr_job_arr, m.h_arr)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr[m.active_fs], m.emp_matrix, m.nr_job_arr, m.h_arr)  # firms measure average wages paid to employees
    update_Wr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)  # firms build wage expectations
    update_Wnr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)
    update_d_w(m.h_arr, m.emp_matrix, m.sigma_w, m.min_w, m.t)  # households decide for desired wages


def household_decisions(m):

    # households update work experience
    update_exp(m.h_arr, m.t, 4)

    # households make consumption decision
    update_p_e(m.h_arr, m.lambda_exp)
    update_d_c(m.h_arr, m.alpha_1, m.alpha_2)


def firm_decisions(m):

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

    # firms calculate profits
    update_pi(m.f_arr)
    update_div_f(m.f_arr[m.active_fs])

    # surviving firms pay dividends
    distribute_dividends(m.h_arr, m.f_arr)


def hh_refin_firms(m):

    # households refinance firms
    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)

    # change this later
    refin_firms(m.f_arr[m.default_fs], m.f_arr[m.active_fs], m.h_arr, m.n_refinanced,
                m.tol, m.t)

    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    


################################## LABOR MODEL ###############################
##############################################################################
class Model:

    def __init__(self,
                 # exogenously chosen steady state parameters
                 H = 200, F = 20, Ah = 1, u_r = 0.08, mu_r = 0.3, W_r = 1, gamma_nr = 0.33,
                 m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25,
                 # exogenous model parameters
                 lambda_LM = 0.5, lambda_exp = 0.25, beta = 1, nu = 0.1, sigma_m = 0.001, sigma_w = 0.005,
                 sigma_delta = 0.001, chi_C = 0.2, T = 500,
                 tol = 1e-10, N_app = 4, nr_to_r = False, a = 100, min_w_par = 0.3 , f_max = 3):


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

        if self.h_arr[66].d_w==0:
            print(self.h_arr[66].d_w)
        self.t += 1

    def run(self):

        # initialize employment
        set_W_fs(self.f_arr, self.emp_matrix, self.nr_job_arr, self.h_arr)

        for t in range(self.T):
            self.step_function()
            
            
##################################  PLOT TOOL  ###############################
##############################################################################         
def plot_lm(m, T, periods, steps):

    t = T - periods

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

    time_array = np.arange(0, T)

    axs[0, 0].clear()
    axs[1, 0].clear()
    axs[0, 1].clear()
    axs[1, 1].clear()

    fontsize = 15

    # GDP plot
    axs[0, 0].grid()
    axs[0, 0].set_title("GDP", fontsize=fontsize)
    axs[0, 0].plot(time_array[t:T:steps], m.GDP[t:T:steps], marker="o", markersize=3, alpha=1, label="GDP")

    # Unemployment plot
    axs[1, 0].grid()
    axs[1, 0].set_title("Unemployment rate", fontsize=fontsize)
    axs[1, 0].plot(time_array[t:T:steps], m.u_r_arr[t:T:steps], marker="o", markersize=3, alpha=0.5, label="Unemployment")

    # mean wages plot
    axs[0, 1].grid()
    axs[0, 1].set_title("Mean wages", fontsize=fontsize)
    axs[0, 1].plot(time_array[t:T:steps], m.mean_r_w_arr[t:T:steps], marker="o", markersize=2, alpha=1, label="Mean wages routine")
    axs[0, 1].plot(time_array[t:T:steps], m.mean_nr_w_arr[t:T:steps], marker="o", markersize=2, alpha=1,
                   label="Mean wages non-routine")
    axs[0, 1].legend(loc="best")

    # Beveridge curve
    axs[1, 1].grid()
    axs[1, 1].set_title("Beveridge curve", fontsize=fontsize)
    axs[1, 1].scatter(m.u_r_arr[t:T:steps], m.open_vs[t:T:steps], alpha=0.5)
    # axs[1, 1].set(xlim = (0, 0.25), ylim = (0, 80))


    # Log-differences mean real wages
    axs2[0, 0].grid()
    axs2[0, 0].set_title("Variance of Wages", fontsize=fontsize)
    log_diff_arr = np.log(m.mean_nr_w_arr[t:T:steps])-np.log(m.mean_r_w_arr[t:T:steps])
    axs2[0, 0].plot(time_array[t:T:steps], log_diff_arr, marker="o", markersize=2, alpha=1,
                   label="log-diff mean real wages", color = "k")

    # nr and r unemployment rates
    axs2[0, 1].grid()
    axs2[0, 1].set_title("Decile Comparison", fontsize=fontsize)
    axs2[0, 1].plot(time_array[t:T:steps], m.nine_to_five[t:T:steps], label="9/5")
    axs2[0, 1].plot(time_array[t:T:steps], m.five_to_one[t:T:steps], label="5/1")
    axs2[0, 1].plot(time_array[t:T:steps], m.nine_to_one[t:T:steps], label="9/1")
    axs2[0, 1].legend(loc="best")
    axs2[0, 1].set_ylabel("(decile) ratio of income")

    axs2[1, 0].grid()
    axs2[1, 0].set_title("Mean prices", fontsize=fontsize)
    axs2[1, 0].plot(time_array[t:T:steps], m.mean_p_arr[t:T:steps], marker="o",
                    markersize=2, alpha=1, color = "green")
    axs2[0, 1].set_ylabel("mean price")

    axs2[1, 1].grid()
    axs2[1, 1].set_title("default rate", fontsize=fontsize)
    axs2[1, 1].set_ylabel("share of refin. firms", color = "c")
    ax3 = axs2[1, 1].twinx()
    ax3.set_ylabel("share of inactive firms", color = "orange")
    # axs2[1, 1].plot(time_array[t:T], m.n_refinanced[t:T]/m.F, alpha=1, label = "share of refinanced firms")
    width = 40
    axs2[1, 1].bar(time_array[t:T:steps], m.n_refinanced[t:T:steps]/m.F, label = "share of refinanced firms", color = "c")
    ax3.bar(time_array[t:T:steps], m.share_inactive[t:T:steps], 10, label="share of inactive firms", color="orange", alpha = 0.3)
    # ax3.plot(time_array[t:T], m.share_inactive[t:T], alpha=1, label = "share of inactive firms", color = "orange")


    return fig, fig2

def get_wage_dist_fig(m):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.grid()
    ax2.grid()
    ax3.grid()

    fontsize = 15
    # look only at employed wages
    wages = np.array([h.w for h in m.h_arr])
    wages = wages[wages > 0]

    # wages of routine worker
    r_wages = np.array([h.w for h in m.h_arr[m.routine_arr]])
    r_wages = r_wages[r_wages > 0]

    # wages of non-routine worker
    nr_wages = np.array([h.w for h in m.h_arr[m.non_routine_arr]])
    nr_wages = nr_wages[nr_wages > 0]

    # norm_wages = (1/np.max(wages))*wages
    sns.histplot(wages, kde=False, ax = ax1)
    sns.histplot(r_wages, kde=False, ax = ax2)
    sns.histplot(nr_wages, kde=False, ax = ax3)

    # titles
    ax1.set_title("Distribution of the wages of all workers", fontsize=fontsize)
    ax2.set_title("Distribution of the wages of routine workers", fontsize=fontsize)
    ax3.set_title("Distribution of the wages of non-routine workers", fontsize=fontsize)

    return fig

def get_aggregate_regs(m, T, periods):

    t = T - periods

    GDP_growth = [m.GDP[i] / m.GDP[i-1] - 1 if i > 0 else 0 for i in range(m.T)]
    GDP_growth = GDP_growth[1:]

    unemp_growth = [np.exp(m.u_r_arr[i]-m.u_r_arr[i-1]) - 1 if i > 0 else 0 for i in range(m.T)]
    unemp_growth = unemp_growth[1:]

    wage_rate = [m.mean_nominal_w_arr[i]/m.mean_nominal_w_arr[i-1] - 1
                 if i > 0 else 0 for i in range(m.T)]

    wage_level = [m.mean_nominal_w_arr[i]/m.mean_p_arr[i] for i in range(m.T)]
    # wage_rate = wage_rate[1:]


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.grid()
    ax2.grid()
    ax3.grid()

    fontsize = 15

    # Beveridge curve
    ax1.set_title("Beveridge curve", fontsize=fontsize)
    ax1.scatter(m.u_r_arr[t:T], m.open_vs[t:T], alpha=0.5, color = "red")
    ax1.set_ylabel("open vacancies")
    ax1.set_ylabel("unemployment rate")

    # Wage curve
    ax2.set_title("Wage curve", fontsize=fontsize)
    ax2.scatter(m.u_r_arr[t:T], wage_level[t:T], alpha=0.5, color = "orange")
    ax2.set_xlabel("unemployment rate")
    ax2.set_ylabel("wage level")

    # Okun curve
    ax3.set_title("Okun curve", fontsize=fontsize)
    ax3.scatter(unemp_growth[t:T], GDP_growth[t:T], alpha=0.5, color = "green")
    ax3.set_ylabel("(real) GDP growth")
    ax3.set_xlabel("unemployment growth")

    return fig