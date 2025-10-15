"""
Agent-Based Model Classes for Labor Market Simulation

This module defines the core agent classes for the labor market ABM:
- Agent: Base class for all agents in the model
- Firm: Represents firms that employ workers and produce goods
- Household: Represents workers/consumers in the labor market

Author: Labor Market ABM Project
"""

import numpy as np
import numpy.random as rd


class Agent(object):
    """
    Base agent class for the labor market model.
    
    This class provides common attributes and methods for all agents
    in the model, including firms and households.
    """

    def __init__(self, _id, A_init, T):
        """
        Initialize a generic agent.
        
        Parameters
        ----------
        _id : int
            Unique identifier for the agent
        A_init : float
            Initial wealth/assets of the agent
        T : int
            Total number of simulation periods
        """

        # Unique identifier
        self.id = _id
        
        # Time horizon
        self.T = T
        
        # Assets/wealth
        self.A = A_init
        
        # Cash holdings
        self.cash = A_init

        # Taxes that agent has to pay
        self.income_tax = 0
        self.wealth_tax = 0

        # Tolerance level for setting values to zero (numerical precision)
        self.tol = 10**(-10)

        # Percentage of wages paid/received (used in default scenarios)
        self.par = 1

    def change_cash(self, c):
        """
        Update agent's cash holdings.
        
        Parameters
        ----------
        c : float
            Change in cash (positive for increase, negative for decrease)
        """
        self.cash += c
        # Set to zero if below tolerance (avoid numerical errors)
        if np.abs(self.cash) < self.tol:
            self.cash = 0


class Firm(Agent):
    """
    Firm agent in the labor market model.
    
    Firms employ workers (routine and non-routine), produce goods,
    set prices and wages, and can potentially default if unprofitable.
    
    Attributes
    ----------
    r_employees : np.ndarray
        Array of routine employee IDs
    nr_employees : np.ndarray
        Array of non-routine employee IDs
    Nr : int
        Number of routine employees
    Nnr : int
        Number of non-routine employees
    d_Nr : int
        Desired number of routine employees
    d_Nnr : int
        Desired number of non-routine employees
    Wr : float
        Average wage paid to routine employees
    Wnr : float
        Average wage paid to non-routine employees
    v_r : int
        Number of routine vacancies
    v_nr : int
        Number of non-routine vacancies
    y : float
        Current production
    d_y : float
        Desired production
    p : float
        Price of goods
    m : float
        Markup over unit cost
    default : bool
        Whether firm has defaulted
    """

    def __init__(self, _id, A_init, T, y, nu, Wr_f, Wnr_f, delta, p, m, pi, div, uc):
        """
        Initialize a firm agent.
        
        Parameters
        ----------
        _id : int
            Unique firm identifier
        A_init : float
            Initial assets
        T : int
            Total simulation periods
        y : float
            Initial production level
        nu : float
            Inventory share parameter
        Wr_f : float
            Initial routine wage
        Wnr_f : float
            Initial non-routine wage
        delta : float
            Dividend payout rate
        p : float
            Initial price
        m : float
            Initial markup
        pi : float
            Initial profit
        div : float
            Initial dividends
        uc : float
            Initial unit cost
        """
        super(Firm, self).__init__(_id, A_init, T)

        # Employee arrays: routine and non-routine workers
        self.r_employees, self.nr_employees = np.array([]), np.array([])

        # Sales: actual and expected
        self.s, self.s_e = y, y

        # Number of employees: actual and desired
        self.Nr, self.d_Nr = 0, 0  # Routine workers
        self.Nnr, self.d_Nnr = 0, 0  # Non-routine workers
        self.n_r_fired, self.n_nr_fired = 0, 0  # Fired workers count

        # Average wages: actual and expected
        self.Wr, self.Wnr = Wr_f, Wnr_f  # Current wages
        self.Wr_e, self.Wnr_e = Wr_f, Wnr_f  # Expected wages

        # Total wage bill for each worker type
        self.Wr_tot, self.Wnr_tot = 0, 0

        # Job applicants: routine and non-routine
        self.apps_r, self.apps_nr = np.array([]), np.array([])

        # Production and pricing
        self.d_y, self.y, self.p = y, y, p  # Desired, actual output, and price
        self.d_y_diff = 0  # Difference in desired production
        
        # Unit cost tracking over time
        self.uc_arr = np.zeros(T)
        self.uc_arr[-1] = uc

        # Vacancies for each worker type
        self.v_r, self.v_nr = 0, 0

        # Inventory management
        self.nu, self.m = nu, m  # Inventory share and markup
        self.inv = nu*y  # Current inventory

        # Financial outcomes
        self.pi, self.div, self.delta = pi, div, delta  # Profit, dividends, dividend rate

        # Default status and wage payment parameter
        self.default, self.par = False, 1


class Household(Agent):
    """
    Household agent in the labor market model.
    
    Households supply labor (routine or non-routine), consume goods,
    earn wages and dividends, and can be employed or unemployed.
    
    Attributes
    ----------
    routine : bool
        Whether household is routine (True) or non-routine (False) type
    nr_job : bool
        Whether current job is non-routine (True) or routine (False)
    u : np.ndarray
        Unemployment status over time (1 = unemployed, 0 = employed)
    w : float
        Current wage
    d_w : float
        Desired wage
    exp : int
        Work experience accumulated
    employer_id : int or None
        ID of current employer (None if unemployed)
    c : float
        Actual consumption
    d_c : float
        Desired consumption
    fired : bool
        Whether household was fired recently
    """

    def __init__(self, _id, A_init, T, routine, w, p, c, div, f_max):
        """
        Initialize a household agent.
        
        Parameters
        ----------
        _id : int
            Unique household identifier
        A_init : float
            Initial assets/wealth
        T : int
            Total simulation periods
        routine : bool
            Whether household has routine skills (True) or non-routine (False)
        w : float
            Initial wage
        p : float
            Initial price level
        c : float
            Initial consumption
        div : float
            Initial dividend income
        f_max : int
            Maximum time since being fired
        """
        super(Household, self).__init__(_id, A_init, T)

        # Skill type: True for routine, False for non-routine
        self.routine = routine
        
        # Public sector worker status
        self.public_worker = False

        # Current job type: True if non-routine job, False if routine job
        self.nr_job = False

        # Job offer history (1 = received offer, 0 = no offer at time t)
        self.job_offer = np.zeros(T)

        # Fired status and timing
        self.fired = False  # Whether recently fired
        self.fired_time = 0  # Time since being fired
        self.fired_time_max = f_max  # Maximum tracked time since firing

        # Unemployment status over time (1 = unemployed, 0 = employed)
        self.u = np.zeros(T)
        self.u[0] = 1  # Start unemployed
        
        # Work experience and employer
        self.exp, self.employer_id = 0, None

        # Wages: desired, actual, expected, and last period
        self.d_w, self.w = w, w  # Desired and actual wage
        self.w_e = w  # Expected wage
        self.last_w = w  # Previous period wage

        # Prices: actual and expected
        self.p, self.p_e = p, p

        # Consumption: desired and actual
        self.d_c, self.c = c, c

        # Consumption expenditure in current period
        self.expenditure = c*p

        # Dividend income: actual and expected
        self.div = div
        self.div_e = div

        # Refinancing contribution (for firm bailouts)
        self.refin = 0
