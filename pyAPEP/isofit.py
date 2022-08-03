"""
====================================
Isotherm fitting module (:py:mod:`pyAPEP.isofit`)
====================================
.. moduleauthor:: jjh <hyo156@kitech.re.kr>
.. note:: 따라하세요

Theory
======
Theory Theory Theory Theory Theory Theory 
Theory Theory Theory Theory Theory Theory 

Tutorials
=========
Tutorials Tutorials Tutorials Tutorials
Tutorials Tutorials Tutorials Tutorials

Functions
=========
This module contains objects to characterize the pure-component adsorption
isotherms from experimental or simulated data. These will be fed into the
IAST functions in pyiast.py.

 * [2022/08/02] - test
"""


# %% Import python packages required
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
# %% Define a class: IdealColumn

def check_isotherm(n_comp, isotherm_fun,):
    P_test = []
    for ii in range(10):
        P_test.append(10*np.random.rand(n_comp))
    T_test = [300, 320, 340]
    q_test_res = []
    for T_t in T_test:
        q_4_P_t = []
        for P_t in P_test:
            try:
                q_t_tmp = isotherm_fun(P_t, T_t)
            except:
                print('ERROR: the input should be in the form of "isotherm_fun(P, T)" ')
                return False
            q_4_P_t.append(q_t_tmp)
        q_test_res.append(q_4_P_t)
    return True

class IdealColumn:
    """
    Class to characterize pure-component isotherm data with an analytical model.
    Data fitting is done during instantiation.
    
    Models supported are as follows. Here, :math:`L` is the gas uptake,
    :math:`P` is pressure (fugacity technically).
    
    * Langmuir isotherm model
    
    .. math::
    
        L(P) = M \\frac{KP}{1+KP},
        
    * Quadratic isotherm model
    
    .. math::
    
        L(P) = M \\frac{(K_a + 2 K_b P)P}{1+K_aP+K_bP^2}
        
    * Brunauer-Emmett-Teller (BET) adsorption isotherm
    
    .. math::
    
        L(P) = M\\frac{K_A P}{(1-K_B P)(1-K_B P+ K_A P)}
        
    * Dual-site Langmuir (DSLangmuir) adsorption isotherm
    
    .. math::
    
        L(P) = M_1\\frac{K_1 P}{1+K_1 P} +  M_2\\frac{K_2 P}{1+K_2 P}
        
    * Asymptotic approximation to the Temkin Isotherm
    (see DOI: 10.1039/C3CP55039G)
    
    .. math::
    
        L(P) = M\\frac{KP}{1+KP} + M \\theta (\\frac{KP}{1+KP})^2 (\\frac{KP}{1+KP} -1)
        
    * Henry's law. Only use if your data is linear, and do not necessarily trust
      IAST results from Henry's law if the result required an extrapolation
      of your data; Henry's law is unrealistic because the adsorption sites
      will saturate at higher pressures.
      
    .. math::
    
        L(P) = K_H P
        
    """
    def __init__(self, n_comp, isotherm_fun = None,):
        # The isotherm funciton should be a function of both pressure and temperature
        if isotherm_fun == None:
            def iso_example(P_part, T):
                bP1 = P_part[0]*0.3
                bP2 = P_part[1]*0.1
                bP3 = P_part[2]*0.8
                bP_arr = np.array([bP1, bP2, bP3])
                bP_sum = np.sum(bP_arr)
                nume = np.array([3, 4, 1])*bP_arr
                deno = 1 + bP_sum
                q_return = nume / deno
                return q_return
            isotherm_fun = iso_example

        check_res = check_isotherm(n_comp, isotherm_fun)
        if check_res:
            self._isofun = isotherm_fun
            self._n_comp = n_comp
            self._str = {'isotherm' : True,
                        'feedcond': False,
                        'opercond': False,}
            
        """
        Instantiation. A `IdealColumn` class is instantiated by passing it the
        pure-component adsorption isotherm in the form of a Pandas DataFrame.
        The least squares data fitting is done here.
        
        :param n_comp: DataFrame pure-component adsorption isotherm data
        :param isotherm_fun: String key for loading column in df
        
        :rtype: IdealColumn
        """

    def isofunct(self, n_comp, isotherm_fun):
        """
        Given stored model parameters, compute loading at pressure P.
        
        :param pressure: Float or Array pressure (in corresponding units as df
            in instantiation)
        :return: predicted loading at pressure P (in corresponding units as df
            in instantiation) using fitted model params in `self.params`.
        :rtype: Float or Array
        """
        if check_isotherm(n_comp, isotherm_fun):
            self._isofun = isotherm_fun
            self._n_comp = n_comp
        else:
            print("Dim. of function output")
            print("should be equal to n_comp")

    def feedcond(self, P_feed, T_feed, y_feed):
        """
        Input feed condtions
        
        """
        if len(y_feed) != self._n_comp:
            print("Dim. of y_feed (feed composition)" )
            print("should be equal to n_comp")
            return
        
        y_feed_norm = np.array(np.array(y_feed)/ np.sum(y_feed))
        
        print(y_feed_norm)
        for ii in range(len(y_feed_norm)):
            if y_feed_norm[ii] < 0.0001:
                y_feed_norm[ii]  = 0.0001
            elif y_feed_norm[ii] > 0.9999:
                y_feed_norm[ii] = 0.9999
        
        y_feed_norm = np.array(np.array(y_feed_norm)/ np.sum(y_feed_norm))

        self._P_feed = P_feed
        self._T_feed = T_feed
        self._y_feed = y_feed_norm
        self._str['feedcond'] = True


    def opercond(self, P_high, P_low):
        """
        Input operation condtions
        
        """
        self._P_high = P_high
        self._P_low = P_low
        self._str['opercond'] = True

    
    def runideal(self, tol = 1E-4):
        """
        Run the ideal PSA simulation.
        
        """
        isomix = self._isofun
        y_feed = self._y_feed
        P_feed = self._P_feed
        T_feed = self._T_feed
        P_high = self._P_high
        P_low = self._P_low
        P_part_feed = P_high*np.array(y_feed)
        n_comp = self._n_comp
        q_sat = isomix(P_part_feed, T_feed)
        def x2x(x_init):
            P_part_des = P_low*np.array(x_init)
            q_des = isomix(P_part_des, T_feed)
            dq = q_sat - q_des
            dq[dq<0] = 0
            dq_tot = np.sum(dq)+ 1E-9
            x_new = dq/dq_tot
            return x_new
        x_guess_all = x2x(y_feed)
        x_guess = x_guess_all[:-1]
        print('x_guess is:')
        print(x_guess)
        def err_x2x(x_gu):
            Penalty = 0
            for xx, ii in zip(x_gu, range(len(x_gu))):
                if xx < 0:
                    Penalty = Penalty + 1000*(xx - 0)**2
                    x_gu[ii] = 1E-6
                elif xx > 1:
                    Penalty = Penalty + 1000*(xx - 1)**2
                    x_gu[ii] = 1 - 1E-6
            x_last = 1 - np.sum(x_gu)
            if x_last < 0:
                Penalty = Penalty + 1000*(x_last - 0)**2
                x_last = 1E-6
            elif x_last > 1:
                Penalty = Penalty + 1000*(x_last - 1)**2
                x_last = 1 - 1E-6
            x_all = np.append(np.array(x_gu), x_last)
            print(x_all)
            x_recal = x2x(x_all)
            err_x = np.sum((x_all - x_recal)**2) + Penalty
            print(Penalty)
            return err_x

        optres = minimize(err_x2x, x_guess, method = 'Nelder-mead')
        x_guess = optres.x
        if optres.fun > tol:
            optres_prev = optres
            optres = minimize(err_x2x, x_guess, method='L-BFGS-B')
            if optres.fun > optres_prev.fun:
                optres = optres_prev
        if optres.fun > tol:
            optres_prev = optres
            bounds = np.ones([n_comp-1, 2])
            bounds[:,0] = 0
            optres = differential_evolution(err_x2x, bounds)
            if optres.fun > optres_prev.fun:
                optres = optres_prev

        x_sol = optres.x

        print('Funciton value of optimizaiton is ')
        print(optres.fun)
        x_last_sol = 1- np.sum(x_sol)
        x_purity = np.append(x_sol, x_last_sol)
        self.x_ideal = x_purity

        return x_purity


    def __str__(self):
        str_out = ''
        for kk in self._str.keys():
            str_out = str_out + kk
            if self._str[kk]:
                str_out = str_out + ': True\n'
            else:
                str_out = str_out + ': False\n'
        return str_out        

