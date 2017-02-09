import numpy as np
from numpy import exp, sqrt, cos, sin, pi, inf
from numpy.linalg import inv, norm
import scipy.stats as st
from scipy.optimize import least_squares


# Von mises orientation tuning
# Von mises fitting function
class VonMises:
    @staticmethod
    def initial_guess(r, theta):
        # initialize phi=peak orientation, A=peak firing rate, B=min firing rate
        # Need a good initialization for k

        unique = np.unique(theta)
        n_angles = unique.shape[0]

        max = 0.
        min = np.inf
        pref_or = -1.

        for angle in unique:
            temp = np.mean(r[theta == angle])
            if temp >= max:
                max = temp
                pref_or = angle
            if temp <= min:
                min = temp

        return [pref_or, 1, max, min]

    @staticmethod
    def vonmises(theta, params):
        phi, k, A, B = params
        """ Von Mises Distribution, defined on interval [0, pi]"""
        return A * exp(k * (cos(2 * (theta - phi)) - 1)) + B

    @staticmethod
    def residuals(p, theta, r):
        err = r - VonMises.vonmises(theta, p)
        return err

    @staticmethod
    def r_squared(theta, r, p):
        ss_res = (VonMises.residuals(p, theta, r)**2).sum()
        ss_tot = ((r - r.mean())**2).sum()
        return 1 - (ss_res/ss_tot)

    @staticmethod
    def fit(r, theta):
        p0 = VonMises.initial_guess(r, theta)
        lst_sqr = least_squares(VonMises.residuals, p0, bounds=([0., 0., 0, 0], [pi, inf, inf, inf]), args=(theta, r))
        if not lst_sqr['success']:
            print('Algorithm did not Converge')
        #     raise RuntimeError('Least Square Algorithm did not converge')

        return lst_sqr['x']

