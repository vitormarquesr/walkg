""" Continuous-Time Quantum Walk """

from walkg.base import Spectral
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt


class CTQW(Spectral):
    def __init__(self, H, _mult=True, _tol=1e-8):
        Spectral.__init__(self, H, _h=True, _mult=_mult, _tol=_tol)

    def transition_matrix(self, t):
        return self.act_eigfun(lambda x: np.exp(1j * t * x))

    def mixing_matrix(self, t):
        Ut = self.transition_matrix(t)
        Mt = abs(Ut) ** 2
        return Mt

    def avg_matrix(self):
        # Formula approach
        E = map(self.get_eigschur, range(1, len(self.evals) + 1))

        M = reduce(operator.add, E)
        return M

    def gavg_matrix(self, dist):
        # Monte Carlo approach
        M_t = map(self.mixing_matrix, dist)

        M = reduce(operator.add, M_t) / len(dist)
        return M

    def plot_density(self, i, j, t_init=0, t_final=None):
        # Set reasonable default t_final
        if t_final is None:
            # Gap between first and second eigenvalue
            delta = self.evals[0] - self.evals[1]

            # Maximum period among cos(delta_rs * t) terms
            period_max = 2 * np.pi * np.ceil(1 / delta)
            t_final = period_max

        grid = np.arange(t_init, t_final, 0.1)

        density = map(lambda t: self.mixing_matrix(t)[i - 1, j - 1], grid)

        plt.plot([*grid], [*density])
        plt.ylim(0, 1)

        plt.show()
