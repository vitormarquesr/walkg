""" Discrete-Time Markov Chains """
from walkg.base import Spectral
import numpy as np
import matplotlib.pyplot as plt


class DTMarkov(Spectral):
    def __init__(self, P, _mult=True, _tol=1e-8):
        # P column-stochastic matrix
        Spectral.__init__(self, P, _h=False, _mult=_mult, _tol=_tol)

    @classmethod
    def from_adjacency(cls, A, _mult=True, _tol=1e-8):
        D = A.sum(axis=0)
        P = A/D
        return cls(P, _mult, _tol)

    @staticmethod
    def _check_time(t):
        if not isinstance(t, int) or t < 0:
            raise ValueError("Invalid time!")

    def transition_matrix(self, t=1):
        DTMarkov._check_time(t)
        return np.linalg.matrix_power(self.matrix, t)

    def perron_vector(self):
        perron_vct = self.evcts[:, 0] / sum(self.evcts[:, 0])

        return perron_vct

    def plot_density(self, i, j, t_init=0, t_final=None):
        # Set reasonable default t_final
        DTMarkov._check_time(t_init)

        if t_final is None:
            # 10 steps at most
            t_final = t_init + 10

        DTMarkov._check_time(t_final)

        grid = range(t_init, t_final+1, 1)

        density = map(lambda t: self.transition_matrix(t)[j - 1, i - 1], grid)

        plt.plot([*grid], [*density])
        plt.ylim(0, 1)

        plt.show()
