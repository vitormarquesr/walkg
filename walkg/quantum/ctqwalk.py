""" Continuous-Time Quantum Walk """

from walkg.base import Spectral
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
        M = self.get_eigschur(1)

        for i in range(2, len(self.evals) + 1):
            M += self.get_eigschur(i)

        return M

    def gavg_matrix(self, dist):
        pass

    def plot(self, i, j, t_init=0, t_final=None):

        # Set reasonable default t_final
        if t_final is None:
            pass

        grid = np.arange(t_init, t_final, 0.1)

        density = map(lambda t: self.mixing_matrix(t)[i-1, j-1], grid)

        plt.plot(list(grid), list(density))
        plt.ylim(0, 1)
        plt.show()
