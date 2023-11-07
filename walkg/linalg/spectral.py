import numpy as np
from scipy.linalg import ishermitian

class Spectral:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.evals, self.evcts  = self._decomp(self.matrix)
    def _is_hermitian(self, matrix):
        return ishermitian(matrix)
    def _decomp(self, matrix):
        if self._is_hermitian(matrix):
            s = np.linalg.eigh(self.matrix)
        else:
            s = np.linalg.eig(self.matrix)
        return s
