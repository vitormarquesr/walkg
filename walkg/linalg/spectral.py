import numpy as np
from scipy.linalg import ishermitian

class Spectral:
    def __init__(self, matrix, _h=None, _mult=True, _tol=1e-8):
        self.matrix = np.array(matrix)
        self._h = _h if _h is not None else ishermitian(self.matrix)
        self.evals, self.elabs, self.evcts  = self._decomp(self.matrix, self._h, _mult, _tol)
    @staticmethod
    def _is_equal(x, y, tol=1e-8):
        return abs(x - y) < tol
    @staticmethod
    def _label_equal(x, tol=1e-8):
        i = 0
        vals = [x[0]]
        labs = [0]
        for k in range(1, len(x)):
            # Equal adjacent numbers receive the same label
            if not Spectral._is_equal(x[k], x[k-1], tol):
                i += 1
                vals.append(x[k])
            labs.append(i)
        return np.array(vals), np.array(labs)
    @staticmethod
    def _decomp(matrix, h, mult, tol):
        
        # Ensure orthogonal eigenbasis for hermitian matrices
        evals, evcts = np.linalg.eigh(matrix) if h else np.linalg.eig(matrix)
        
        # Ensure spectra is in decreasing order
        sorted_idxs = np.argsort(evals)[::-1]
        evals = evals[sorted_idxs]
        elabs = np.arange(len(evals))
        evcts = evcts[:, sorted_idxs]

        # Label equal eigenvalues
        if mult:
            evals, elabs = Spectral._label_equal(evals, tol)

        return evals, elabs, evcts
