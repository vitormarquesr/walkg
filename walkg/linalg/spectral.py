import numpy as np
from scipy.linalg import ishermitian


class Spectral:
    """
    Spectral Analysis of Matrices

    Parameters
    ----------
    matrix : array_like
        Matrix to be analyzed.
    _h : bool, optional
        Whether the matrix is Hermitian. If None, checks
        for hermiticity.
    _mult : bool, optional
        Flag indicating to account for repeated eigenvalues
        (Default value = True).
    _tol : float, optional
        Tolerance for numerical equality (Default value = 1e-8).

    Attributes
    ----------
    matrix: np.array
        matrix being analyzed
    _h: bool
        flag indicating whether the matrix is hermitian
    evals: np.array
        unique eigenvalues of the matrix sorted in decreasing order
    _elabs: np.array
        labels indicating each eigenvalue appearance in the spectrum
    evcts: np.array
        square matrix whose columns are eigenvectors ordered according to
        evals
    emult: np.array
        eigenvalue multiplicity
    """

    def __init__(self, matrix, _h=None, _mult=True, _tol=1e-8):
        self.matrix = np.array(matrix)
        self._h = _h if _h is not None else ishermitian(self.matrix)
        self.evals, self._elabs, self.evcts = self._decomp(
            self.matrix, self._h, _mult, _tol
        )
        self.emult = np.unique(self._elabs, return_counts=True)[1]

    def __str__(self):
        str_spectra = ', '.join([f'{ev:.2f}'.center(5) for ev in self.evals])
        str_emult = ', '.join([str(ml).center(5) for ml in self.emult])
        string = f'''Spectral(
    Hermitian: {self._h}
    Spectra:      {str_spectra}
    Multiplicity: {str_emult}
)'''
        return string

    @staticmethod
    def _is_equal(x, y, tol=1e-8):
        """
        Check whether two numbers are equal up to a margin of error
        """
        return abs(x - y) < tol

    @staticmethod
    def _label_equal(x, tol=1e-8):
        """
        Label equal eigenvalues
        """
        i = 1
        vals = [x[0]]
        labs = [1]
        for k in range(1, len(x)):
            # Equal adjacent numbers receive the same label
            if not Spectral._is_equal(x[k], x[k - 1], tol):
                i += 1
                vals.append(x[k])
            labs.append(i)
        return np.array(vals), np.array(labs)

    @staticmethod
    def _decomp(matrix, h, mult, tol):
        """
        Perform enhanced spectral decomposition
        """

        # Ensure orthogonal eigenbasis for hermitian matrices
        evals, evcts = np.linalg.eigh(matrix) if h else np.linalg.eig(matrix)

        # Ensure spectra is in decreasing order
        sorted_idxs = np.argsort(evals)[::-1]
        evals = evals[sorted_idxs]
        elabs = np.arange(len(evals)) + 1
        evcts = evcts[:, sorted_idxs]

        # Label equal eigenvalues
        if mult:
            evals, elabs = Spectral._label_equal(evals, tol)

        return evals, elabs, evcts

    def get_spectra(self):
        """
        The spectra of the matrix

        Returns
        -------
        self.evals: array_like
            The unique eigenvalues
        self.emult: array_like
            The eigenvalues multiplicities
        """
        spectra = (self.evals, self.emult)
        return spectra

    def get_eigspace(self, id):
        """
        The eigenspace of the matrix

        Parameters
        ----------
        id : int

        Returns
        ----------
        espace: array_like
            rectangular matrix whose columns form the eigenbasis
            of the desired eigensapce.
        """

        if id > self._elabs.max() or id < 1:
            raise IndexError('Id out of range')
        espace = self.evcts[:, self._elabs == id]
        return espace
