"""
Prototypical, non-parallel implementation of the parallel cyclic reduction (PCR) algorithm from Hockney & Jesshope.
Details can be found in "Parallel Computers 2, Hockney & Jesshope, 1988" p. 475 - 483

Author: Hannes Signer, 23.11.2025 (signer@uni-potsdam.de)
"""

import numba as nb
from numba import float64, char, int32, optional
from numba.experimental import jitclass

import numpy as np


@nb.njit
def get_diagonal(A, offset):
    res = np.empty(A.shape[0] - abs(offset), dtype=A.dtype)
    for i in range(res.shape[0]):
        if offset >= 0:
            res[i] = A[i, i + offset]
        else:
            res[i] = A[i - offset, i]
    return res


specdiagonal = [
    ("diagonal", float64[:]),
    ("boundary_property", optional(nb.types.string)),
]


@jitclass(spec=specdiagonal)
class diagonal:
    """Represents a diagonal of a tridiagonal matrix with boundary conditions.

    This class stores a diagonal vector and defines the behavior for
    out-of-bounds accesses based on the boundary position (left, center, right).
    These special values are derived in Parallel Computers 2, p. 480.

    Attributes:
        diagonal (np.ndarray): Vector of diagonal elements.
        boundary_property (str): Defines the boundary position ('left', 'center', 'right').
    """

    def __init__(self, diagonal_vector, boundary=None):
        """Initializes a diagonal object.

        Args:
            diagonal_vector (np.ndarray): Vector containing the diagonal elements.
            boundary (str, optional): Boundary position ('left', 'center', 'right'). Default is None.
        """
        self.diagonal = diagonal_vector
        self.boundary_property = boundary

    def get(self, idx):
        """Returns the element at the specified index.

        For out-of-bounds indices, a default value is returned based on the boundary position:
        - 'left': 0
        - 'center': 1
        - 'right': 0

        Args:
            idx (int): Index of the desired element (0-based).

        Returns:
            float: Element at index or default value for boundary elements.
        """
        if idx >= 0 and idx < len(self.diagonal):
            return self.diagonal[idx]

        else:
            if self.boundary_property == "left":
                return 0

            elif self.boundary_property == "center":
                return 1

            elif self.boundary_property == "right":
                return 0

            else:
                return 0


epsilon = 1e-25


@nb.njit
def solve(A, y):
    """Solves the tridiagonal system using the PCR algorithm.
    Iteratively performs the reduction steps over all levels and
    computes the solution x of the system Ax = y.
    Returns:
        np.ndarray: Solution vector x.
    """

    total_levels = int(np.ceil(np.log2(A.shape[0])))
    a_diag = get_diagonal(A, -1)
    b_diag = get_diagonal(A, 0)
    c_diag = get_diagonal(A, 1)
    y = diagonal(y.astype(np.float64), None)

    a = diagonal(
        np.concatenate((np.array([0], dtype=np.float64), a_diag.copy())), "left"
    )
    b = diagonal(b_diag.copy(), "center")
    c = diagonal(
        np.concatenate((c_diag.copy(), np.array([0], dtype=np.float64))), "right"
    )

    for i in range(1, total_levels + 1):
        _update_coefficients(a, b, c, y, level=i)

    result = y.diagonal / b.diagonal
    return result


@nb.njit
def _alpha_gamma_calculation(a, b, c, row: int, level: int):
    """Computes the alpha and gamma coefficients for a row.
    These coefficients are needed for the decoupling of equations in the
    PCR algorithm. Division by zero is avoided using epsilon.
    The calculation of alpha and gamma is shown in Parallel Computers 2, p. 477
    Args:
        row (int): Row number (1-based).
        level (int): Current PCR level.
    Returns:
        tuple[float, float]: Tuple (alpha, gamma) with the decoupling coefficients.
    """
    denominator_alpha = b.get(row - 2 ** (level - 1) - 1)
    denominator_gamma = b.get(row + 2 ** (level - 1) - 1)

    # add epsilon to avoid division by 0 during the algorithm
    # this is necessary for special problem classes
    if b.get(row - 2 ** (level - 1) - 1) == 0.0:
        denominator_alpha += epsilon
    if b.get(row + 2 ** (level - 1) - 1) == 0.0:
        denominator_gamma += epsilon
    alpha = -a.get(row - 1) / denominator_alpha
    gamma = -c.get(row - 1) / denominator_gamma
    return alpha, gamma


@nb.njit
def _decoupling_step(a, b, c, y, row: int, level: int):
    """Computes the new coefficients for a row at the current level.
    Applies the PCR reduction formulas to compute new values for the diagonals
    a, b, c and the right-hand side y.
    Args:
        row (int): Row number (1-based to match the mathematical notation of the algorithm).
        level (int): Current PCR level.
    Hint: Parallelizable over the rows.
    Returns:
        np.ndarray: Array with [a_new, b_new, c_new, y_new] for the row.
    """
    alpha, gamma = _alpha_gamma_calculation(a, b, c, row, level)
    a_row_coeff = alpha * a.get(row - 2 ** (level - 1) - 1)
    c_row_coeff = gamma * c.get(row + 2 ** (level - 1) - 1)
    b_row_coeff = (
        b.get(row - 1)
        + alpha * c.get(row - 2 ** (level - 1) - 1)
        + gamma * a.get(row + 2 ** (level - 1) - 1)
    )
    y_row_coeff = (
        y.get(row - 1)
        + alpha * y.get(row - 2 ** (level - 1) - 1)
        + gamma * y.get(row + 2 ** (level - 1) - 1)
    )
    return np.array([a_row_coeff, b_row_coeff, c_row_coeff, y_row_coeff])


@nb.njit(parallel=True)
def _update_coefficients(a, b, c, y, level):
    """Updates all coefficients for the given level.
    Computes new coefficients for all rows and updates the diagonals
    a, b, c and y. Uses temporary arrays to ensure all calculations
    are performed with the old values.
    Args:
        level (int): PCR level for the update.
    """
    a_new = np.zeros(len(a.diagonal))
    b_new = np.zeros(len(b.diagonal))
    c_new = np.zeros(len(c.diagonal))
    y_new = np.zeros(len(y.diagonal))

    for i in nb.prange(1, len(a.diagonal) + 1):
        coeff = _decoupling_step(a, b, c, y, i, level)
        a_new[i - 1] = coeff[0]
        b_new[i - 1] = coeff[1]
        c_new[i - 1] = coeff[2]
        y_new[i - 1] = coeff[3]

    a.diagonal = a_new
    b.diagonal = b_new
    c.diagonal = c_new
    y.diagonal = y_new


def set_diagonals(matrix, diagonal, diag_index, level):
    """Sets a diagonal into a matrix.

    Writes the values of a diagonal object into the corresponding diagonal
    of a matrix, where the position depends on diag_index and level.
    This method is not necessary for the algorithm but helpful for the visualization
    of the matrices over the pcr levels.

    Args:
        matrix (np.ndarray): Target matrix.
        diagonal (diagonal): Diagonal object with values to insert.
        diag_index (int): Diagonal index (-1: lower, 0: main, 1: upper diagonal).
        level (int): PCR level for determining the start position.

    Returns:
        np.ndarray: Matrix with inserted diagonal.
    """
    v = diagonal.diagonal.copy()
    if diag_index == -1:
        start_col = 0
        start_row = 2**level
        if 2**level < matrix.shape[0]:
            v = np.delete(v, [i for i in range(0, 2**level)])
    elif diag_index == 0:
        start_row = 0
        start_col = 0
    elif diag_index == 1:
        start_row = 0
        start_col = 2**level
        v = np.delete(v, -1)
    for i in range(0, matrix.shape[0] - np.abs(diag_index)):
        idx_row = i + start_row
        idx_col = i + start_col
        if (
            idx_col < 0
            or idx_col > matrix.shape[0] - 1
            or idx_row < 0
            or idx_row > matrix.shape[0] - 1
        ):
            break
        matrix[idx_row, idx_col] = v[i]
    return matrix


def update_matrix(a, b, c, level):
    """Creates a tridiagonal matrix from the diagonals.

    Constructs a complete matrix from the three diagonals a, b, c
    for a given PCR level.
    As for _set_diagonals this method is not necessary for the algorithm but
    helpful for the matrix visualization.

    Args:
        a (diagonal): Lower diagonal.
        b (diagonal): Main diagonal.
        c (diagonal): Upper diagonal.
        level (int): PCR level.

    Returns:
        np.ndarray: Tridiagonal matrix.
    """
    matrix = np.zeros((len(b.diagonal), len(b.diagonal)))
    matrix = matrix.astype(float)
    diag_index = -1
    diagonal_vectors = (a, b, c)
    for i in diagonal_vectors:
        matrix = set_diagonals(matrix, i, diag_index, level)
        diag_index += 1
    return matrix
