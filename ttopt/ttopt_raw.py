"""Multidimensional minimizer based on the cross-maximum-volume principle.

This module contains the main function "ttopt" that finds the approximate
minimum of the given multidimensional array (tensor), which can represent a
discretized multidimensional function.

Note:
    For the task of finding the extremum of a function of many variables or
    multidimensional array, a wrapper class "TTOpt" (from "ttopt.py") could be
    used. It provides a set of methods for discretizing the function, caching
    previously requested values and logging intermediate results. In this case,
    a wrapper "TTOpt.comp_min" should be passed to the function "ttopt" as its
    first argument.

"""
from maxvolpy.maxvol import maxvol
from maxvolpy.maxvol import rect_maxvol
import numpy as np


def ttopt(f, n, rmax=5, evals=None, Y0=None):
    """Find the minimum element of the implicitly given multidimensional array.

    This function computes the minimum of the implicitly given d-dimensional
    (d >= 2) array (tensor). The adaptive method based on the tensor train (TT)
    approximation and the cross-maximum-volume principle are used.

    Args:
        f (function): The function that returns tensor values for the given set
            of the indices. Its arguments are (I, i_min, y_min, opt_min), where
            I represents several multiindices (samples) for calculation (is 2D
            numpy array of the shape [samples, d]), i_min represents the
            current multiindex of argmin approximation (is 1D numpy array of
            the shape [d]; while the first call it will be None), y_min
            represents the current approximated minimum of the tensor (is
            float; while the first call it will be None) and opt_min is the
            value of the auxiliary quantity corresponding to the multi-index
            i_min (it is used for debugging and in specific parallel
            calculations). The output of the function should be the
            corresponding values in the given indices (1D numpy array of the
            shape [samples]) and related values of the auxiliary quantities at
            the requested points (1D numpy array of the shape [samples] of any).
            If the function returns None instead of the tensor values, then the
            algorithm will be interrupted and the current approximation will be
            returned.
        n (list of len d of int): Number of grid points for every dimension
            (i.e., the shape of the tensor). Note that the tensor must have a
            dimension of at least 2.
        rmax (int): Maximum rank.
        evals (int or float): Number of available calls to function. If it is
            None, then the algorithm will run until the target function returns
            a None instead of the y-value.
        Y0 (list of 3D np.ndarrays): optional initial tensor in the TT-format
            (it should be represented as a list of the TT-cores). If not
            specified, then a random TT-tensor with TT-rank rmax will be used.

    Returns:
        [np.ndarray, float]: The multiindex that gives the minimum value of the
        tensor (is 1D numpy array of length d of int; "i_min") and the minimum
        value of the tensor (is float; "y_min") that corresponds to the
        multi-index "i_min".

    """
    # Number of dimensions:
    d = len(n)

    # Number of possible function calls:
    evals = int(evals) if evals else None

    # Grid:
    I_grid = [np.reshape(np.arange(n[i]), (-1, 1)) for i in range(d)]

    # Initial tensor:
    Y0, R = ttopt_init(n, rmax, Y0, with_rank=True)

    # Points:
    J = [np.empty(0, dtype=int)] * (d + 1)
    for i in range(d - 1):
        G = Y0[i].reshape(-1, Y0[i].shape[-1])
        q, r = _qr(G)
        ind = _maxvol(q)
        J[i + 1] = ttopt_index_stack_l2r(J[i], I_grid[i], n[i], R[i], ind)

    i_min = None         # Approximation of argmin for tensor
    y_min = None         # Approximation of min for tensor (float('inf'))
    opt_min = None       # Additional option related to i_min

    eval = 0             # Number of performed calls to function:
    iter = 0             # Iteration (sweep) number
    i = d - 1            # Index of the current core (0, 1, ..., d-1)
    direction = -1       # Core traversal direction: right (+1) or left (-1)

    while True:
        # We select sample points [samples, d]:
        if np.size(J[i]) == 0:
            w1 = _zeros(R[i] * n[i] * R[i + 1])
        else:
            w1 = _kron(_ones(n[i] * R[i + 1]), J[i])
        w2 = _kron(_kron(_ones(R[i + 1]), I_grid[i]), _ones(R[i]))
        if np.size(J[i + 1]) == 0:
            w3 = _zeros(R[i] * n[i] * R[i + 1])
        else:
            w3 = _kron(J[i + 1], _ones(R[i] * n[i]))
        I_curr = np.hstack((w1, w2, w3))

        # We check if the maximum number of requests has been exceeded:
        eval_curr = I_curr.shape[0]
        if evals is not None and eval + eval_curr > evals:
            I_curr = I_curr[:(evals-eval), :]

        # We compute the function of interest "f" in the sample points I_curr:
        Y_curr, opt_curr = f(I_curr, i_min, y_min, opt_min)

        # Function "f" can return None to interrupt the algorithm execution:
        if Y_curr is None:
            return i_min, y_min
        else:
            eval += Y_curr.size

        # We find and check the minimum value on a set of sampled points:
        i_min, y_min, opt_min = ttopt_find(
            I_curr, Y_curr, opt_curr, i_min, y_min, opt_min)

        # If the max number of requests exceeded, we interrupt the algorithm:
        if evals is not None and eval >= evals:
            return i_min, y_min

        # If computed points less then requested, we interrupt the algorithm:
        if Y_curr.shape[0] < I_curr.shape[0]:
            return i_min, y_min

        # We transform sampled points into "core tensor" and smooth it out:
        Z = _reshape(Y_curr, (R[i], n[i], R[i + 1]))
        Z = ttopt_fs(Z, y_min)

        # We update ranks and points of interest according to the "core tensor":
        if direction > 0 and i < d - 1:
            # This is left to right sweep:
            Z = _reshape(Z, (R[i] * n[i], R[i + 1]))
            q, r = _qr(Z)
            ind = _maxvol_rect(q)
            R[i + 1] = ind.size
            J[i + 1] = ttopt_index_stack_l2r(J[i], I_grid[i], n[i], R[i], ind)

        if direction < 0 and i > 0:
            # This is right to left sweep:
            Z = _reshape(Z, (R[i], n[i] * R[i + 1]))
            Z = Z.T
            u, s, v = _svd(Z)
            R[i] = min(R[i], rmax)
            q = u[:, :R[i]]
            ind = _maxvol_rect(q)
            R[i] = ind.size
            J[i] = ttopt_index_stack_r2l(J[i+1], I_grid[i], n[i], R[i+1], ind)

        # We update the current core index according to the traversal direction
        # (when we go through the first or the latest core, then we inverse the
        # traversal direction and increment the iteration counter):
        i += direction
        if i == -1 or i == d:
            iter += 1
            direction = -direction
            i += direction

    return i_min, y_min


def ttopt_find(I_curr, Y_curr, opt_curr, i_min, y_min, opt_min):
    """Find the minimum value on a set of sampled points."""
    ind = np.argmin(Y_curr)
    y_min_curr = Y_curr[ind]

    if y_min is not None and y_min_curr >= y_min:
        return i_min, y_min, opt_min

    return I_curr[ind, :], y_min_curr, opt_curr[ind]


def ttopt_fs(p, p0=0.):
    """Smooth function that transforms max to min."""
    # return np.exp(-10*(p - p0))
    return np.pi/2 - np.arctan(p - p0)


def ttopt_index_stack_r2l(J, x, n, r, ind):
    w1 = _kron(_ones(r), x)
    w2 = _zeros(n * r) if np.size(J) == 0 else _kron(J, _ones(n))
    J_new = np.hstack((w1, w2))
    J_new = _reshape(J_new, (n * r, -1))
    J_new = J_new[ind, :]
    return J_new


def ttopt_index_stack_l2r(J, x, n, r, ind):
    w1 = _kron(_ones(n), J)
    w2 = _kron(x, _ones(r))
    J_new = np.hstack((w1, w2))
    J_new = _reshape(J_new, (r * n, -1))
    J_new = J_new[ind, :]
    return J_new


def ttopt_init(n, rmax, Y0=None, with_rank=False):
    """Build initial approximation for the main algorithm."""
    d = len(n)

    R = [1]
    for i in range(1, d):
        R.append(min(rmax, n[i-1] * R[i-1]))
    R.append(1)

    if Y0 is None:
        Y0 = [np.random.randn(R[i], n[i], R[i + 1]) for i in range(d)]

    if with_rank:
        return Y0, R
    else:
        return Y0


def _kron(a, b):
    return np.kron(a, b)


def _maxvol(a, tol=1.01, max_iters=100):
    return maxvol(a, tol=tol, max_iters=max_iters)[0]


def _maxvol_rect(a, kickrank=1, rf=1, tol=1.):
    if kickrank is not None and rf is not None:
        maxK = a.shape[1] + kickrank + rf
    else:
        maxK = None

    return rect_maxvol(a, tol=tol, min_add_K=kickrank, maxK=maxK,
        start_maxvol_iters=10, identity_submatrix=False)[0]


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _qr(a):
    return np.linalg.qr(a)


def _reshape(a, shape):
    return np.reshape(a, shape, order='F')


def _svd(a, full_matrices=False):
    try:
        return np.linalg.svd(a, full_matrices=full_matrices)
    except:
        b = a + 1.E-14 * np.max(np.abs(a)) * np.random.randn(*a.shape)
        return np.linalg.svd(b, full_matrices=full_matrices)


def _zeros(k, m=0):
    return np.zeros((k, m), dtype=int)
