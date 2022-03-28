"""Multidimensional minimizer based on the cross-maximum-volume principle.

This module contains the main function "ttopt" that finds the approximate
minimum of the given multidimensional array (tensor), which can represent a
discretized multivariable function.

Note:
    For the task of finding the extremum of a function of many variables or
    multidimensional array, a wrapper class "TTOpt" (from "ttopt.py") could be
    used. It provides a set of methods for discretizing the function, caching
    previously requested values and logging intermediate results. In this case,
    a wrapper "TTOpt.comp_min" should be passed to the function "ttopt" as its
    first argument (the method "TTOpt.minimize" provides the related interface).

"""
from maxvolpy.maxvol import maxvol
from maxvolpy.maxvol import rect_maxvol
import numpy as np


def ttopt(f, n, rmax=5, evals=None, Y0=None, fs_opt=None):
    """Find the minimum element of the implicitly given multidimensional array.

    This function computes the minimum of the implicitly given d-dimensional
    (d >= 2) array (tensor). The adaptive method based on the tensor train (TT)
    approximation and the cross-maximum-volume principle are used.

    Args:
        f (function): the function that returns tensor values for the given set
            of the indices. Its arguments are (I, i_min, y_min, opt_min), where
            "I" represents several multi-indices (samples) for calculation (it
            is 2D np.ndarray of the shape [samples, dimensions]), "i_min"
            represents the current multi-index of the argmin approximation (it
            is 1D np.ndarray of the shape [dimensions]; note that while the
            first call it will be None), "y_min" represents the current
            approximated minimum of the tensor (it is float; note that while
            the first call it will be None) and "opt_min" is the value of the
            auxiliary quantity corresponding to the multi-index "i_min" (it is
            used for debugging and in specific parallel calculations). The
            output of the function should be the corresponding values in the
            given indices (1D np.ndarray of the shape [samples]) and related
            values of the auxiliary quantities at the requested points (1D
            np.ndarray of the shape [samples] of any). If the function returns
            None instead of the tensor values, then the algorithm will be
            interrupted and the current approximation will be returned.
        n (list of len d of int): number of grid points for every dimension
            (i.e., the shape of the tensor). Note that the tensor must have a
            dimension of at least 2.
        rmax (int): maximum used rank for unfolding matrices.
        evals (int or float): number of available calls to function (i.e.,
            computational budget). If it is None, then the algorithm will run
            until the target function returns a None instead of the y-value.
        Y0 (list of 3D np.ndarrays): optional initial tensor in the TT-format
            (it should be represented as a list of the TT-cores). If it is not
            specified, then a random TT-tensor with TT-rank "rmax" will be used.
        fs_opt (float): the parameter of the smoothing function. If it is None,
            then "arctan" function will be used. Otherwise, the function
            "exp(-1 * fs_opt * (p - p0))" will be used.

    Returns:
        [np.ndarray, float]: the multi-index that gives the minimum value of the
        tensor (it is 1D np.ndarray of length "d" of int; i.e., "i_min") and
        the minimum value of the tensor (it is float; i.e., "y_min") that
        corresponds to the multi-index "i_min".

    """
    # Number of dimensions:
    d = len(n)

    # Number of possible function calls:
    evals = int(evals) if evals else None

    # Grid:
    J_grid = [np.reshape(np.arange(k), (-1, 1)) for k in n]

    # Initial tensor:
    Y0, R = ttopt_init(n, rmax, Y0, with_rank=True)

    # Selected multi-indices for all unfolding matrices:
    J = [None] * (d + 1)
    for i in range(d - 1):
        G = Y0[i].reshape(-1, Y0[i].shape[-1])
        q, r = np.linalg.qr(G)
        ind = _maxvol(q)
        J[i + 1] = _stack(J[i], J_grid[i], ind, l2r=True)

    i_min = None         # Approximation of argmin for tensor
    y_min = None         # Approximation of min for tensor (float('inf'))
    opt_min = None       # Additional option related to i_min

    eval = 0             # Number of performed calls to function:
    iter = 0             # Iteration (sweep) number
    i = d - 1            # Index of the current core (0, 1, ..., d-1)
    direction = -1       # Core traversal direction: right (+1) or left (-1)

    while True:
        # We select sample points [samples, d]:
        I = _merge(J[i], J[i+1], J_grid[i])

        # We check if the maximum number of requests has been exceeded:
        eval_curr = I.shape[0]
        if evals is not None and eval + eval_curr > evals:
            I = I[:(evals-eval), :]

        # We compute the function of interest "f" in the sample points I:
        y, opt = f(I, i_min, y_min, opt_min)

        # Function "f" can return None to interrupt the algorithm execution:
        if y is None:
            return i_min, y_min
        else:
            eval += y.size

        # We find and check the minimum value on a set of sampled points:
        i_min, y_min, opt_min = ttopt_find(I, y, opt, i_min, y_min, opt_min)

        # If the max number of requests exceeded, we interrupt the algorithm:
        if evals is not None and eval >= evals:
            return i_min, y_min

        # If computed points less then requested, we interrupt the algorithm:
        if y.shape[0] < I.shape[0]:
            return i_min, y_min

        # We transform sampled points into "core tensor" and smooth it out:
        Z = _reshape(y, (R[i], n[i], R[i + 1]))
        Z = ttopt_fs(Z, y_min, fs_opt)

        # We update ranks and points of interest according to the "core tensor":
        if direction > 0 and i < d - 1:
            # This is left to right sweep:
            Z = _reshape(Z, (R[i] * n[i], R[i + 1]))
            q, r = np.linalg.qr(Z)
            ind = _maxvol_rect(q)
            J[i + 1] = _stack(J[i], J_grid[i], ind, l2r=True)
            R[i + 1] = ind.size
        if direction < 0 and i > 0:
            # This is right to left sweep:
            Z = _reshape(Z, (R[i], n[i] * R[i + 1])).T
            q, r = np.linalg.qr(Z)
            R[i] = min(R[i], rmax)
            q = q[:, :R[i]]
            ind = _maxvol_rect(q)
            J[i] = _stack(J[i+1], J_grid[i], ind, l2r=False)
            R[i] = ind.size

        # We update the current core index according to the traversal direction
        # (when we go through the first or the latest core, then we inverse the
        # traversal direction and increment the iteration counter):
        i += direction
        if i == -1 or i == d:
            iter += 1
            direction = -direction
            i += direction

    return i_min, y_min


def ttopt_find(I, y, opt, i_min, y_min, opt_min):
    """Find the minimum value on a set of sampled points."""
    ind = np.argmin(y)
    y_min_curr = y[ind]

    if y_min is not None and y_min_curr >= y_min:
        return i_min, y_min, opt_min

    return I[ind, :], y_min_curr, opt[ind]


def ttopt_fs(p, p0=0., opt=None):
    """Smooth function that transforms max to min."""
    if opt is None:
        return np.pi/2 - np.arctan(p - p0)
    else:
        return np.exp(-1. * opt * (p - p0))


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


def _maxvol(A, tol=1.01, max_iters=100):
    return maxvol(A, tol=tol, max_iters=max_iters)[0]


def _maxvol_rect(A, kickrank=1, rf=1, tol=1.):
    if kickrank is not None and rf is not None:
        maxK = A.shape[1] + kickrank + rf
    else:
        maxK = None

    return rect_maxvol(A, tol=tol, min_add_K=kickrank, maxK=maxK,
        start_maxvol_iters=10, identity_submatrix=False)[0]


def _merge(J1, J2, Jg):
    n = Jg.shape[0]
    r1 = J1.shape[0] if J1 is not None else 1
    r2 = J2.shape[0] if J2 is not None else 1

    I = np.kron(np.kron(_ones(r2), Jg), _ones(r1))

    if J1 is not None:
        J1_ = np.kron(_ones(n * r2), J1)
        I = np.hstack((J1_, I))

    if J2 is not None:
        J2_ = np.kron(J2, _ones(r1 * n))
        I = np.hstack((I, J2_))

    return I


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _reshape(A, n):
    return np.reshape(A, n, order='F')


def _stack(J, Jg, ind, l2r=True):
    n = Jg.shape[0]
    r = J.shape[0] if J is not None else 1

    J_new = np.kron(Jg, _ones(r)) if l2r else np.kron(_ones(r), Jg)

    if J is not None:
        J_old = np.kron(_ones(n), J) if l2r else np.kron(J, _ones(n))
        J_new = np.hstack((J_old, J_new)) if l2r else np.hstack((J_new, J_old))

    return J_new[ind, :]
