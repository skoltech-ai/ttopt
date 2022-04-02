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


def ttopt(f, n, rmax=5, evals=None, Y0=None, fs_opt=1., add_opt_inner=True, add_opt_outer=False, add_opt_rect=False, add_rnd_inner=False, add_rnd_outer=False):
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
    Jg_list = [np.reshape(np.arange(k), (-1, 1)) for k in n]

    # Initial tensor:
    Y0, r = ttopt_init(n, rmax, Y0, with_rank=True)

    # Prepare initial multi-indices for all unfolding matrices:
    J_list = [None] * (d + 1)
    for i in range(d - 1):
        J_list[i+1] = _iter(Y0[i], J_list[i], Jg_list[i], l2r=True)

    i_min = None         # Approximation of argmin for tensor
    y_min = None         # Approximation of min for tensor (float('inf'))
    opt_min = None       # Additional option related to i_min

    eval = 0             # Number of performed calls to function
    iter = 0             # Iteration (sweep) number
    i = d - 1            # Index of the current core (0, 1, ..., d-1)
    l2r = False          # Core traversal direction (left <-> right)

    while True:
        # We select multi-indices [samples, d], which will requested from func:
        I = _merge(J_list[i], J_list[i+1], Jg_list[i])

        # We check if the maximum number of requests has been exceeded:
        eval_curr = I.shape[0]
        if evals is not None and eval + eval_curr > evals:
            I = I[:(evals-eval), :]

        # We compute the function of interest "f" in the sample points I:
        y, opt = f(I, i_min, y_min, opt_min)

        # Function "f" can return None to interrupt the algorithm execution:
        if y is None:
            return i_min, y_min

        # We find and check the minimum value on a set of sampled points:
        i_min, y_min, opt_min = ttopt_find(I, y, opt, i_min, y_min, opt_min)

        # If the max number of requests exceeded, we interrupt the algorithm:
        eval += y.size
        if evals is not None and eval >= evals:
            return i_min, y_min

        # If computed points less then requested, we interrupt the algorithm:
        if y.shape[0] < I.shape[0]:
            return i_min, y_min

        # We transform sampled points into "core tensor" and smooth it out:
        Z = _reshape(y, (r[i], n[i], r[i + 1]))
        Z = ttopt_fs(Z, y_min, fs_opt)

        # We perform iteration:
        if l2r and i < d - 1:
            J_list[i+1] = _iter(Z, J_list[i], Jg_list[i], l2r,
                add_opt_inner, add_opt_rect, add_rnd_inner)
            if add_opt_outer:
                J_list[i+1] = _add_row(J_list[i+1], i_min[:(i+1)])
            if add_rnd_outer:
                J_list[i+1] = _add_random(J_list[i+1], n[:(i+1)])
            r[i+1] = J_list[i+1].shape[0]
        if not l2r and i > 0:
            J_list[i] = _iter(Z, J_list[i+1], Jg_list[i], l2r,
                add_opt_inner, add_opt_rect, add_rnd_inner)
            if add_opt_outer:
                J_list[i] = _add_row(J_list[i], i_min[i:])
            if add_rnd_outer:
                J_list[i] = _add_random(J_list[i], n[i:])
            r[i] = J_list[i].shape[0]

        # We update the current core index:
        i, iter, l2r = _update_iter(d, i, iter, l2r)

    return i_min, y_min


def ttopt_find(I, y, opt, i_min, y_min, opt_min):
    """Find the minimum value on a set of sampled points."""
    ind = np.argmin(y)
    y_min_curr = y[ind]

    if y_min is not None and y_min_curr >= y_min:
        return i_min, y_min, opt_min

    return I[ind, :], y_min_curr, opt[ind]


def ttopt_fs(y, y0=0., opt=1.):
    """Smooth function that transforms max to min."""
    if opt is None or opt == 0:
        return np.pi/2 - np.arctan(y - y0)
    else:
        return np.exp(opt * (y0 - y))


def ttopt_init(n, rmax, Y0=None, with_rank=False):
    """Build initial approximation for the main algorithm."""
    d = len(n)

    r = [1]
    for i in range(1, d):
        r.append(min(rmax, n[i-1] * r[i-1]))
    r.append(1)

    if Y0 is None:
        Y0 = [np.random.randn(r[i], n[i], r[i + 1]) for i in range(d)]

    if with_rank:
        return Y0, r
    else:
        return Y0


def _add_random(J, n):
    i_rnd = [np.random.choice(k) for k in n]
    i_rnd = np.array(i_rnd, dtype=int)
    J_new = np.vstack((J, i_rnd.reshape(1, -1)))
    return J_new


def _add_row(J, i_new):
    J_new = np.vstack((J, i_new.reshape(1, -1)))
    return J_new


def _iter(Z, J, Jg, l2r=True, add_opt_inner=True, add_opt_rect=False, add_rnd_inner=False):
    r1, n, r2 = Z.shape

    Z = _reshape(Z, (r1 * n, r2)) if l2r else _reshape(Z, (r1, n * r2)).T

    Q, R = np.linalg.qr(Z)

    ind = _maxvol(Q, is_rect=add_opt_rect)

    if add_opt_inner:
        i_max, j_max = np.divmod(np.abs(Z).argmax(), Z.shape[1])
        if not i_max in ind:
            ind[-1] = i_max

    if add_rnd_inner and len(ind) > 1:
        i_rnd = np.random.choice(Z.shape[0])
        if not i_rnd in ind:
            ind[-2] = i_rnd

    J_new = _stack(J, Jg, l2r)
    J_new = J_new[ind, :]

    return J_new


def _maxvol(A, tol=1.001, max_iters=1000, is_rect=False):
    if is_rect:
        tol_rect = 1.
        kickrank = 1
        rf = 1.

        if kickrank is not None and rf is not None:
            maxK = A.shape[1] + kickrank + rf
        else:
            maxK = None

        return rect_maxvol(A, tol=tol_rect, min_add_K=kickrank, maxK=maxK,
            start_maxvol_iters=10, identity_submatrix=False)[0]
    else:
        return maxvol(A, tol=tol, max_iters=max_iters)[0]


def _merge(J1, J2, Jg):
    r1 = J1.shape[0] if J1 is not None else 1
    r2 = J2.shape[0] if J2 is not None else 1
    n = Jg.shape[0]

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


def _stack(J, Jg, l2r=True):
    r = J.shape[0] if J is not None else 1
    n = Jg.shape[0]

    J_new = np.kron(Jg, _ones(r)) if l2r else np.kron(_ones(r), Jg)

    if J is not None:
        J_old = np.kron(_ones(n), J) if l2r else np.kron(J, _ones(n))
        J_new = np.hstack((J_old, J_new)) if l2r else np.hstack((J_new, J_old))

    return J_new


def _update_iter(d, i, iter, l2r):
    i += 1 if l2r else -1

    if i == -1 or i == d:
        iter += 1
        l2r = not l2r
        i += 1 if l2r else -1

    return i, iter, l2r
