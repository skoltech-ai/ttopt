import numpy as np
from time import perf_counter as tpc


from .ttopt_raw import ttopt
from .ttopt_raw import ttopt_find


class TTOpt():
    """Multidimensional minimizer based on the cross-maximum-volume principle.

    Class for computation of the minimum for the implicitly given d-dimensional
    array (tensor) or a function of d-dimensional argument. An adaptive method
    based on the tensor train (TT) approximation and the the cross-maximum
    volume principle is used. Cache of requested values (its usage leads to
    faster computation if one point is computed for a long time) and QTT-based
    representation of the grid (its usage in many cases leads to more accurate
    results) are supported.

    Args:
        f (function): The function of interest. Its argument X should represent
            several spatial points for calculation (is 2D numpy array of the
            shape [samples, d]) if "is_vect" flag is True, and it is one
            spatial point for calculation (is 1D numpy array of the shape [d])
            in the case if "is_vect" flag is False. For the case of the tensor
            approximation (if "is_func" flag is False), the argument X relates
            to the one or many (depending on the value of the flag "is_vect")
            multi indices of the corresponding array/tensor. Function should
            return the values in the requested points (is 1D numpy array of the
            shape [samples] of float or only one float value depending on the
            value of "is_vect" flag). If "with_opt" flag is True, then function
            should also return the second argument (is 1D numpy array of the
            shape [samples] of any or just one value depending on the "is_vect"
            flag) which is the auxiliary quantity corresponding to the
            requested points (it is used for debugging and in specific parallel
            calculations; the value of this auxiliary quantity related to the
            "argmin" point will be passed to "callback" function).
        d (int): Number of function dimensions.
        a (float or list of len d of float): Grid lower bounds for every
            dimension. If a number is given, then this value will be used for
            each dimension.
        b (float or list of len d of float): Grid upper bounds for every
            dimension. If a number is given, then this value will be used for
            each dimension.
        n (int or list of len d of int): Number of grid points for every
            dimension. If a number is given, then this value will be used for
            each dimension. If this parameter is not specified, then instead of
            it the values for both "p" and "q" should be set.
        p (int): The grid size factor (if is given, then there will be n=p^q
            points for each dimension). This parameter can be specified instead
            of "n". If this parameter is specified, then the parameter "q" must
            also be specified, and in this case the QTT-based approach will be
            used.
        q (int): The grid size factor (if is given, then there will be n=p^q
            points for each dimension). This parameter can be specified instead
            of "n". If this parameter is specified, then the parameter "p" must
            also be specified, and in this case the QTT-based approach will be
            used.
        evals (int or float): The number of requests to the target function
            that will be made.
        name (str): Optional display name for the function of interest. It is
            the empty string by default.
        callback (function): Optional function that will be called after each
            optimization step (in Func.comp.min) with related info (it is used
            for debugging and in specific parallel calculations).
        x_min_real (list of len d): Optional real value of x-minima (x). If
            this value is specified, then it will be used to display the
            current approximation error within the algorithm iterations (this
            is convenient for debugging and testing/research).
        y_min_real (float): Optional real value of y-minima (y=f(x)). If
            this value is specified, then it will be used to display the
            current approximation error within the algorithm iterations (this
            is convenient for debugging and testing/research).
        is_func (bool): If flag is True, then we minimize the function (the
            arguments of f correspond to continuous spatial points), otherwise
            we approximate the tensor (the arguments of f correspond to
            discrete multidimensional tensor indices). It is True by default.
        is_vect (bool): If flag is True, then function should accept 2D
            numpy array of the shape [samples, d] and return 1D numpy array
            of the shape [samples]. Otherwise, the function should accept 1D
            numpy array (one multidimensional point) and return the float
            value. It is True by default.
        with_cache (bool): If flag is True, then all requested values are
            stored and retrieved from the storage upon repeated requests.
            Note that this leads to faster computation if one point is
            computed for a long time. On the other hand, this can in some
            cases slow down the process, due to the additional time spent
            on checking the storage and using unvectorized slow loops in
            python. It is False by default.
        with_log (bool): If flag is True, then text messages will be
            displayed during the optimizer query process. It is False by
            default.
        with_opt (bool): If flag is True, then function of interest returns
            opts related to output y (scalar or vector) as second argument
            (it will be also saved and passed to "callback" function). It is
            False by default.

    Note:
        Call "calc" to evaluate function for one tensor multi index and call
        "comp" to evaluate function in the set of grid points (both of these
        functions can be called regardless of the value of the flag "is_vect").

    """

    def __init__(self, f, d, a=None, b=None, n=None, p=None, q=None, evals=None,
        name=None, callback=None, x_min_real=None, y_min_real=None,
        is_func=True, is_vect=True, with_cache=False, with_log=False,
        with_opt=False):
        # The target function and its dimension:
        self.f = f
        self.d = d

        # Set grid lower bound:
        if isinstance(a, (int, float)):
            self.a = np.ones(self.d, dtype=float) * a
        elif a is not None:
            self.a = np.asanyarray(a, dtype=float)
        else:
            if is_func:
                raise ValueError('Grid lower bound (a) should be set')
            self.a = None
        if self.a is not None:
            assert self.a.size == self.d

        # Set grid upper bound:
        if isinstance(b, (int, float)):
            self.b = np.ones(self.d, dtype=float) * b
        elif a is not None:
            self.b = np.asanyarray(b, dtype=float)
        else:
            if is_func:
                raise ValueError('Grid upper bound (b) should be set')
            self.b = None
        if self.b is not None:
            assert self.b.size == self.d

        # Set number of grid points:
        if n is None:
            if p is None or q is None:
                raise ValueError('If n is not set, then p and q should be set')
            self.p = int(p)
            self.q = int(q)
            self.n = np.ones(self.d * self.q, dtype=int) * self.p
            self.n_func = np.ones(self.d, dtype=int) * self.p**self.q
        else:
            if p is not None or q is not None:
                raise ValueError('If n is set, then p and q should be None')
            self.p = None
            self.q = None
            if isinstance(n, (int, float)):
                self.n = np.ones(self.d, dtype=int) * int(n)
            else:
                self.n = np.asanyarray(n, dtype=int)
            self.n_func = self.n.copy()
        assert self.n_func.size == self.d

        # Set other options according to the input arguments:
        self.evals = int(evals) if evals else None
        self.name = name or ''
        self.callback = callback
        self.x_min_real = x_min_real
        self.y_min_real = y_min_real
        self.is_func = bool(is_func)
        self.is_vect = bool(is_vect)
        self.with_cache = bool(with_cache)
        self.with_log = bool(with_log)
        self.with_opt = bool(with_opt)

        # Inner variables:
        self.cache = {}     # Cache for the results of requests to function
        self.cache_opt = {} # Cache for the options while requests to function
        self.k_cache = 0    # Number of requests, then cache was used
        self.k_evals = 0    # Number of requests, then function was called
        self.t_evals = 0.   # Total time of function calls
        self.t_total = 0.   # Total time of computations (including cache usage)
        self.t_minim = 0    # Total time of work for minimizator
        self._opt = None    # Function opts related to its output

        # Approximations for argmin/min/opts of the function while iterations:
        self.i_min_list = []
        self.x_min_list = []
        self.y_min_list = []
        self.opt_min_list = []

    @property
    def e_x(self):
        """Current error for approximation of argmin of the function."""
        if self.x_min_real is not None and  self.x_min is not None:
            return np.linalg.norm(self.x_min - self.x_min_real)

    @property
    def e_y(self):
        """Current error for approximation of the minumum of the function."""
        if self.y_min_real is not None or self.y_min is not None:
            return np.abs(self.y_min - self.y_min_real)

    @property
    def i_min(self):
        """Current approximation of argmin (ind) of the function of interest."""
        return self.i_min_list[-1] if len(self.i_min_list) else None

    @property
    def k_total(self):
        """Total number of requests (both function calls and cache usage)."""
        return self.k_cache + self.k_evals

    @property
    def opt_min(self):
        """Current value of option of the function related to min-point."""
        return self.opt_min_list[-1] if len(self.opt_min_list) else None

    @property
    def t_evals_mean(self):
        """Average time spent to real function call for 1 point."""
        return self.t_evals / self.k_evals if self.k_evals else 0.

    @property
    def t_total_mean(self):
        """Average time spent to return one function value."""
        return self.t_total / self.k_total if self.k_total else 0.

    @property
    def x_min(self):
        """Current approximation of argmin of the function of interest."""
        return self.x_min_list[-1] if len(self.x_min_list) else None

    @property
    def y_min(self):
        """Current approximation of min of the function of interest."""
        return self.y_min_list[-1] if len(self.y_min_list) else None

    def _eval(self, i):
        """Helper that computes target function in one or many points."""
        t_evals = tpc()

        if isinstance(i, list):
            i = np.array(i)
        i = i.astype(int)

        is_many = len(i.shape) == 2

        if self.is_func:
            x = self.i2x_many(i) if is_many else self.i2x(i)
        else:
            x = i

        if self.with_opt:
            y, self._opt = self.f(x)
        else:
            y = self.f(x)
            self._opt = [None for _ in range(y.size)] if is_many else None

        self.k_evals += y.size if is_many else 1
        self.t_evals += tpc() - t_evals

        return y

    def calc(self, i):
        """Calculate the function for the given multiindex.

        Args:
            i (np.ndarray): The input for the function, that is 1D numpy array
                of the shape [d] of int (indices).

        Returns:
            float: The output of the function.

        """
        if self.is_vect:
            return self.comp(i.reshape(1, -1))[0]

        t_total = tpc()

        if not self.with_cache:
            y = self._eval(i)
            self.t_total += tpc() - t_total
            return y

        s = self.i2s(i.astype(int).tolist())

        if not s in self.cache:
            y = self._eval(i)
            self.cache[s] = y
            self.cache_opt[s] = self._opt
        else:
            y = self.cache[s]
            self._opt = self.cache_opt[s]
            self.k_cache += 1

        self.t_total += tpc() - t_total

        return y

    def comp(self, I):
        """Compute the function for the set of multi indices (samples).

        Args:
            I (np.ndarray): The inputs for the function, that are collected in
                2D numpy array of the shape [samples, d] of int (indices).

        Returns:
            np.ndarray: The outputs of the function, that are collected in
                1D numpy array of the shape [samples].

        Note:
            The set of points (I) should not contain duplicate points. If it
            contains duplicate points (that are not in the cache), then each of
            them will be recalculated without using the cache.

        Todo:
            This function may be implemented more efficiently.

        """
        if not self.is_vect:
            Y, _opt = [], []
            for i in I:
                Y.append(self.calc(i))
                _opt.append(self._opt)
            self._opt = _opt
            return np.array(Y)

        t_total = tpc()

        if not self.with_cache:
            Y = self._eval(I)
            self.t_total += tpc() - t_total
            return Y

        # Requested points:
        I = I.tolist()

        # Points that are not presented in the cache:
        J = [i for i in I if self.i2s(i) not in self.cache]
        self.k_cache += len(I) - len(J)

        # We add new points (J) to the storage:
        if len(J):
            Z = self._eval(J)
            for k, j in enumerate(J):
                s = self.i2s(j)
                self.cache[s] = Z[k]
                self.cache_opt[s] = self._opt[k]

        # We obtain the values for requested points from the updated storage:
        Y = np.array([self.cache[self.i2s(i)] for i in I])
        self._opt = np.array([self.cache_opt[self.i2s(i)] for i in I])

        self.t_total += tpc() - t_total
        return Y

    def comp_min(self, I, i_min=None, y_min=None, opt_min=None):
        """Compute the function for the set of points and save current minimum.

        This helper function (this is wrapper for function "comp") can be
        passed to the optimizer. When making requests, the optimizer must pass
        the grid points of interest (I) as arguments, as well as the current
        approximation of the argmin (i_min), the corresponding value (y_min)
        and related option value (opt_min).

        Todo:
            Update callback args.

        """
        # We return None if the limit for function requests is exceeded:
        if self.evals is not None and self.k_evals >= self.evals:
            return None, None

        # We truncate the list of requested points if it exceeds the limit:
        eval_curr = I.shape[0]
        is_last = self.evals is not None and self.k_evals+eval_curr>=self.evals
        if is_last:
            I = I[:(self.evals-self.k_evals), :]

        if self.q:
            # The QTT is used, hence we should transform the indices:
            if I is not None:
                I = self.qtt_parse_many(I)
            if i_min is not None:
                i_min = self.qtt_parse_many(i_min.reshape(1, -1))[0, :]

        Y = self.comp(I)

        # If this is last iteration, we should "manually" check for y_opt_new:
        if is_last:
            i_min, y_min, opt_min = ttopt_find(
                I, Y, self._opt, i_min, y_min, opt_min)

        if i_min is None:
            return Y, self._opt

        if self.is_func:
            x_min = self.i2x(i_min)
        else:
            x_min = i_min.copy()

        self.i_min_list.append(i_min.copy())
        self.x_min_list.append(x_min)
        self.y_min_list.append(y_min)
        self.opt_min_list.append(opt_min)

        is_better = len(self.y_min_list) == 1 or (y_min < self.y_min_list[-2])
        if self.callback and is_better:
            last = {'last': [x_min, y_min, i_min, opt_min, self.k_evals]}
            self.callback(last)

        if self.with_log:
            print(self.info(is_final=False))

        return Y, self._opt

    def i2s(self, i):
        """Transforms array of int like [1, 2, 3] into string like '1-2-3'."""
        return '-'.join([str(v) for v in i])

    def i2x(self, i):
        """Transforms multiindex into point of the uniform grid.

        Todo:
            We can add support for nonuniform grids.

        """
        t = i * 1. / (self.n_func - 1)
        x = t * (self.b - self.a) + self.a
        return x

    def i2x_many(self, I):
        """Transforms multiindices (samples) into grid points.

        Todo:
            We can add support for nonuniform grids.

        """
        A = np.repeat(self.a.reshape((1, -1)), I.shape[0], axis=0)
        B = np.repeat(self.b.reshape((1, -1)), I.shape[0], axis=0)
        N = np.repeat(self.n_func.reshape((1, -1)), I.shape[0], axis=0)
        T = I * 1. / (N - 1)
        X = T * (B - A) + A
        return X

    def info(self, is_final=True):
        """Return text description of the progress of optimizer work."""
        text = ''

        if self.name:
            text += self.name + f'-{self.d}d | '

        if self.with_cache:
            text += f'k={self.k_evals:-8.2e}+{self.k_cache:-8.2e} | '
        else:
            text += f'k={self.k_total:-8.2e} | '

        if is_final:
            text += f't_all={self.t_minim:-8.2e} | '
        else:
            text += f't_cur={self.t_total:-8.2e} | '

        if self.y_min_real is None and self.y_min is not None:
            text += f'y={self.y_min:-.6f} '
        else:
            if self.e_x is not None:
                text += f'e_x={self.e_x:-8.2e} '
            if self.e_y is not None:
                text += f'e_y={self.e_y:-8.2e} '

        return text

    def minimize(self, rmax=10, Y0=None):
        """Perform the function minimization process by TT-based approach.

        Args:
            rmax (int): Maximum TT-rank.
            Y0 (list of 3D np.ndarrays of float): optional initial tensor in
                the TT format as a list of the TT-cores.

        """
        t_minim = tpc()
        i_min, y_min = ttopt(self.comp_min, self.n, rmax, None, Y0)
        self.t_minim = tpc() - t_minim

    def qtt_parse_many(self, I_qtt):
        """Transforms tensor indices from QTT (long) to base (short) format.

        Todo:
            Optimize the code!

        """
        samples = I_qtt.shape[0]
        n_qtt = [self.n[0]]*self.q
        I = np.zeros((samples, self.d))
        for i in range(self.d):
            J_curr = I_qtt[:, self.q*i:self.q*(i+1)].T
            I[:, i] = np.ravel_multi_index(J_curr, n_qtt, order='F')
        return I

    def s2i(self, s):
        """Transforms string like '1-2-3' into array of int like [1, 2, 3]."""
        return np.array([int(v) for v in s.split('-')], dtype=int)
