"""The demo of using ttopt. Example with QTT.

We'll find the minimum for the 10-dimensional Rosenbrock function with
vectorized input. The target function for minimization has the form f(X), where
input X is the [samples, dimension] numpy array.

The target function and all the selected parameters are the same as in the
"base,py", except that we replace grid size "n" by grid-factors "p" and "q".

As a result of the script work we expect the output in console like this:
"
...
Rosen-10d | k=1.00e+06 | t_cur=4.21e-01 | e_x=6.51e-05 e_y=1.48e-06
----------------------------------------------------------------------
Rosen-10d | k=1.00e+06 | t_all=4.74e+00 | e_x=6.51e-05 e_y=1.48e-06 
"

"""
import numpy as np
from scipy.optimize import rosen


from ttopt import TTOpt
from ttopt import ttopt_init


np.random.seed(282)


d = 10                      # Number of function dimensions:
rmax = 8                    # Maximum TT-rank while cross-like iterations
def f(X):                   # Target function
    return rosen(X.T)


# We initialize the TTOpt class instance with the correct parameters:
tto = TTOpt(
    f=f,                    # Function for minimization. X is [samples, dim]
    d=d,                    # Number of function dimensions
    a=-2.,                  # Grid lower bound (number or list of len d)
    b=+2.,                  # Grid upper bound (number or list of len d)
    p=2,                    # The grid size factor (there will n=p^q points)
    q=20,                   # The grid size factor (there will n=p^q points)
    evals=1.E+6,            # Number of function evaluations
    name='Rosen',           # Function name for log (this is optional)
    x_min_real=np.ones(d),  # Real value of x-minima (x; this is for test)
    y_min_real=0.,          # Real value of y-minima (y=f(x); this is for test)
    with_log=True)


# And now we launching the minimizer:
tto.minimize(rmax)


# We can extract the results of the computation:
x = tto.x_min          # The found value of the minimum of the function (x)
y = tto.y_min          # The found value of the minimum of the function (y=f(x))
x_l = tto.x_min_list   # Intermediate appr. of minima (x) while iterations
y_l = tto.y_min_list   # Intermediate appr. of minima (y=f(x)) while iterations
k_c = tto.k_cache      # Total number of cache usage (should be 0 in this demo)
k_e = tto.k_evals      # Total number of requests to func (is always = evals)
k_t = tto.k_total      # Total number of requests (k_cache + k_evals)
t_f = tto.t_evals_mean # Average time spent to real function call for 1 point
                       # ... (see "ttopt.py" and docs for more details)


# We log the final state:
print('-' * 70 + '\n' + tto.info() +'\n\n')
