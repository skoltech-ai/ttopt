Documentation
=============

:ttopt:
    Gradient-free global optimization algorithm for multivariable functions and multidimensional arrays based on the low rank tensor train (TT) format and maximum volume principle.


.. toctree::
  :maxdepth: 1

  ttopt
  ttopt_raw

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`


The demo-scripts with detailed comments are collected in the folder `demo`:

- `base.py` - we find the minimum for the 10-dimensional function with vectorized input;

- `qtt.py` - we do almost the same as in the `base.py` script, but use the QTT-based approach (note that results are much more better then in the `base.py` example);

- `qtt_100d.py` - we do almost the same as in the `qtt.py` script, but approximate the 100-dimensional function;

- `vect.py` - we find the minimum for the simple analytic function with "simple input" (the function is not vectorized);

- `cache.py` - we find the minimum for the simple analytic function to demonstrate the usage of cache;

- `tensor.py` - in this example we find the minimum for the multidimensional array/tensor (i.e., discrete function).

- `tensor_init_spec` - we do almost the same as in the `tensor.py` script, but use special method of initialization (instead of a random tensor, we select a set of starting multi-indices for the search).
