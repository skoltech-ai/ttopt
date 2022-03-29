# ttopt

> Gradient-free optimization method for multivariable functions based on the low rank tensor train (TT) format and maximal-volume principle.


## Installation

1. Install [python](https://www.python.org) (version >= 3.7; you may use [anaconda](https://www.anaconda.com) package manager);
2. Install basic dependencies:
    ```bash
    pip install numpy cython scipy
    ```
3. Install additional dependency [maxvolpy](https://bitbucket.org/muxas/maxvolpy/src/master/) with effective realization of `maxvol` algorithm:
    ```bash
    pip install maxvolpy
    ```
4. Install dependencies for demo calculations (it is optional):
    ```bash
    pip install matplotlib cma
    ```
5. Install the `ttopt` package (from the root folder of the package):
    ```bash
    python setup.py install
    ```


## Documentation

The documentation is located in the `doc` folder. To view the documentation, simply open the file `doc/_build/html/index.html` in a web browser.


## Examples

The demo-scripts with detailed comments are collected in the folder `demo`:

- `base.py` - we find the minimum for the 10-dimensional function with vectorized input;
- `qtt.py` - we do almost the same as in the `base.py` script, but use the QTT-based approach (note that results are much more better then in the `base.py` example);
- `qtt_100d.py` - we do almost the same as in the `qtt.py` script, but approximate the 100-dimensional function;
- `vect.py` - we find the minimum for the simple analytic function with "simple input" (the function is not vectorized);
- `cache.py` - we find the minimum for the simple analytic function to demonstrate the usage of cache;
- `tensor.py` - in this example we find the minimum for the multidimensional array/tensor (i.e., discrete function).


## Calculations for benchmarks

The scripts for comparison of our approach with baselines (ES algorithms) for the analytical benchmark functions are located in the folder `demo_calc`. To run calculations, you can proceed as follows `python demo_calc/run.py --KIND`. Possible values for `KIND`: `comp` - compare different solvers; `iter` - check dependency on number of calls for the target function; `quan` - check effect of the QTT-usage; `rank` - check dependency on the rank; `show` - show results of the previous calculations.

> All results will be collected in the folders `demo_calc/res_data` (saved results in the pickle format), `demo_calc/res_logs` (text files with logs) and `demo_calc/res_plot` (figures with results).

To reproduce the results from the paper (it is currently in the process of being published), run the following scripts from the root folder of the package:
1. Run `python demo_calc/run.py -d 10 -p 2 -q 25 -r 4 --evals 1.E+5 --reps 10 --kind comp`;
2. Run `python demo_calc/run.py -d 10 -p 2 -q 25 -r 4 --reps 10 --kind iter`;
3. Run `python demo_calc/run.py -d 10 -r 4 --evals 1.E+5 --reps 10 --kind quan`;
4. Run `python demo_calc/run.py -d 10 -p 2 -q 25 --evals 1.E+5 --reps 10 --kind rank`;
5. Run `python demo_calc/run.py -d 10 --kind show`. The results will be saved to the `demo_calc/res_logs` and `demo_calc/res_plot` folders.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov) (a.chertkov@skoltech.ru);
- [Ivan Oseledets](https://github.com/oseledets) (i.oseledets@skoltech.ru);
- [Roman Schutski](https://github.com/Qbit-) (r.schutski@skoltech.ru);
- [Konstantin Sozykin](https://github.com/gogolgrind) (konstantin.sozykin@skoltech.ru).
