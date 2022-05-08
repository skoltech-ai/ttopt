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

The documentation is located in the `doc` folder. To view the documentation, simply open the file `doc/_build/html/index.html` in any web browser.


## Examples

The demo-scripts with detailed comments are collected in the folder `demo`:

- `base.py` - we find the minimum for the 10-dimensional function with vectorized input;
- `qtt.py` - we do almost the same as in the `base.py` script, but use the QTT-based approach (note that results are much more better then in the `base.py` example);
- `qtt_100d.py` - we do almost the same as in the `qtt.py` script, but approximate the 100-dimensional function;
- `vect.py` - we find the minimum for the simple analytic function with "simple input" (the function is not vectorized);
- `cache.py` - we find the minimum for the simple analytic function to demonstrate the usage of cache;
- `tensor.py` - in this example we find the minimum for the multidimensional array/tensor (i.e., discrete function).
- `tensor_init_spec` - we do almost the same as in the `tensor.py` script, but use special method of initialization (instead of a random tensor, we
select a set of starting multi-indices for the search).


## Calculations for benchmarks

The scripts for comparison of our approach with baselines (ES algorithms) for the analytical benchmark functions are located in the folder `demo_calc`. To run calculations, you can proceed as follows `python demo_calc/run.py --KIND`. Possible values for `KIND`: `comp` - compare different solvers; `iter` - check dependency on number of calls for the target function; `quan` - check effect of the QTT-usage; `rank` - check dependency on the rank; `show` - show results of the previous calculations.

> All results will be collected in the folders `demo_calc/res_data` (saved results in the pickle format), `demo_calc/res_logs` (text files with logs) and `demo_calc/res_plot` (figures with results).

To reproduce the results from the paper (it is currently in the process of being published), run the following scripts from the root folder of the package:
1. Run `python demo_calc/run.py -d 10 -p 2 -q 25 -r 4 --evals 1.E+5 --reps 10 --kind comp`;
2. Run `python demo_calc/run.py -d 10 -p 2 -q 25 -r 4 --reps 10 --kind iter`;
3. Run `python demo_calc/run.py -d 10 -r 4 --evals 1.E+5 --reps 10 --kind quan`;
4. Run `python demo_calc/run.py -d 10 -p 2 -q 25 --evals 1.E+5 --reps 10 --kind rank`;
5. Run `python demo_calc/run.py -d 100 -p 2 -q 25 -r 4 --evals 1.E+6 --reps 10 --kind comp`;
6. Run `python demo_calc/run.py -p 2 -q 25 -r 4 --reps 1 --kind dim`;
7. Run `python demo_calc/run.py -d 4 -p 2 -q 25 -r 4 --evals 1.E+5 --reps 10 --kind comp`;
8. Run `python demo_calc/run.py -d 10 --kind show`. The results will be saved to the `demo_calc/res_logs` and `demo_calc/res_plot` folders.

> **The scripts in this folder have not been updated since the transition to the new version of the code. The presented results correspond to the version 0.2 of the software product.**


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov) (a.chertkov@skoltech.ru);
- [Ivan Oseledets](https://github.com/oseledets) (i.oseledets@skoltech.ru);
- [Roman Schutski](https://github.com/Qbit-) (r.schutski@skoltech.ru);
- [Konstantin Sozykin](https://github.com/gogolgrind) (konstantin.sozykin@skoltech.ru).


## Citation

If you find this approach and/or code useful in your research, please consider citing:

```bibtex
@article{sozykin2022ttopt,
    author    = {Sozykin, Konstantin and Chertkov, Andrei and Schutski, Roman and Phan, Anh-Huy and Cichocki, Andrzej and Oseledets, Ivan},
    year      = {2022},
    title     = {TTOpt: A Maximum Volume Quantized Tensor Train-based Optimization and its Application to Reinforcement Learning},
    journal   = {ArXiv},
    volume    = {abs/2205.00293},
    doi       = {10.48550/ARXIV.2205.00293},
    url       = {https://arxiv.org/abs/2205.00293}
}
```
