# ttopt


## Description

Gradient-free global optimization algorithm for multidimensional functions based on the low rank tensor train (TT) format and maximum volume principle.

**Notes**:

- At the moment, this is a draft version of the software product. Significant changes in algorithms and interface are possible without maintaining backward compatibility.
- Within the framework of the current implementation, the result essentially depends on the choice of the initial approximation, but we have a number of ideas on how to fix this and, in general, how to significantly improve the accuracy/performance of the algorithm.
- If you are planning to use this software product to solve a specific problem, please contact the developers.


## Installation

The package (it requires the [Python](https://www.python.org) programming language of the version >= 3.6) can be installed via pip: `pip install ttopt`. It can be also downloaded from the repository [ttopt](https://github.com/SkoltechAI/ttopt) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages [numpy](https://numpy.org), [scipy](https://www.scipy.org) and [maxvolpy](https://bitbucket.org/muxas/maxvolpy/src/master/) will be automatically installed during the installation of the main software product. Note that in some cases it is better to install the `maxvolpy` package manually.


## Documentation

The documentation is located in the `doc` folder. To view the documentation, simply open the file `doc/_build/html/index.html` in a web browser.

> At the moment, this is a draft of documentation, however, there are already detailed comments on the use of functions and the meaning of their arguments. In the future, the documentation will be hosted on a separate web page.


## Examples

The demo-scripts with detailed comments are collected in the folder `demo`:
- `base.py` - we find the minimum for the 10-dimensional Rosenbrock function with vectorized input;
- `qtt.py` - we do almost the same as in the `base.py` script, but use the QTT-based approach (note that results are much more better then in the `base.py` example);
- `qtt_100d.py` - we do almost the same as in the `qtt.py` script, but approximate the 100-dimensional Rosenbrock function;
- `vect.py` - we find the minimum for the simple analytic function with "simple input" (the function is not vectorized);
- `cache.py` - we find the minimum for the simple analytic function to demonstrate the usage of cache;
- `tensor.py` - in this example we find the minimum for the multidimensional array/tensor (i.e., discrete function).


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov) (a.chertkov@skoltech.ru);
- [Ivan Oseledets](https://github.com/oseledets) (i.oseledets@skoltech.ru);
- [Roman Schutski](https://github.com/Qbit-) (r.schutski@skoltech.ru);
- [Konstantin Sozykin](https://github.com/gogolgrind) (konstantin.sozykin@skoltech.ru).
