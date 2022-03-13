import numpy as np


class Opt():
    """Base class for minimizer.

    Note:
        Concrete minimizers should extend this class.

    """
    name = 'Base minimizer'

    def __init__(self, func, verb=True):
        self._f0 = func.calc     # Scalar function
        self._f = func.comp      # Vector function
        self.d = func.d          # Dimension
        self.a = func.a          # Grid lower limit
        self.b = func.b          # Grid upper limit
        self.x_real = func.x_min # Real x (arg) for minimum (for error check)
        self.y_real = func.y_min # Real y (f(x)) for minimum (for error check)
        self.verb = verb         # Verbosity of output (True/False)

        self.prep()
        self.init()

    @property
    def e_x(self):
        if self.x_real is None or self.x is None:
            return None
        return np.linalg.norm(self.x_real - self.x)

    @property
    def e_y(self):
        if self.y_real is None or self.y is None:
            return None
        return abs(self.y_real - self.y)

    @property
    def e_x_list(self):
        if self.x_real is None:
            return []
        return [np.linalg.norm(self.x_real - x) for x in self.x_list]

    @property
    def e_y_list(self):
        if self.y_real is None:
            return []
        return [abs(self.y_real - y) for y in self.y_list]

    def info(self, text_spec=''):
        text = ''
        text += f'{self.name}' + ' '*(12 - len(self.name)) + ' | '

        text += f't={self.t:-6.1f}'

        #if self.e_x is not None:
        #    text += f' | ex={self.e_x:-7.1e}'

        if self.e_y is not None:
            text += f' | ey={self.e_y:-7.1e}'

        text += f' | evals={self.m:-7.1e}'

        if text_spec:
            text += f' | {text_spec}'

        return text

    def init(self):
        self.t = 0.              # Work time (sec)
        self.m = 0               # Number of function calls
        self.x = None            # Found x (arg) for minimum
        self.y = None            # Found y (f(x)) for minimum
        self.x_list = []         # Values of x while iterations
        self.y_list = []         # Values of y while iterations
        self.m_list = []         # Numbers of requested points related to y_list
        self.c_list = []         # Numbers of requested cache points

        return self

    def f0(self, x):
        self.m += 1
        return self._f0(x)

    def f0_max(self, x):
        return -self.f0(x)

    def f0_scipy(self, x):
        y = self.f0(x)
        self.x_list.append(x)
        self.y_list.append(y)
        self.m_list.append(1)
        return y

    def f(self, X):
        self.m += X.shape[0]
        return self._f(X)

    def f_max(self, X):
        return -self.f(X)

    def prep(self):
        return self

    def run_estool(self, solver):
        for j in range(self.iters):
            solutions = solver.ask()

            fitness_list = np.zeros(solver.popsize)
            for i in range(solver.popsize):
                fitness_list[i] = self.f_max(solutions[i].reshape(1, -1))[0]

            solver.tell(fitness_list)
            result = solver.result()

            self.x_list.append(result[0])
            self.y_list.append(result[1])
            self.m_list.append(solver.popsize)

            if self.verb and (j+1) % 100 == 0:
                print("Fitness at iteration", (j+1), result[1])

        self.x = self.x_list[-1]
        self.y = self.y_list[-1]

    def solve(self):
        raise NotImplementedError()

    def to_dict(self):
        return {
            'name': self.name,
            'a': self.a,
            'b': self.b,
            't': self.t,
            'm': self.m,
            'e_x': self.e_x,
            'e_y': self.e_y,
            'e_x_list': self.e_x_list,
            'e_y_list': self.e_y_list,
            'm_list': self.m_list,
            'c_list': self.c_list,
        }
