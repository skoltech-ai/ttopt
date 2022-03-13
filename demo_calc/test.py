"""Test for TTOpt."""
import numpy as np
import sys


sys.path.append('./demo_calc/opt')
from opt_tt_opt import OptTTOpt


from demo_func import DemoFunc


class FuncGabor2D(DemoFunc):
    def __init__(self, d=2, th=np.pi/3, lam=20., sig=10.):
        super().__init__(d, 'Gabor')

        if d != 2:
            raise ValueError('It works only for 2D case')

        self.par_th = th
        self.par_lam = lam
        self.par_sig = sig

        self.set_lim(0., +64.)
        self.set_min(None, None)

    def calc(self, x):
        cx = 0.5 * (self.b[0] - self.a[0])
        cy = 0.5 * (self.b[1] - self.a[1])

        x_th = +(x[0]-cx)*np.cos(self.par_th) + (x[1]-cy)*np.sin(self.par_th)
        y_th = -(x[0]-cx)*np.sin(self.par_th) + (x[1]-cy)*np.cos(self.par_th)

        y = np.exp(-0.5 * (x_th**2 + y_th**2) / self.par_sig**2)
        y *= np.cos(2 * np.pi / self.par_lam * x_th)

        return y

    def comp(self, X):
        return np.array([self.calc(x) for x in X])


class FuncGabor3D(DemoFunc):
    def __init__(self, d=3, th=0.5*np.pi, ph=0.5*np.pi, lam=20., sig=10.):
        super().__init__(d, 'Gabor')

        if d != 3:
            raise ValueError('It works only for 3D case')

        self.par_th = th
        self.par_ph = ph
        self.par_lam = lam
        self.par_sig = sig

        self.set_lim(0., +64.)
        self.set_min(None, None)

    def calc(self, x):
        cx = 0.5 * (self.b[0] - self.a[0])
        cy = 0.5 * (self.b[1] - self.a[1])
        cz = 0.5 * (self.b[2] - self.a[2])

        x_th = +(x[0]-cx)*np.cos(self.par_th) + (x[1]-cy)*np.sin(self.par_th)
        y_th = -(x[0]-cx)*np.sin(self.par_th) + (x[1]-cy)*np.cos(self.par_th)
        z_th = +x[2]

        x_th = x_th*np.cos(self.par_ph)-(z_th - cz)*np.sin(self.par_ph)
        y_th = y_th
        z_th = x_th*np.sin(self.par_ph)+(z_th - cz)*np.cos(self.par_ph)

        y = np.exp(-0.5 * (x_th**2 + y_th**2 + z_th**2) / self.par_sig**2)
        y *= np.cos(2 * np.pi / self.par_lam * x_th)

        return y

    def comp(self, X):
        return np.array([self.calc(x) for x in X])


def run2(d=2, p=2, q=12, r=4, evals=650):
    func = FuncGabor2D(d)
    # func.plot(fpath='demo_calc/tmp.png')
    print(f'--- Minimize function {func.name}-{d}dim\n')

    opt = OptTTOpt(func, verb=True)
    opt.prep(None, p, q, r, evals)
    opt.init()
    opt.solve()
    print(opt.info())
    print('Found x min  : ' , opt.x_list[-1])


def run(d=3, p=2, q=10, r=2, evals=1E+4):
    func = FuncGabor3D(d)
    # func.plot(fpath='demo_calc/tmp.png')
    print(f'--- Minimize function {func.name}-{d}dim\n')

    opt = OptTTOpt(func, verb=True)
    opt.prep(None, p, q, r, evals)
    opt.init()
    opt.solve()
    print(opt.info())
    print('Found x min  : ' , opt.x_list[-1])


if __name__ == '__main__':
    run()
