import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_animate
from ttopt import ttopt_init


def run(d=2, p=2, q=12, evals=1.E+4, rmax=4, with_cache=False):
    n = np.ones(d * q, dtype=int) * p
    Y0 = ttopt_init(n, rmax)

    for func in teneva.func_demo_all(d=d):
        name = func.name + ' ' * (15 - len(func.name))
        tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
            name=name, x_min_real=func.x_min, y_min_real=func.y_min,
            p=p, q=q, evals=evals, with_cache=with_cache, with_full_info=True)
        tto.minimize(rmax, Y0, fs_opt=1.)
        print(tto.info(with_e_x=False))

        fpath = f'./check/animation/{func.name}.gif'
        ttopt_animate(func, tto, fpath=fpath)


def run_base(d=2, n=256, evals=1.E+4, rmax=4, with_cache=False):
    n = [n] * d
    Y0 = ttopt_init(n, rmax)

    for func in teneva.func_demo_all(d=d):
        name = func.name + ' ' * (15 - len(func.name))
        tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
            name=name, x_min_real=func.x_min, y_min_real=func.y_min,
            n=n, evals=evals, with_cache=with_cache, with_full_info=True)
        tto.minimize(rmax, Y0, fs_opt=1.)
        print(tto.info(with_e_x=False))

        fpath = f'./check/animation/{func.name}_base.gif'
        ttopt_animate(func, tto, fpath=fpath)


if __name__ == '__main__':
    np.random.seed(16333)
    run()
    run_base()
