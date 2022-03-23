"""Test script.

Expected output:
"
Ackley         -10d | k=1.00e+05 | t_all=1.21e+00 | e_y=3.91e-06
Brown          -10d | k=1.00e+05 | t_all=1.27e+00 | e_y=4.65e-02
Grienwank      -10d | k=1.00e+05 | t_all=1.33e+00 | e_y=1.69e-04
Michalewicz    -10d | k=1.00e+05 | t_all=1.33e+00 | e_y=8.67e-02
Rastrigin      -10d | k=1.00e+05 | t_all=1.29e+00 | e_y=4.62e-11
Rosenbrock     -10d | k=1.00e+05 | t_all=1.32e+00 | e_y=1.19e+00
Schaffer       -10d | k=1.00e+05 | t_all=1.29e+00 | e_y=1.87e-01
Schwefel       -10d | k=1.00e+05 | t_all=1.29e+00 | e_y=1.09e-01
"

"""
import numpy as np
import teneva


from ttopt import TTOpt


np.random.seed(16333)


def run(d=10, p=2, q=25, evals=1.E+5, rmax=4):
    for func in teneva.func_demo_all(d=d):
        name = func.name + ' ' * (15 - len(func.name))
        tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
            name=name, x_min_real=func.x_min, y_min_real=func.y_min,
            p=p, q=q, evals=evals)
        tto.minimize(rmax=rmax)
        print(tto.info(with_e_x=False))


def run_rep(d=10, p=2, q=25, evals=1.E+5, rmax=4):
    for func in teneva.func_demo_all(d=d):
        e_list = []
        for i in range(10):
            name = func.name + ' ' * (15 - len(func.name))
            tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                p=p, q=q, evals=evals)
            tto.minimize(rmax=rmax)
            e_list.append(tto.e_y)

        text = f'{name} | '
        text += f'e_avg: {np.mean(e_list):-7.1e} | '
        text += f'e_min: {np.min(e_list):-7.1e} | '
        text += f'e_max: {np.max(e_list):-7.1e} | '
        print(text)


def run_many(d=10, p=2, q=25, evals=1.E+5, rmax=4):
    for func in teneva.func_demo_all(d=d):
        lim = func.b[0] - func.a[0]
        for fs_opt in [None, 1000., 100., 10., 1., 0.1, 0.01]:
            for i in range(5):
                name = func.name + ' ' * (15 - len(func.name))
                tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                    name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                    p=p, q=q, evals=evals)
                tto.minimize(rmax=rmax, fs_opt=fs_opt)
                print(tto.info(with_e_x=False) + f' | {lim:-6.2f} | opt = {fs_opt}')
            print('')
        print('\n')


if __name__ == '__main__':
    run_rep()
