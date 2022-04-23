import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_init


NAMES = ['Rastrigin'] # ['Ackley', 'Rastrigin', 'Rosenbrock', 'Schwefel']


def run(p=2, q=26, rmax=4, reps=1):
    for d in [10, 25, 50, 100, 250, 500]:
        evals= d * 1.E+4

        for func in teneva.func_demo_all(d=d, names=NAMES):
            e_list, t_list, t_func_list = [], [], []

            for i in range(reps):
                np.random.seed(i)

                name = func.name + f'-{d}D'
                name = name + ' ' * (14 - len(name))

                tto = TTOpt(f=func.get_f_poi, d=func.d, a=func.a, b=func.b,
                    name=name, x_min_real=func.x_min, y_min_real=func.y_min,
                    p=p, q=q, evals=evals, use_old=True, with_cache=False)
                tto.minimize(rmax=rmax)#, fs_opt=None)

                e_list.append(tto.e_y)
                t_list.append(tto.t_minim)
                t_func_list.append(tto.t_total)

            e = np.mean(e_list)
            t = np.mean(t_list)
            t_func = np.mean(t_func_list)
            text = f'{name} | '
            text += f'err: {e:-7.1e} | '
            text += f'evals: {evals:-7.1e} | '
            text += f't: {t:-7.4f} | '
            text += f't_part_alg: {(t-t_func)/t:-7.2f}'
            print(text)


if __name__ == '__main__':
    run()
