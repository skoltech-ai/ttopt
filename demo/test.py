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


if __name__ == '__main__':
    run()
