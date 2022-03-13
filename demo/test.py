import numpy as np
import teneva


from ttopt import TTOpt
from ttopt import ttopt_init


np.random.seed(16333)


for func in teneva.func_demo_all(d=10):
    name = func.name + ' ' * (15 - len(func.name))
    tto = TTOpt(f=func.get_f, d=func.d, a=func.a, b=func.b,
        name=name, x_min_real=func.x_min, y_min_real=func.y_min,
        p=2, q=20, evals=1.E+6)
    tto.minimize(rmax=8)
    print(tto.info(with_e_x=False))
