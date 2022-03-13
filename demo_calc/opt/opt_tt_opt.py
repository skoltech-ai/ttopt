import sys
from time import perf_counter as tpc


from opt import Opt


from ttopt import TTOpt


class OptTTOpt(Opt):
    """Minimizer based on the TTOpt."""
    name = 'TTOpt'

    def info(self):
        if self.n is not None:
            text = f'n={self.n:-5d}'
        else:
            text = f'p={self.p:-1d}, q={self.q:-2d}'
        text += f', r={self.r:-3d}'
        return super().info(text)

    def prep(self, n=None, p=2, q=12, r=6, evals=1.E+7):
        self.n = n
        self.p = p
        self.q = q
        self.r = r
        self.evals = int(evals)

        return self

    def solve(self):
        t = tpc()

        tto = TTOpt(
            self.f,
            d=self.d,
            a=self.a,
            b=self.b,
            n=self.n,
            p=self.p,
            q=self.q,
            evals=self.evals,
            with_log=self.verb,
            with_cache=False)
        tto.minimize(self.r)

        self.t = tpc() - t

        self.x = tto.x_min
        self.y = tto.y_min

        self.x_list = tto.x_min_list
        self.y_list = tto.y_min_list
        self.m_list = tto.evals_min_list
        self.c_list = tto.cache_min_list

    def to_dict(self):
        res = super().to_dict()
        res['n'] = self.n
        res['p'] = self.p
        res['q'] = self.q
        res['r'] = self.r
        res['evals'] = self.evals
        return res
