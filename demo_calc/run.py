"""Investigating TTOpt performance for analytical benchmark functions."""
import argparse
import numpy as np
import pickle
import sys


import matplotlib.pyplot as plt
params = {
    'text.usetex' : False,
    'font.size' : 36,
    'legend.fancybox':True,
    'legend.loc' : 'best',
    'legend.framealpha': 0.9,
    "legend.fontsize" : 27}
plt.rcParams.update(params)


sys.path.append('./demo_calc/opt')
from opt_ga import OptGA
from opt_es import OptES
from opt_es_cma import OptESCMA
from opt_tt_opt import OptTTOpt


from demo_func import DemoFuncAckley
from demo_func import DemoFuncGrienwank
from demo_func import DemoFuncMichalewicz
from demo_func import DemoFuncRastrigin
from demo_func import DemoFuncRosenbrock
from demo_func import DemoFuncSchwefel


# Minimizer classes:
OPTS = [OptTTOpt, OptGA, OptES, OptESCMA]


# Minimizer names:
OPT_NAMES = {
    'GA': 'simpleGA',
    'ES-OpenAI': 'openES',
    'ES-CMA': 'cmaES',
    'TTOpt': 'TTOpt'}


# Function classes:
FUNCS = [
    DemoFuncAckley,
    DemoFuncGrienwank,
    DemoFuncMichalewicz,
    DemoFuncRastrigin,
    DemoFuncRosenbrock,
    DemoFuncSchwefel,
]


# Function names:
FUNC_NAMES = [
    'Ackley',
    'Grienwank',
    'Michalewicz',
    'Rastrigin',
    'Rosenbrock',
    'Schwefel']


# List of ranks to check dependency of TTOpt on rank:
R_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# List of grid sizes to check QTT-effect (should be power of 2 and 4):
N_LIST = [2**8, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20]


# List of numbers of function calls to check the related dependency:
EVALS_LIST = [1.E+4, 5.E+4, 1.E+5, 5.E+5, 1.E+6, 5.E+6, 1.E+7]


# Population size for genetic-based algorithms:
GA_POPSIZE = 255


def load(d, name, kind):
    fpath = f'./demo_calc/res_data/{name}_{kind}_{d}dim.pickle'
    try:
        with open(fpath, 'rb') as f:
            res = pickle.load(f)
    except Exception as e:
        res = None

    return res


def log(text, d, name, kind, is_init=False):
    fpath = f'./demo_calc/res_logs/{name}_{kind}_{d}dim.txt'
    print(text)
    with open(fpath, 'w' if is_init else 'a') as f:
        f.write(text + '\n')


def run_comp(d, p, q, r, evals, reps=1, name='calc1', with_log=False):
    """Compare different methods for benchmark analytic functions."""
    log(f'', d, name, 'comp', is_init=True)
    res = {}

    for func_class in FUNCS:
        func = func_class(d)
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'comp')

        res[func.name] = {}

        for opt_class in OPTS:
            opt = opt_class(func, verb=with_log)
            if opt.name == 'TTOpt':
                opt.prep(None, p, q, r, evals)
            else:
                opt.prep(popsize=GA_POPSIZE, iters=evals/GA_POPSIZE)

            res[func.name][opt.name] = solve(opt, d, name, 'comp', reps)
            save(res, d, name, 'comp')

        log('\n\n', d, name, 'comp')


def run_iter(d, p, q, r, reps=1, name='calc1', with_log=False):
    """Check dependency of TTOpt on evals for benchmark analytic functions."""
    log(f'', d, name, 'iter', is_init=True)
    res = {}
    for func_class in FUNCS:
        func = func_class(d)
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'iter')
        res[func.name] = []
        for evals in EVALS_LIST:
            opt = OptTTOpt(func, verb=with_log)
            opt.prep(None, p, q, r, evals)
            res[func.name].append(solve(opt, d, name, 'iter', reps))
            save(res, d, name, 'iter')
        log('\n\n', d, name, 'iter')


def run_quan(d, r, evals, reps=1, name='calc1', with_log=False):
    """Check effect of QTT-based approach for benchmark analytic functions."""
    log(f'', d, name, 'quan', is_init=True)
    res = {}

    for func_class in FUNCS:
        func = func_class(d)
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'quan')

        res[func.name] = {'q0': [], 'q2': [], 'q4': []}

        for n in N_LIST:
            n = int(n)
            q2 = int(np.log2(n))
            q4 = int(q2 / 2)

            if 2**q2 != n or 4**q4 != n:
                raise ValueError(f'Invalid grid size "{n}"')

            opt = OptTTOpt(func, verb=with_log)
            opt.prep(n, None, None, r, evals)
            res[func.name]['q0'].append(solve(opt, d, name, 'quan', reps))

            opt = OptTTOpt(func, verb=with_log)
            opt.prep(None, 2, q2, r, evals)
            res[func.name]['q2'].append(solve(opt, d, name, 'quan', reps))

            opt = OptTTOpt(func, verb=with_log)
            opt.prep(None, 4, q4, r, evals)
            res[func.name]['q4'].append(solve(opt, d, name, 'quan', reps))

        save(res, d, name, 'quan')
        log('\n\n', d, name, 'quan')


def run_rank(d, p, q, evals, reps=1, name='calc1', with_log=False):
    """Check dependency of TTOpt on rank for benchmark analytic functions."""
    log(f'', d, name, 'rank', is_init=True)
    res = {}
    for func_class in FUNCS:
        func = func_class(d)
        log(f'--- Minimize function {func.name}-{d}dim\n', d, name, 'rank')
        res[func.name] = []
        for r in R_LIST:
            opt = OptTTOpt(func, verb=with_log)
            opt.prep(None, p, q, r, evals)
            res[func.name].append(solve(opt, d, name, 'rank', reps))
            save(res, d, name, 'rank')
        log('\n\n', d, name, 'rank')


def run_show(d, name='calc1'):
    """Show results of the previous calculations."""
    log(f'', d, name, 'show', is_init=True)
    run_show_comp(d, name)
    run_show_iter(d, name)
    run_show_quan(d, name)
    run_show_rank(d, name)


def run_show_comp(d, name='calc1'):
    """Show results of the previous calculations for "comp"."""
    res = load(d, name, 'comp')
    if res is None:
        log('>>> Comp-result is not available\n\n', d, name, 'show')
        return

    text = '>>> Comp-result (part of latex table): \n\n'

    text += '% ------ AUTO CODE START\n\n'

    for name_opt, name_opt_text in OPT_NAMES.items():
        text += '\\multirow{2}{*}{' + name_opt_text + '}'

        text += '\n& $\\epsilon$ '
        for name_func in FUNC_NAMES:
            v = res[name_func][name_opt]['e']
            vals = [res[name_func][nm]['e']
                for nm in OPT_NAMES.keys() if nm != name_opt]
            if v <= np.min(vals):
                text += '& \\textbf{' + f'{v:-8.1e}' + '} '
            else:
                text += f'& {v:-8.1e} '

        text += '\n\\\\'

        text += '\n& $\\tau$ '
        for name_func in FUNC_NAMES:
            v = res[name_func][name_opt]['t']
            text += '& \\textit{' + f'{v:-6.2f}' + '} '

        text += ' \\\\ \\hline \n\n'

    text += '% ------ AUTO CODE END\n\n'

    log(text, d, name, 'show')


def run_show_iter(d, name='calc1'):
    """Show results of the previous calculations for "iter"."""
    res = load(d, name, 'iter')
    if res is None:
        log('>>> Iter-result is not available', d, name, 'show')
        return

    text = '>>> Iter-result (png file with plot): \n\n'

    plt.figure(figsize=(14, 8))
    plt.xlabel('number of queries')
    plt.ylabel('absolute error')

    for name_func in FUNC_NAMES:
        v = [item['e'] for item in res[name_func]]
        plt.plot(EVALS_LIST, v, label=name_func, marker='o')

    plt.grid()
    plt.semilogx()
    plt.semilogy()
    plt.legend(loc='best')

    fpath = f'./demo_calc/res_plot/{name}_iter_{d}dim.png'
    plt.savefig(fpath, bbox_inches='tight')
    text += f'Figure saved to file "{fpath}"\n\n'

    log(text, d, name, 'show')


def run_show_quan(d, name='calc1'):
    """Show results of the previous calculations for "quan"."""
    res = load(d, name, 'quan')
    if res is None:
        log('>>> Quan-result is not available', d, name, 'show')
        return

    text = '>>> Quan-result (part of latex table): \n\n'

    text += '% ------ AUTO CODE START\n'

    for i, n in enumerate(N_LIST):
        text += '\\multirow{2}{*}{' + str(n) + '}'

        text += '\n& TT '
        for name_func in FUNC_NAMES:
            v = res[name_func]['q0'][i]['e']
            text += f'& {v:-8.1e} '
        text += '\\\\'

        text += '\n& QTT '
        for name_func in FUNC_NAMES:
            v = res[name_func]['q2'][i]['e']
            text += f'& {v:-8.1e} '
        text += ' \\\\ \\hline \n'

    text += '% ------ AUTO CODE END\n\n'

    log(text, d, name, 'show')


    return


def run_show_rank(d, name='calc1'):
    """Show results of the previous calculations for "rank"."""
    res = load(d, name, 'rank')
    if res is None:
        log('>>> Rank-result is not available', d, name, 'show')
        return

    text = '>>> Rank-result (png file with plot): \n\n'

    plt.figure(figsize=(14, 8))
    plt.xlabel('rank')
    plt.ylabel('absolute error')
    plt.xticks(R_LIST)

    for name_func in FUNC_NAMES:
        v = [item['e'] for item in res[name_func]]
        plt.plot(R_LIST, v, label=name_func, marker='o')

    plt.grid()
    plt.semilogy()
    plt.legend(loc='best')

    fpath = f'./demo_calc/res_plot/{name}_rank_{d}dim.png'
    plt.savefig(fpath, bbox_inches='tight')
    text += f'Figure saved to file "{fpath}"\n\n'

    log(text, d, name, 'show')


def save(res, d, name, kind):
    fpath = f'./demo_calc/res_data/{name}_{kind}_{d}dim.pickle'
    with open(fpath, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def solve(opt, d, name, kind, reps=1):
    t, y, e, m = [], [], [], []
    for rep in range(reps):
        np.random.seed(rep)

        opt.init()
        opt.solve()

        log(opt.info(), d, name, kind)

        t.append(opt.t)
        y.append(opt.y)
        e.append(opt.e_y)
        m.append(opt.m)

    return {
        't': np.mean(t),
        'e': np.mean(e),
        'e_var': np.var(e),
        'e_min': np.min(e),
        'e_max': np.max(e),
        'e_avg': np.mean(e),
        'e_all': e,
        'y_all': y,
        'y_real': opt.y_real,
        'evals': int(np.mean(m))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Investigating TTOpt performance for analytic functions')
    parser.add_argument('-d', default=10,
        type=int, help='dimension d')
    parser.add_argument('-p', default=2,
        type=int, help='grid param p')
    parser.add_argument('-q', default=25,
        type=int, help='grid param q')
    parser.add_argument('-r', default=4,
        type=int, help='rank')
    parser.add_argument('--evals', default=1.E+5,
        type=float, help='evals')
    parser.add_argument('--reps', default=1,
        type=int, help='repetitions (only for TTOpt method)')
    parser.add_argument('--name', default='calc1',
        type=str, help='calculation name (the corresponding prefix will be used for the files with results)')
    parser.add_argument('--kind', default='comp',
        type=str, help='kind of calculation ("comp" - compare different solvers; "iter" - check dependency on number of calls for target function; "quan" - check effect of qtt-usage; "rank" - check dependency on rank; "show" - show results of the previous calculations)')
    parser.add_argument('--verb', default=False,
        type=bool, help='if True, then intermediate results of the optimization process will be printed to the console')

    args = parser.parse_args()

    if args.kind == 'comp' or args.kind == 'all':
        run_comp(args.d, args.p, args.q, args.r, args.evals, args.reps,
            args.name, args.verb)
    elif args.kind == 'iter':
        run_iter(args.d, args.p, args.q, args.r, args.reps,
            args.name, args.verb)
    elif args.kind == 'quan':
        run_quan(args.d, args.r, args.evals, args.reps,
            args.name, args.verb)
    elif args.kind == 'rank':
        run_rank(args.d, args.p, args.q, args.evals, args.reps,
            args.name, args.verb)
    elif args.kind == 'show':
        run_show(args.d, args.name)
    else:
        raise ValueError(f'Invalid kind of calculation "{args.kind}"')
