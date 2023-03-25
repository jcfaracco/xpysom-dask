import numpy as np
import argparse
import xpysom.neighborhoods as neigs
from bench_utils import bench, generate_randi

try:
    import cupy as cp
except Exception as e:
    cp = None
    print("Can't import cupy", e)

def init_neig(x, y, topology, xp):
    neigx = xp.arange(x)
    neigy = xp.arange(y)
    xx, yy = xp.meshgrid(neigx, neigy)
    xx = xx.astype(float)
    yy = yy.astype(float)

    if topology == 'hexagonal':
        xx[::-2] -= 0.5

    return neigx, neigy, xx, yy

def get_neig_func(f, neigx, neigy, std_coef, compact_support, sigma, xp):
    def _inner(cx, cy):
        args = [neigx, neigy]
        if f not in [neigs.bubble, neigs.triangle]:
            args.append(std_coef)
        if f not in [neigs.bubble]:
            args.append(compact_support)
        args += [(cx, cy), sigma, xp]
        return f(*args)
    return _inner

def use_neig(f):
    return f in [
        neigs.gaussian_rect,
        neigs.mexican_hat_rect,
        neigs.bubble,
        neigs.triangle,
    ]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'NeighborhoodBenchmarker',
                    description = 'Benchmark neighborhood algorithms')
    parser.add_argument('neighborhood', help="Function name to test")
    parser.add_argument('-x', '--xp', choices=['numpy', 'cupy'], default='numpy', help="Executor")
    parser.add_argument('-s', '--size', default=64, type=int, help="Grid size (both x and y)")
    parser.add_argument('-n', default=10000, type=int, help="Number of samples")
    parser.add_argument('-t', '--topology', default='rectangular', choices=['rectangular', 'hexagonal'], help="Topology")
    parser.add_argument('-c', '--compact', action='store_true', default=False, help="compact support")
    parser.add_argument('--std', default=0.5, type=float, help="standard coefficient")
    parser.add_argument('--sigma', default=1, type=float, help="sigma")
    parser.add_argument('-p', default=None, type=int, help="Norm-p p parameter")
    parser.add_argument('-r', default=10, type=int, help="Number of repetitions")
    parser.add_argument('--warmup', default=1, type=int, help="Number of warm-up repetitions")
    parser.add_argument('-d', action='store_true', default=False, help="dump output")
    args = parser.parse_args()

    neig = neigs.__dict__[args.neighborhood]
    xp = cp if args.xp == 'cupy' else np


    neigx, neigy, xx, yy = init_neig(args.size, args.size, args.topology, xp)
    neig_f = get_neig_func(neig,
        neigx if use_neig(neig) else xx,
        neigy if use_neig(neig) else yy,
        args.std, args.compact, args.sigma, xp)
    bench(
        neig_f,
        args.n,
        args.n,
        generator=lambda *fargs, **fkwargs: generate_randi(*fargs, **fkwargs, low=0, high=args.size),
        repeat=args.r,
        warmup=args.warmup,
        name=args.neighborhood,
        xp=xp,
        dump=args.d
    )

