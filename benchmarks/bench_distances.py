import numpy as np
import argparse
import xpysom.distances as dists
from bench_utils import bench

try:
    import cupy as cp
except Exception as e:
    cp = None
    print("Can't import cupy", e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'DistanceBenchmarker',
                    description = 'Benchmark distance algorithms')
    parser.add_argument('distance', help="Function name to test")
    parser.add_argument('-x', '--xp', choices=['numpy', 'cupy'], default='numpy', help="Executor")
    parser.add_argument('-n', default=10000, type=int, help="Number of samples")
    parser.add_argument('-w', default=256, type=int, help="Number of neurons (grid size)")
    parser.add_argument('-m', default=100, type=int, help="Number of dimensions")
    parser.add_argument('-p', default=None, type=int, help="Norm-p p parameter")
    parser.add_argument('-r', default=10, type=int, help="Number of repetitions")
    parser.add_argument('--warmup', default=1, type=int, help="Number of warm-up repetitions")
    parser.add_argument('-d', action='store_true', default=False, help="dump output")
    args = parser.parse_args()

    dist = dists.__dict__[args.distance]
    xp = cp if args.xp == 'cupy' else np
    shape_x = (args.n, args.m)
    shape_w = (args.w, args.m)
    kwargs = {'p': args.p} if args.p else {}
    kwargs['xp'] = xp

    bench(dist, shape_x, shape_w, kwargs=kwargs, repeat=args.r, warmup=args.warmup, name=dist, xp=xp, dump=args.d)