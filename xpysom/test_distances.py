import numpy as np

XPS = [np]
try:
    import cupy as cp
    XPS.append(cp)
except Exception as e:
    print("can't import cupy", e)

import pytest
from .distances import (
    euclidean_squared_distance_part,
    euclidean_squared_distance,
    euclidean_distance,
    cosine_distance,
    manhattan_distance,
    norm_p_power_distance,
)

def apply_distance(dist, x, y):
    z = np.empty((len(x), len(y)))
    for i, vx in enumerate(x):
        for j, vy in enumerate(y):
            z[i, j] = dist(np.array(vx), np.array(vy))
    return z

def int_to_binary_array(x, l):
    out = []
    p = 1
    n = 0
    while n < l:
        out.append(1 if p&x else 0)
        p *= 2
        n += 1
    return out

def get_inputs():
    inputs = []
    for l in range(1, 4):
        xys = []
        for xb in range(2**l):
            x = int_to_binary_array(xb, l)
            for yb in range(2**l):
                y = int_to_binary_array(yb, l)
                xys.append((x, y))

        # all combinations with 1 item per array
        inputs += [([x], [y]) for x, y in xys]

        # all combinations with the first x and any y
        inputs.append((
            [xys[0][0]],
            [y for _x, y in xys],
        ))

        # all combinations with the first y and any x
        inputs.append((
            [x for x, _y in xys],
            [xys[0][1]],
        ))

        # one big matrix with all prossible values
        inputs.append((
            [x for x, _y in xys],
            [y for _x, y in xys],
        ))

        # one big matrix with all prossible values but half ys
        inputs.append((
            [x for x, _y in xys],
            [y for _x, y in xys[::2]],
        ))

        # one big matrix with all prossible values but half xs
        inputs.append((
            [x for x, _y in xys[::2]],
            [y for _x, y in xys],
        ))

    # some fuzzy inputs
    np.random.seed(0)
    for n in (2, 7):
        for m in (3, 11):
            for l in (5, 13):
                x = np.random.rand(n, l).tolist()
                y = np.random.rand(m, l).tolist()
                inputs.append((x, y))
    return inputs

INPUTS = get_inputs()

DISTANCES = [
    (
        euclidean_squared_distance_part,
        lambda vx, vy: -2 * np.dot(vx, vy) + np.dot(vy,vy),
        {},
    ),
    (
        euclidean_squared_distance,
        lambda vx, vy: np.sum(np.power(vx-vy, 2)),
        {},
    ),
    (
        euclidean_distance,
        lambda vx, vy: np.linalg.norm(vx - vy),
        {},
    ),
    (
        cosine_distance,
        lambda vx, vy: 1 - np.nan_to_num(
            np.dot(vx, vy) / (np.linalg.norm(vx) * np.linalg.norm(vy))
        ),
        {},
    ),
    (
        manhattan_distance,
        lambda vx, vy: np.linalg.norm(vx-vy, ord=1),
        {},
    ),
    (
        norm_p_power_distance,
        lambda vx, vy: np.sum(np.power(vx-vy, 2)),
        {'p': 2},
    ),
    (
        norm_p_power_distance,
        lambda vx, vy: np.sum(np.power(np.abs(vx-vy), 3)),
        {'p': 3},
    ),
    (
        norm_p_power_distance,
        lambda vx, vy: np.sum(np.power(np.abs(vx-vy), 4)),
        {'p': 4},
    ),
]

@pytest.mark.parametrize('x,y', INPUTS)
@pytest.mark.parametrize('xp', XPS)
@pytest.mark.parametrize('dist,dist_simple,dist_kwargs', DISTANCES)
class TestDistances:
    def test_distance(self, xp, dist, dist_simple, dist_kwargs, x, y):
        x_a = xp.array(x)
        y_a = xp.array(y)

        z_out = dist(x_a, y_a, **dist_kwargs, xp=xp)
        if xp.__name__ == 'cupy':
            z_out = xp.asnumpy(z_out)

        z_expected = apply_distance(
            dist_simple,
            x, y
        )

        np.testing.assert_almost_equal(z_out, z_expected)
