import numpy as np
try:
    import cupy as cp
    default_xp = cp
    _cupy_available = True
except:
    print("WARNING: CuPy could not be imported")
    default_xp = np
    _cupy_available = False

def euclidean_squared_distance_part(x, w, w_sq=None, xp=default_xp):
    """Calculate partial squared L2 distance

    This function does not sum x**2 to the result since it's not needed to
    compute the best matching unit (it's not dependent on the neuron but
    it's a constant addition on the row).

    NB: result shape is (N,X*Y)
    """
    if w_sq is None:
        w_sq = xp.power(w, 2).sum(axis=1, keepdims=True)
    cross_term = xp.dot(x, w.T)
    return -2 * cross_term + w_sq.T

def euclidean_squared_distance(x, w, w_sq=None, xp=default_xp):
    """Calculate squared L2 distance

    NB: result shape is (N,X*Y)
    """
    x_sq = xp.power(x, 2).sum(axis=1, keepdims=True)
    return euclidean_squared_distance_part(x, w, w_sq, xp) + x_sq

def euclidean_distance(x, w, w_sq=None, xp=default_xp):
    """Calculate L2 distance

    NB: result shape is (N,X*Y)
    """
    return xp.nan_to_num(
        xp.sqrt(
            euclidean_squared_distance(x, w, w_sq, xp)
        )
    )

def cosine_distance(x, w, w_sq=None, xp=default_xp):
    """Calculate cosine distance

    NB: result shape is (N,X*Y)
    """
    if w_sq is None:
        w_sq = xp.power(w, 2).sum(axis=1, keepdims=True)

    x_sq = xp.power(x, 2).sum(axis=1, keepdims=True)

    num = xp.dot(x, w.T)
    denum = xp.sqrt(x_sq * w_sq.T)
    similarity = xp.nan_to_num(num/denum)

    return 1 - similarity

def norm_p_power_distance_generic(x, w, p=2, xp=default_xp):
    """Calculate norm-p distance raised to the power of p

    This is just the summed differences raised to the power of p.
    NB: p-th root is not calculated since it doesn't influence order.

    NB: result shape is (N,X*Y)
    """
    return xp.sum(
        xp.power(
            xp.abs(x[:,xp.newaxis,:] - w[xp.newaxis,:,:]),
            p
        ),
        axis=2
    )

def norm_p_power_distance_even(x, w,p=2, xp=default_xp):
    """Calculate norm-p distance raised to the power of p for even p

    This is an optimization of norm_p_power_distance_generic when p is even

    NB: result shape is (N,X*Y)
    """
    if p % 2 != 0:
        raise ValueError("p must be even")

    acc = xp.zeros((len(x), len(w)))
    k = 1
    for e in range(p+1):
        acc += (
            (-1 if e % 2 == 1 else 1) * k *
            xp.dot(x**(p-e), (w**e).T)
        )
        # next binomial coefficient
        k = (k * (p - e)) // (e + 1)
    return acc

def norm_p_power_distance(x, w, p=2, xp=default_xp):
    """Calculate norm-p distance raised to the power of p
    This function chooses the fastest implementation depending on p

    NB: result shape is (N,X*Y)
    """
    if p % 2 == 0:
        return norm_p_power_distance_even(x, w, p, xp)
    else:
        return norm_p_power_distance_generic(x, w, p, xp)

if _cupy_available:
    _manhattan_distance_kernel = cp.ReductionKernel(
        'T x, T w', # input params
        'T y',      # output params
        'abs(x-w)', # map
        'a+b',      # reduce
        'y = a',    # post=reduction map
        '0',        # identity value
        'l1norm'    # kernel name
    )

def manhattan_distance_cuda(x, w, xp=default_xp):
    """Calculate Manhattan distance

    Uses a custom CUDA reduction kernel to improve speed 3x.

    NB: result shape is (N,X*Y)
    """

    if xp.__name__ != 'cupy':
        raise ValueError("This function only works with cupy")

    return _manhattan_distance_kernel(
        x[:,xp.newaxis,:],
        w[xp.newaxis,:,:],
        axis=2
    )

def manhattan_distance_no_opt(x, w, xp=default_xp):
    """Calculate Manhattan distance

    It is very slow (~10x) compared to euclidean distance

    NB: result shape is (N,X*Y)
    """
    return norm_p_power_distance_generic(x, w, p=1, xp=xp)


def manhattan_distance(x, w, xp=default_xp):
    """Calculate Manhattan distance

    Uses the improved CUDA version if using GPU

    NB: result shape is (N,X*Y)
    """

    if xp.__name__ == 'cupy':
        return manhattan_distance_cuda(x, w, xp=xp)
    else:
        return manhattan_distance_no_opt(x, w, xp=xp)

class DistanceFunction:
    def __init__(self, name, kwargs, xp):
        distance_functions = {
            'euclidean': euclidean_squared_distance_part,
            'euclidean_no_opt': euclidean_squared_distance,
            'manhattan': manhattan_distance,
            'manhattan_no_opt': manhattan_distance_no_opt,
            'cosine': cosine_distance,
            'norm_p': norm_p_power_distance,
            'norm_p_no_opt': norm_p_power_distance_generic,
        }

        if name not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (name,
                                    ', '.join(distance_functions.keys())))

        self.__kwargs = {'xp': xp, **kwargs}
        self.__distance_function = distance_functions[name]
        self.can_cache = name in [
            'euclidean',
            'euclidean_no_opt',
            'cosine',
        ]

    def __call__(self, x, w, w_flat_sq=None):
        w_flat = w.reshape(-1, w.shape[2])
        if w_flat_sq is not None:
            return self.__distance_function(x, w_flat, **self.__kwargs)
        else:
            return self.__distance_function(x, w_flat, w_flat_sq, **self.__kwargs)
