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

def manhattan_distance(x, w, xp=default_xp):
    """Calculate Manhattan distance

    It is very slow (~10x) compared to euclidean distance
    TODO: improve performance. Maybe a custom kernel is necessary

    NB: result shape is (N,X*Y)
    """

    if xp.__name__ == 'cupy':
        return _manhattan_distance_kernel(
            x[:,xp.newaxis,:], 
            w[xp.newaxis,:,:], 
            axis=2
        )
    else:
        return xp.linalg.norm(
            x[:,xp.newaxis,:]-w[xp.newaxis,:,:], 
            ord=1,
            axis=2,
        )
        )

class DistanceFunction:
    def __init__(self, name, xp):
        distance_functions = {
            'euclidean': euclidean_squared_distance_part,
            'euclidean_no_opt': euclidean_squared_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance,
        }

        if name not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (name,
                                    ', '.join(distance_functions.keys())))
        
        self.__kwargs = {'xp': xp}
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
