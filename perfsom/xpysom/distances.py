import numpy as np
try:
    import cupy as cp
    default_xp = cp
    _cupy_available = True
except:
    print("WARNING: CuPy could not be imported")
    default_xp = np
    _cupy_available = False

def euclidean_squared_distance_part(x, w, w_flat_sq=None, xp=default_xp):
    """Calculate partial squared L2 distance

    This function does not sum x**2 to the result since it's not needed to 
    compute the best matching unit (it's not dependent on the neuron but
    it's a constant addition on the row).

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = xp.power(w_flat, 2).sum(axis=1, keepdims=True)
    cross_term = xp.dot(x, w_flat.T)
    return -2 * cross_term + w_flat_sq.T

def euclidean_squared_distance(x, w, w_flat_sq=None, xp=default_xp):
    """Calculate squared L2 distance

    NB: result shape is (N,X*Y)
    """
    x_sq = xp.power(x, 2).sum(axis=1, keepdims=True)
    return euclidean_squared_distance_part(x, w, w_flat_sq, xp) + x_sq

def euclidean_distance(x, w, w_flat_sq=None, xp=default_xp):
    """Calculate L2 distance

    NB: result shape is (N,X*Y)
    """
    return xp.nan_to_num(
        xp.sqrt(
            euclidean_squared_distance(x, w, w_flat_sq, xp)
        )
    )

def cosine_distance(x, w, w_flat_sq=None, xp=default_xp):
    """Calculate cosine distance

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = xp.power(w_flat, 2).sum(axis=1, keepdims=True)

    x_sq = xp.power(x, 2).sum(axis=1, keepdims=True)

    num = xp.dot(x, w_flat.T)
    denum = xp.sqrt(x_sq * w_flat_sq.T)
    similarity = xp.nan_to_num(num/denum)

    return 1 - similarity

if _cupy_available:
    _manhattan_distance_kernel = cp.ReductionKernel(
        'T x, T w',
        'T y',
        'abs(x-w)',
        'a+b',
        'y = a',
        '0',
        'l1norm'
    )

def manhattan_distance(x, w, xp=default_xp):
    """Calculate Manhattan distance

    It is very slow (~10x) compared to euclidean distance
    TODO: improve performance. Maybe a custom kernel is necessary

    NB: result shape is (N,X*Y)
    """

    if xp.__name__ == 'cupy':
        d = _manhattan_distance_kernel(
            x[:,xp.newaxis,xp.newaxis,:], 
            w[xp.newaxis,:,:,:], 
            axis=3
        )
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
    else:
        d = xp.linalg.norm(
            x[:,xp.newaxis,xp.newaxis,:]-w[xp.newaxis,:,:,:], 
            ord=1,
            axis=3,
        )
        return d.reshape(x.shape[0], w.shape[0]*w.shape[1])
