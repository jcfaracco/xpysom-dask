import numpy as np
import cupy as cp

def euclidean_squared_distance_part(x, w, w_flat_sq=None):
    """Calculate partial squared L2 distance

    This function does not sum x**2 to the result since it's not needed to 
    compute the best matching unit (it's not dependent on the neuron but
    it's a constant addition on the row).

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = cp.power(w_flat, 2).sum(axis=1, keepdims=True)
    cross_term = cp.dot(x, w_flat.T)
    return -2 * cross_term + w_flat_sq.T

def euclidean_squared_distance(x, w, w_flat_sq=None):
    """Calculate squared L2 distance

    NB: result shape is (N,X*Y)
    """
    x_sq = cp.power(x, 2).sum(axis=1, keepdims=True)
    return euclidean_squared_distance_part(x, w, w_flat_sq) + x_sq

def euclidean_distance(x, w, w_flat_sq=None):
    """Calculate L2 distance

    NB: result shape is (N,X*Y)
    """
    return cp.nan_to_num(
        cp.sqrt(
            euclidean_squared_distance(x, w, w_flat_sq)
        )
    )

def cosine_distance(x, w, w_flat_sq=None):
    """Calculate cosine distance

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    if w_flat_sq is None:
        w_flat_sq = cp.power(w_flat, 2).sum(axis=1, keepdims=True)

    x_sq = cp.power(x, 2).sum(axis=1, keepdims=True)

    num = cp.dot(x, w_flat.T)
    denum = cp.sqrt(x_sq * w_flat_sq.T)
    similarity = cp.nan_to_num(num/denum)

    return 1 - similarity

_manhattan_distance_kernel = cp.ReductionKernel(
    'T x, T w',
    'T y',
    'abs(x-w)',
    'a+b',
    'y = a',
    '0',
    'l1norm'
)
def manhattan_distance(x, w):
    """Calculate Manhattan distance

    It is very slow (~10x) compared to euclidean distance
    TODO: improve performance. Maybe a custom kernel is necessary

    NB: result shape is (N,X*Y)
    """
    d = _manhattan_distance_kernel(
        x[:,np.newaxis,np.newaxis,:], 
        w[np.newaxis,:,:,:], 
        axis=3
    )
    return d.reshape(x.shape[0], w.shape[0]*w.shape[1])