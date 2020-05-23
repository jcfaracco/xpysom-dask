import numpy as np
import cupy as cp

def euclidean_distance(x, w):
    """Calculate L2 distance

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    x_sq = cp.power(x, 2).sum(axis=1, keepdims=True)
    w_flat_sq = cp.power(w_flat, 2).sum(axis=1, keepdims=True)
    cross_term = cp.dot(x, w_flat.T)
    sq_dist = -2 * cross_term + x_sq + w_flat_sq.T
    dist = cp.nan_to_num(cp.sqrt(sq_dist))
    return dist


def euclidean_squared_distance(x, w):
    """Calculate squared L2 distance

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    x_sq = cp.power(x, 2).sum(axis=1, keepdims=True)
    w_flat_sq = cp.power(w_flat, 2).sum(axis=1, keepdims=True)
    cross_term = cp.dot(x, w_flat.T)
    return -2 * cross_term + x_sq + w_flat_sq.T

def cosine_distance(x, w):
    """Calculate cosine distance

    NB: result shape is (N,X*Y)
    """
    w_flat = w.reshape(-1, w.shape[2])
    x_sq = cp.power(x, 2).sum(axis=1, keepdims=True)
    w_flat_sq = cp.power(w_flat, 2).sum(axis=1, keepdims=True)

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