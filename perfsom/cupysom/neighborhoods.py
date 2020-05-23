import numpy as np
import cupy as cp

def prepare_neig_func(func, *first_args):
    def _inner(*args):
        return func(*first_args, *args)
    return _inner

def gaussian_rect(neigx, neigy, std_coeff, c, sigma):
    """Returns a Gaussian centered in c on a rect topology

    This function is optimized wrt the generic one.
    """
    d = 2*std_coeff**2*sigma**2

    nx = neigx[cp.newaxis,:]
    ny = neigy[cp.newaxis,:]
    cx = c[0][:,cp.newaxis]
    cy = c[1][:,cp.newaxis]

    ax = cp.exp(-cp.power(nx-cx, 2, dtype=cp.float32)/d)
    ay = cp.exp(-cp.power(ny-cy, 2, dtype=cp.float32)/d)
    return ax[:,:,cp.newaxis]*ay[:,cp.newaxis,:]

def gaussian_generic(xx, yy, std_coeff, c, sigma):
    """Returns a Gaussian centered in c on any topology
    
    TODO: this function is much slower than the _rect one
    """
    d = 2*std_coeff**2*sigma**2

    nx = xx[cp.newaxis,:,:]
    ny = yy[cp.newaxis,:,:]
    cx = xx.T[c][:, cp.newaxis, cp.newaxis]
    cy = yy.T[c][:, cp.newaxis, cp.newaxis]

    ax = cp.exp(-cp.power(nx-cx, 2, dtype=cp.float32)/d)
    ay = cp.exp(-cp.power(ny-cy, 2, dtype=cp.float32)/d)
    return (ax*ay).transpose((0,2,1))

def mexican_hat_rect(neigx, neigy, std_coeff, c, sigma):
    """Mexican hat centered in c (only rect topology)"""
    d = 2*std_coeff**2*sigma**2

    nx = neigx[cp.newaxis,:]
    ny = neigy[cp.newaxis,:]
    cx = c[0][:,cp.newaxis]
    cy = c[1][:,cp.newaxis]

    px = cp.power(nx-cx, 2, dtype=cp.float32)
    py = cp.power(ny-cy, 2, dtype=cp.float32)
    p = px[:,:,cp.newaxis] + py[:,cp.newaxis,:]
    
    return cp.exp(-p/d)*(1-2/d*p)

def mexican_hat_generic(xx, yy, std_coeff, c, sigma):
    """Mexican hat centered in c on any topology
    
    TODO: this function is much slower than the _rect one
    """
    d = 2*std_coeff**2*sigma**2

    nx = xx[cp.newaxis,:,:]
    ny = yy[cp.newaxis,:,:]
    cx = xx.T[c][:, cp.newaxis, cp.newaxis]
    cy = yy.T[c][:, cp.newaxis, cp.newaxis]

    px = cp.power(nx-cx, 2, dtype=cp.float32)
    py = cp.power(ny-cy, 2, dtype=cp.float32)
    p = px + py
    
    return (cp.exp(-p/d)*(1-2/d*p)).transpose((0,2,1))

def bubble(neigx, neigy, c, sigma):
    """Constant function centered in c with spread sigma.
    sigma should be an odd value.
    """
    nx = neigx[cp.newaxis,:]
    ny = neigy[cp.newaxis,:]
    cx = c[0][:,cp.newaxis]
    cy = c[1][:,cp.newaxis]

    ax = cp.logical_and(nx > cx-sigma,
                        nx < cx+sigma)
    ay = cp.logical_and(ny > cy-sigma,
                        ny < cy+sigma)
    return (ax[:,:,cp.newaxis]*ay[:,cp.newaxis,:]).astype(cp.float32)

def triangle(neigx, neigy, c, sigma):
    """Triangular function centered in c with spread sigma."""
    nx = neigx[cp.newaxis,:]
    ny = neigy[cp.newaxis,:]
    cx = c[0][:,cp.newaxis]
    cy = c[1][:,cp.newaxis]

    triangle_x = (-cp.abs(cx - nx)) + sigma
    triangle_y = (-cp.abs(cy - ny)) + sigma
    triangle_x[triangle_x < 0] = 0.
    triangle_y[triangle_y < 0] = 0.
    return triangle_x[:,:,cp.newaxis]*triangle_y[:,cp.newaxis,:]
