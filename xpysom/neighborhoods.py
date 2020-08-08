import numpy as np
try:
    import cupy as cp
    default_xp = cp
except:
    print("WARNING: CuPy could not be imported")
    default_xp = np

def prepare_neig_func(func, *first_args):
    def _inner(*args, **kwargs):
        return func(*first_args, *args, **kwargs)
    return _inner

def gaussian_rect(neigx, neigy, std_coeff, compact_support, c, sigma, xp=default_xp):
    """Returns a Gaussian centered in c on a rect topology

    This function is optimized wrt the generic one.
    """
    d = 2*std_coeff**2*sigma**2

    nx = neigx[xp.newaxis,:]
    ny = neigy[xp.newaxis,:]
    cx = c[0][:,xp.newaxis]
    cy = c[1][:,xp.newaxis]

    ax = xp.exp(-xp.power(nx-cx, 2, dtype=xp.float32)/d)
    ay = xp.exp(-xp.power(ny-cy, 2, dtype=xp.float32)/d)

    if compact_support:
        ax *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        ay *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)

    return ax[:,:,xp.newaxis]*ay[:,xp.newaxis,:]

def gaussian_generic(xx, yy, std_coeff, compact_support, c, sigma, xp=default_xp):
    """Returns a Gaussian centered in c on any topology
    
    TODO: this function is much slower than the _rect one
    """
    d = 2*std_coeff**2*sigma**2

    nx = xx[xp.newaxis,:,:]
    ny = yy[xp.newaxis,:,:]

    cx = xx.T[c][:, xp.newaxis, xp.newaxis]
    cy = yy.T[c][:, xp.newaxis, xp.newaxis]

    ax = xp.exp(-xp.power(nx-cx, 2, dtype=xp.float32)/d)
    ay = xp.exp(-xp.power(ny-cy, 2, dtype=xp.float32)/d)

    if compact_support:
        ax *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        ay *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)

    return (ax*ay).transpose((0,2,1))

def mexican_hat_rect(neigx, neigy, std_coeff, compact_support, c, sigma, xp=default_xp):
    """Mexican hat centered in c (only rect topology)"""
    d = 2*std_coeff**2*sigma**2

    nx = neigx[xp.newaxis,:]
    ny = neigy[xp.newaxis,:]
    cx = c[0][:,xp.newaxis]
    cy = c[1][:,xp.newaxis]

    px = xp.power(nx-cx, 2, dtype=xp.float32)
    py = xp.power(ny-cy, 2, dtype=xp.float32)
    p = px[:,:,xp.newaxis] + py[:,xp.newaxis,:]
    
    if compact_support:
        ax *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        ay *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)

    return xp.exp(-p/d)*(1-2/d*p)

def mexican_hat_generic(xx, yy, std_coeff, compact_support, c, sigma, xp=default_xp):
    """Mexican hat centered in c on any topology
    
    TODO: this function is much slower than the _rect one
    """
    d = 2*std_coeff**2*sigma**2

    nx = xx[xp.newaxis,:,:]
    ny = yy[xp.newaxis,:,:]
    cx = xx.T[c][:, xp.newaxis, xp.newaxis]
    cy = yy.T[c][:, xp.newaxis, xp.newaxis]

    px = xp.power(nx-cx, 2, dtype=xp.float32)
    py = xp.power(ny-cy, 2, dtype=xp.float32)
    p = px + py
    
    return (xp.exp(-p/d)*(1-2/d*p)).transpose((0,2,1))

def bubble(neigx, neigy, c, sigma, xp=default_xp):
    """Constant function centered in c with spread sigma.
    sigma should be an odd value.
    """
    nx = neigx[xp.newaxis,:]
    ny = neigy[xp.newaxis,:]
    cx = c[0][:,xp.newaxis]
    cy = c[1][:,xp.newaxis]

    ax = xp.logical_and(nx > cx-sigma,
                        nx < cx+sigma)
    ay = xp.logical_and(ny > cy-sigma,
                        ny < cy+sigma)
    return (ax[:,:,xp.newaxis]*ay[:,xp.newaxis,:]).astype(xp.float32)

def triangle(neigx, neigy, compact_support, c, sigma, xp=default_xp):
    """Triangular function centered in c with spread sigma."""
    nx = neigx[xp.newaxis,:]
    ny = neigy[xp.newaxis,:]
    cx = c[0][:,xp.newaxis]
    cy = c[1][:,xp.newaxis]

    triangle_x = (-xp.abs(cx - nx)) + sigma
    triangle_y = (-xp.abs(cy - ny)) + sigma
    triangle_x[triangle_x < 0] = 0.
    triangle_y[triangle_y < 0] = 0.

    if compact_support:
        triangle_x *= xp.logical_and(nx > cx-sigma, nx < cx+sigma)
        triangle_y *= xp.logical_and(ny > cy-sigma, ny < cy+sigma)

    return triangle_x[:,:,xp.newaxis]*triangle_y[:,xp.newaxis,:]
