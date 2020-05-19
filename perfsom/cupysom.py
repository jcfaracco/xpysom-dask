from math import sqrt, ceil
import unittest

from collections import defaultdict
from warnings import warn
import numpy as np
import cupy as cp

from perfsom.minisom import MiniSom, asymptotic_decay, fast_norm, print_progress

# In my GPU it looks like this is the best performance/memory trade-off
DEFAULT_CORE_OVERSUBSCRIPTION = 4

def find_cuda_cores():
    try:
        import subprocess
        return int(subprocess.check_output("nvidia-settings -q CUDACores -t", shell=True))
    except:
        print("Could not infer #cuda_cores")
        return 0

def euclidean_squared_distance(x, w):
    """Calculate L2 distance

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

manhattan_distance_kernel = cp.ReductionKernel(
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
    d = manhattan_distance_kernel(
        x[:,np.newaxis,np.newaxis,:], 
        w[np.newaxis,:,:,:], 
        axis=3
    )
    return d.reshape(x.shape[0], w.shape[0]*w.shape[1])

def ravel_idx_2d(idx, cols):
    return idx[0] * cols + idx[1]

def unravel_idx_2d(i, cols):
    return (i % cols, cp.floor_divide(i, cols))

class CupySom(MiniSom):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function='exponential', neighborhood_function='gaussian', topology='rectangular', activation_distance='euclidean', normalize_weights=False, random_seed=None, n_parallel=0):
        # passing some mock parameters to disable checks
        super().__init__(x, y, input_len, sigma=sigma, learning_rate=learning_rate, decay_function=decay_function, neighborhood_function='gaussian', topology=topology, activation_distance='euclidean', random_seed=random_seed)

        if n_parallel == 0:
            n_parallel = find_cuda_cores()*DEFAULT_CORE_OVERSUBSCRIPTION    
 
            if n_parallel == 0:
                raise ValueError("n_parallel was not specified and could not be infered from system")
        
        self._n_parallel = n_parallel

        if topology == 'rectangular':
            neig_functions = {
                'gaussian': self._gaussian_rect,
                'mexican_hat': self._mexican_hat_rect,
                'bubble': self._bubble,
                'triangle': self._triangle,
            }
        elif topology == 'hexagonal':
            neig_functions = {
                'gaussian': self._gaussian_generic,
                'mexican_hat': self._mexican_hat_generic,
                'bubble': self._bubble,
            }
        else:
            neig_functions = {}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {
            'euclidean': euclidean_squared_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance,
        }

        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]

        # print("Using n_parallel = " + str(self._n_parallel))
        self._unravel_precomputed = cp.unravel_index(cp.arange(x*y, dtype=cp.int64), (x,y))
        self._weights_gpu = None

        self._normalizeWeights = normalize_weights

        self._neigx_gpu = cp.arange(x, dtype=cp.float32)
        self._neigy_gpu = cp.arange(y, dtype=cp.float32) 

        if topology == 'hexagonal':
            self._xx_gpu = cp.array(self._xx)
            self._yy_gpu = cp.array(self._yy)

    def _activate(self, x_gpu):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x"""
        if len(x_gpu.shape) == 1:
            x_gpu = cp.expand_dims(x_gpu, axis=1)

        self._activation_map_gpu = self._activation_distance(
                x_gpu, 
                self._weights_gpu
        )

    def _gaussian_rect(self, c, sigma):
        """Returns a Gaussian centered in c on a rect topology

        This function is optimized wrt the generic one.
        """
        d = 2*np.pi*sigma*sigma

        nx = self._neigx_gpu[cp.newaxis,:]
        ny = self._neigy_gpu[cp.newaxis,:]
        cx = c[0][:,cp.newaxis]
        cy = c[1][:,cp.newaxis]

        ax = cp.exp(-cp.power(nx-cx, 2, dtype=cp.float32)/d)
        ay = cp.exp(-cp.power(ny-cy, 2, dtype=cp.float32)/d)
        return ax[:,:,cp.newaxis]*ay[:,cp.newaxis,:]

    def _gaussian_generic(self, c, sigma):
        """Returns a Gaussian centered in c on any topology
        
        TODO: this function is much slower than the _rect one
        """
        d = 2*np.pi*sigma*sigma

        nx = self._xx_gpu[cp.newaxis,:,:]
        ny = self._yy_gpu[cp.newaxis,:,:]
        cx = self._xx_gpu.T[c][:, cp.newaxis, cp.newaxis]
        cy = self._yy_gpu.T[c][:, cp.newaxis, cp.newaxis]

        ax = cp.exp(-cp.power(nx-cx, 2, dtype=cp.float32)/d)
        ay = cp.exp(-cp.power(ny-cy, 2, dtype=cp.float32)/d)
        return (ax*ay).transpose((0,2,1))
    
    def _mexican_hat_rect(self, c, sigma):
        """Mexican hat centered in c (only rect topology)"""
        d = 2*np.pi*sigma*sigma

        nx = self._neigx_gpu[cp.newaxis,:]
        ny = self._neigy_gpu[cp.newaxis,:]
        cx = c[0][:,cp.newaxis]
        cy = c[1][:,cp.newaxis]

        px = cp.power(nx-cx, 2, dtype=cp.float32)
        py = cp.power(ny-cy, 2, dtype=cp.float32)
        p = px[:,:,cp.newaxis] + py[:,cp.newaxis,:]
        
        return cp.exp(-p/d)*(1-2/d*p)

    def _mexican_hat_generic(self, c, sigma):
        """Mexican hat centered in c on any topology
        
        TODO: this function is much slower than the _rect one
        """
        d = 2*np.pi*sigma*sigma

        nx = self._xx_gpu[cp.newaxis,:,:]
        ny = self._yy_gpu[cp.newaxis,:,:]
        cx = self._xx_gpu.T[c][:, cp.newaxis, cp.newaxis]
        cy = self._yy_gpu.T[c][:, cp.newaxis, cp.newaxis]

        px = cp.power(nx-cx, 2, dtype=cp.float32)
        py = cp.power(ny-cy, 2, dtype=cp.float32)
        p = px + py
        
        return (cp.exp(-p/d)*(1-2/d*p)).transpose((0,2,1))

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        nx = self._neigx_gpu[cp.newaxis,:]
        ny = self._neigy_gpu[cp.newaxis,:]
        cx = c[0][:,cp.newaxis]
        cy = c[1][:,cp.newaxis]

        ax = cp.logical_and(nx > cx-sigma,
                         nx < cx+sigma)
        ay = cp.logical_and(ny > cy-sigma,
                         ny < cy+sigma)
        return (ax[:,:,cp.newaxis]*ay[:,cp.newaxis,:]).astype(cp.float32)

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        nx = self._neigx_gpu[cp.newaxis,:]
        ny = self._neigy_gpu[cp.newaxis,:]
        cx = c[0][:,cp.newaxis]
        cy = c[1][:,cp.newaxis]

        triangle_x = (-cp.abs(cx - nx)) + sigma
        triangle_y = (-cp.abs(cy - ny)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return triangle_x[:,:,cp.newaxis]*triangle_y[:,cp.newaxis,:]

    def _winner(self, x_gpu):
        """Computes the coordinates of the winning neuron for the sample x"""
        if len(x_gpu.shape) == 1:
            x_gpu = cp.expand_dims(x_gpu, axis=1)

        # Manca il controllo sulla finitezza

        self._activate(x_gpu)
        raveled_idxs = self._activation_map_gpu.argmin(axis=1)
        return (self._unravel_precomputed[0][raveled_idxs], self._unravel_precomputed[1][raveled_idxs])

    def update(self, x_gpu, wins, eta, sig):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        """
        
        g_gpu = self.neighborhood(wins, sig)*eta

        sum_g_gpu = cp.sum(g_gpu, axis=0)
        w_sum_g_gpu = self._weights_gpu * sum_g_gpu[:,:,cp.newaxis]

        sum_xg = cp.dot(
            g_gpu.reshape(g_gpu.shape[0],g_gpu.shape[1]*g_gpu.shape[2]).T, 
            x_gpu
        )

        self._numerator_gpu += sum_xg.reshape(w_sum_g_gpu.shape) - w_sum_g_gpu
        self._denominator_gpu += sum_g_gpu[:,:,cp.newaxis]


    def merge_updates(self):
        self._weights_gpu += cp.nan_to_num(
            self._numerator_gpu / self._denominator_gpu
        )
        
        if self._normalizeWeights:
            norms_gpu = cp.linalg.norm(self._weights_gpu, axis=2)
            self._weights_gpu = cp.nan_to_num(
                self._weights_gpu / norms_gpu[:,:,cp.newaxis]
            )


    def train(self, data, num_iteration, iter_beg=0, iter_end=None, verbose=False):
        if iter_end is None:
            iter_end = num_iteration

        # Copy arrays to device
        self._weights_gpu = cp.asarray(self._weights, dtype=cp.float32)
        data_gpu = cp.asarray(data, dtype=cp.float32)
        
        if verbose:
            print_progress(-1, num_iteration*len(data))

        for iteration in range(iter_beg, iter_end):
            try: # reuse already allocated memory
                self._numerator_gpu.fill(0)
                self._denominator_gpu.fill(0)
            except AttributeError: # whoops, I haven't allocated it yet
                self._numerator_gpu = cp.zeros(
                    self._weights_gpu.shape, 
                    dtype=cp.float32
                )
                self._denominator_gpu = cp.zeros(
                    (self._weights_gpu.shape[0], self._weights_gpu.shape[1],1),
                    dtype=cp.float32
                )

            eta = self._decay_function(self._learning_rate, iteration, num_iteration)
            # sigma and learning rate decrease with the same rule
            sig = self._decay_function(self._sigma, iteration, num_iteration)

            for i in range(0, len(data), self._n_parallel):
                start = i
                end = start + self._n_parallel
                if end > len(data):
                    end = len(data)

                self.update(data_gpu[start:end], self._winner(data_gpu[start:end]), eta, sig)

                if verbose:
                    print_progress(
                        iteration*len(data)+end-1, 
                        num_iteration*len(data)
                    )
                    
            self.merge_updates()

        # Copy back arrays to host
        self._weights = cp.asnumpy(self._weights_gpu)
        
        # free temporary memory
        del self._numerator_gpu
        del self._denominator_gpu
        del self._activation_map_gpu
        
        if verbose:
            print('\n quantization error:', self.quantization_error(data))

    def train_batch(self, data, num_iteration, verbose=False):
        return train(data, num_iteration, verbose=verbose)

    def train_random(self, data, num_iteration, verbose=False):
        print("WARNING: due to batch SOM algorithm, random order is not supported. Falling back to train_batch.")
        return train(data, num_iteration, verbose=verbose)

class TestCupySom(unittest.TestCase):
    def setUp(self):
        self.som = CupySom(5, 5, 1)
        self.minisom = MiniSom(5, 5, 1)

        for i in range(5):
            for j in range(5):
                # checking weights normalization
                np.testing.assert_almost_equal(1.0, np.linalg.norm(self.som._weights[i, j]))
        self.som._weights = np.zeros((5, 5, 1))  # fake weights
        self.som._weights[2, 3] = 5.0
        self.som._weights[1, 1] = 2.0
        
        np.random.seed(1234)
        cp.random.seed(1234)

    def test_euclidean_distance(self):
        x = np.random.rand(100, 20)
        w = np.random.rand(10,10,20)
        cs_dist = cp.asnumpy(euclidean_squared_distance(cp.array(x), cp.array(w)))
        cs_dist = cs_dist.reshape((100,10,10))
        for i, sample in enumerate(x):
            ms_dist = self.minisom._euclidean_distance(sample, w)**2
            np.testing.assert_array_almost_equal(ms_dist, cs_dist[i])

    def test_cosine_distance(self):
        x = np.random.rand(100, 20)
        w = np.random.rand(10,10,20)
        cs_dist = cp.asnumpy(cosine_distance(cp.array(x), cp.array(w)))
        cs_dist = cs_dist.reshape((100,10,10))
        for i, sample in enumerate(x):
            ms_dist = self.minisom._cosine_distance(sample, w)
            np.testing.assert_array_almost_equal(ms_dist, cs_dist[i])

    def test_manhattan_distance(self):
        x = np.random.rand(100, 20)
        w = np.random.rand(10,10,20)
        cs_dist = cp.asnumpy(manhattan_distance(cp.array(x), cp.array(w)))
        cs_dist = cs_dist.reshape((100,10,10))
        for i, sample in enumerate(x):
            ms_dist = self.minisom._manhattan_distance(sample, w)
            np.testing.assert_array_almost_equal(ms_dist, cs_dist[i])

    def test_gaussian(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_gauss = cp.asnumpy(self.som._gaussian_rect(c, 1))
        print(cs_gauss.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_gauss = self.minisom._gaussian((x,y), 1)
            np.testing.assert_array_almost_equal(ms_gauss, cs_gauss[i])

    def test_mexican_hat(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(self.som._mexican_hat_rect(c, 1))
        print(cs_mex.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._mexican_hat((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_bubble(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(self.som._bubble(c, 1))
        print(cs_mex.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._bubble((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_triangle(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(self.som._triangle(c, 1))
        print(cs_mex.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._triangle((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])


class TestCupySomHex(unittest.TestCase):
    def setUp(self):
        self.som = CupySom(5, 5, 1, topology='hexagonal')
        self.minisom = MiniSom(5, 5, 1, topology='hexagonal')

        for i in range(5):
            for j in range(5):
                # checking weights normalization
                np.testing.assert_almost_equal(1.0, np.linalg.norm(self.som._weights[i, j]))
        self.som._weights = np.zeros((5, 5, 1))  # fake weights
        self.som._weights[2, 3] = 5.0
        self.som._weights[1, 1] = 2.0
        
        np.random.seed(1234)
        cp.random.seed(1234)

    def test_gaussian(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_gauss = cp.asnumpy(self.som._gaussian_generic(c, 1))
        print(cs_gauss.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            print(x,y)
            ms_gauss = self.minisom._gaussian((x,y), 1)
            np.testing.assert_array_almost_equal(ms_gauss, cs_gauss[i])

    def test_mexican_hat(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(self.som._mexican_hat_generic(c, 1))
        print(cs_mex.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._mexican_hat((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_bubble(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(self.som._bubble(c, 1))
        print(cs_mex.shape)

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._bubble((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])            
