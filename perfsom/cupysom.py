from math import sqrt, ceil

from collections import defaultdict
from warnings import warn
import numpy as np
import cupy as cp

from perfsom.minisom import MiniSom, asymptotic_decay, fast_norm

def find_cuda_cores():
    try:
        import subprocess
        return int(subprocess.check_output("nvidia-settings -q CUDACores -t", shell=True))
    except:
        print("Could not infer #cuda_cores")
        return 0

add_a_and_b_over_c = cp.ElementwiseKernel(
    'T a, T b, T c',
    'T y',
    'y = a + b/c',
    'divide_sum')

calc_update = cp.ReductionKernel(
    'T x, T w, T g',
    'T y',
    '(x - w) * g',
    'a + b',
    'y = a',
    '0',
    'calc_update')

l2dist_squared = cp.ReductionKernel(
    'T x, T y',
    'T z',
    '(x - y) * (x - y)',
    'a + b',
    'z = a',
    '0',
    'l2dist_squared')

def ravel_idx_2d(idx, cols):
    return idx[0] * cols + idx[1]

def unravel_idx_2d(i, cols):
    return (i % cols, cp.floor_divide(i, cols))

class CupySom(MiniSom):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function='exponential', neighborhood_function='gaussian', topology='rectangular', activation_distance='euclidean', random_seed=None, n_parallel=0):
        # passing some mock parameters to disable checks
        super().__init__(x, y, input_len, sigma=sigma, learning_rate=learning_rate, decay_function=decay_function, neighborhood_function='gaussian', topology='rectangular', activation_distance='euclidean', random_seed=random_seed)

        if n_parallel == 0:
            n_parallel = find_cuda_cores()//2    # In my GPU it looks like this is the best performance/memory trade-off
            
            if n_parallel == 0:
                raise ValueError("n_parallel was not specified and could not be infered from system")
        
        self._n_parallel = n_parallel

        if topology not in ['rectangular']:
            msg = '%s not supported only rectangular available'
            raise ValueError(msg % topology)

        neig_functions = {
            'gaussian': self._gaussian,
            'gaussian_precomputed': self._gaussian_precomputed,
        }

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        self.neighborhood = neig_functions[neighborhood_function]

        if neighborhood_function == 'gaussian_precomputed':
            self._precompute_gaussian()

        distance_functions = {'euclidean': l2dist_squared}

        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]

        # print("Using n_parallel = " + str(self._n_parallel))
        self._unravel_precomputed = cp.unravel_index(cp.arange(x*y, dtype=cp.int64), (x,y))
        self._weights_gpu = None

        self._normalizeWeights = False

        self._neigx = cp.arange(x, dtype=cp.float32)
        self._neigy = cp.arange(y, dtype=cp.float32) 

    def _activate(self, x_gpu):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x"""
        if len(x_gpu.shape) == 1:
            x_gpu = cp.expand_dims(x_gpu, axis=1)

        self._activation_map_gpu = self._activation_distance(
                x_gpu[:,:,cp.newaxis,cp.newaxis], 
                self._weights_gpu[:,cp.newaxis,:,:], 
                axis=0)

    # I tried to speed up gaussian function, which takes 15% of the overall
    # execution time, by precalculating some parts and caching others but
    # performance only improved by 3% overall, give or take.
    # Maybe a lazier approach would be better. But it would have to be iterative.

    def _precompute_gaussian(self):
        """Returns a Gaussian centered in c"""
        x = self._weights.shape[0]
        y = self._weights.shape[1]

        nx = cp.arange(x, dtype=cp.float32).reshape((1,x))
        ny = cp.arange(y, dtype=cp.float32).reshape((1,y))
        cx = cp.arange(x, dtype=cp.float32).reshape((x,1))
        cy = cp.arange(y, dtype=cp.float32).reshape((y,1))

        # Precompute ax and ay so I will only have to ^(1/d) to calculate
        # "real" ax and ay
        self._precomputed_ax = cp.exp(-cp.power(nx-cx, 2))
        self._precomputed_ay = cp.exp(-cp.power(ny-cy, 2))

        self._old_sigma = None
        i = cp.arange(x*y, dtype=cp.int32)
        self._idxs_precomputed = unravel_idx_2d(i, x)

    def _gaussian_precomputed(self, c, sigma):
        x = self._weights.shape[0]
        y = self._weights.shape[1]

        # If sigma changes, precompute gaussian matrix for every possible winning
        # vector
        if sigma != self._old_sigma:
            self._old_sigma = sigma

            d = 2*np.pi*sigma*sigma
            ax = cp.power(self._precomputed_ax, 1/d) 
            ay = cp.power(self._precomputed_ay, 1/d) 
            self._gaussian_cache = cp.einsum('ij, kl -> ikjl', ax, ay) 

        # Take the winning vectors' precomputed gaussian matrix
        return self._gaussian_cache[c[0], c[1], :, :]

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c"""
        d = 2*np.pi*sigma*sigma

        nx = self._neigx[cp.newaxis,:]
        ny = self._neigy[cp.newaxis,:]
        cx = c[0][:,cp.newaxis]
        cy = c[1][:,cp.newaxis]

        ax = cp.exp(-cp.power(nx-cx, 2, dtype=cp.float32)/d)
        ay = cp.exp(-cp.power(ny-cy, 2, dtype=cp.float32)/d)
        return ax[:,:,cp.newaxis]*ay[:,cp.newaxis,:]

    def _winner(self, x_gpu):
        """Computes the coordinates of the winning neuron for the sample x"""
        if len(x_gpu.shape) == 1:
            x_gpu = cp.expand_dims(x_gpu, axis=1)

        # Manca il controllo sulla finitezza

        self._activate(x_gpu)
        raveled_idxs = self._activation_map_gpu.argmin(axis=(1,2))
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

        
        # wins = cp.unravel_index(wins_raveled, self._activation_map_gpu.shape)

        # improves the performances
        g_gpu = self.neighborhood(wins, sig)*eta

        # idx = cp.arange(g_gpu.shape[0]) * (g_gpu.shape[1]*g_gpu.shape[2]) + wins[0] * g_gpu.shape[2] + wins[1]
        # mins = g_gpu.ravel()[idx]
        # # Same as assigning 0 to masked values NB: need to negate condition!
        # g_gpu *= (g_gpu >= self._neigh_threshold * mins[:,cp.newaxis, cp.newaxis])

        self._numerator_gpu += calc_update(
                x_gpu[:,:,cp.newaxis,cp.newaxis],
                self._weights_gpu[:,cp.newaxis,:,:],
                g_gpu[cp.newaxis,:,:,:],
                axis=1
        )

        self._denominator_gpu += cp.sum(g_gpu, axis=0)[cp.newaxis,:,:]


    def merge_updates(self):
        # self._denominator_gpu[self._denominator_gpu == 0] = 1   # no div0
        self._weights_gpu = add_a_and_b_over_c(self._weights_gpu, self._numerator_gpu, self._denominator_gpu)
        if self._normalizeWeights:
            norms_gpu = cp.linalg.norm(self._weights_gpu, axis=2)
            norms_gpu[norms_gpu == 0] = 1   # Avoid divide by zero
            norms_gpu = norms_gpu[:,:,cp.newaxis]  # prepare for broadcast
            self._weights_gpu = cp.divide(self._weights_gpu, norms_gpu)


    def train(self, data, num_iteration, iter_beg=0, iter_end=None, verbose=False):
        if iter_end is None:
            iter_end = num_iteration

        # Copy arrays to device
        self._weights_gpu = cp.transpose(cp.asarray(self._weights, dtype=cp.float32), axes=(2,0,1))
        data_gpu = cp.transpose(cp.asarray(data, dtype=cp.float32), axes=(1,0))

        batch_size = len(data)
        setIdx = np.arange(ceil(len(data)/self._n_parallel))
        currIdx = 0
        perc = -1
        for iteration in range(iter_beg, iter_end):
            if verbose:
                new_perc = int(100 * iteration / num_iteration)
                if new_perc > perc:
                    print("Training [" + str(new_perc) + "%]...")
                    perc = new_perc

            self._numerator_gpu   = cp.zeros(self._weights_gpu.shape, dtype=cp.float32)
            self._denominator_gpu = cp.zeros((1, self._weights_gpu.shape[1], self._weights_gpu.shape[2]), dtype=cp.float32)

            eta = self._decay_function(self._learning_rate, iteration, num_iteration)
            # sigma and learning rate decrease with the same rule
            sig = self._decay_function(self._sigma, iteration, num_iteration)

            for i in range(0, batch_size, self._n_parallel):
                start = setIdx[currIdx] * self._n_parallel
                end = start + self._n_parallel
                if end > len(data):
                    end = len(data)
                self.update(data_gpu[:,start:end], self._winner(data_gpu[:,start:end]), eta, sig)
                currIdx = (currIdx + 1) % len(setIdx)
            self.merge_updates()

        # Copy back arrays to host
        self._weights = cp.asnumpy(cp.transpose(self._weights_gpu, axes=(1,2,0)))
        
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
