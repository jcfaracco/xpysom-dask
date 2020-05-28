from math import sqrt, ceil

from collections import defaultdict
from warnings import warn

import numpy as np
try:
    import cupy as cp
    default_xp = cp
except:
    print("WARNING: CuPy could not be imported")
    default_xp = np

from perfsom.minisom import MiniSom, asymptotic_decay, fast_norm, print_progress

from .distances import cosine_distance, manhattan_distance, euclidean_squared_distance, euclidean_squared_distance_part, euclidean_distance
from .neighborhoods import gaussian_generic, gaussian_rect, mexican_hat_generic, mexican_hat_rect, bubble, triangle, prepare_neig_func

# In my machine it looks like these are the best performance/memory trade-off.
# As a rule of thumb, executing more items at a time does not decrease 
# performance but it may increase the memory footprint without providing 
# significant gains.
DEFAULT_CUDA_CORE_OVERSUBSCRIPTION = 4
DEFAULT_CPU_CORE_OVERSUBSCRIPTION = 500

def find_cuda_cores():
    try:
        import subprocess
        return int(subprocess.check_output("nvidia-settings -q CUDACores -t", shell=True))
    except:
        print("Could not infer #cuda_cores")
        return 0

def find_cpu_cores():
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        print("Could not infer #CPU_cores")
        return 0

class CupySom(MiniSom):
    def __init__(self, x, y, input_len, sigma=0, sigmaN=1, learning_rate=0.5, learning_rateN=0.01, decay_function='exponential', neighborhood_function='gaussian', std_coeff=0.5, topology='rectangular', activation_distance='euclidean', normalize_weights=False, random_seed=None, n_parallel=0, xp=default_xp):
        # passing some mock parameters to disable checks
        super().__init__(x, y, input_len, sigma=sigma, sigmaN=sigmaN, learning_rate=learning_rate, learning_rateN=learning_rateN, decay_function=decay_function, neighborhood_function='gaussian', std_coeff=std_coeff, topology=topology, activation_distance='euclidean', random_seed=random_seed)

        self.xp = xp

        if n_parallel == 0:
            if self.xp.__name__ == 'cupy':
                n_parallel = find_cuda_cores()*DEFAULT_CUDA_CORE_OVERSUBSCRIPTION    
            else:
                n_parallel = find_cpu_cores()*DEFAULT_CPU_CORE_OVERSUBSCRIPTION  
 
            if n_parallel == 0:
                raise ValueError("n_parallel was not specified and could not be infered from system")
        
        self._n_parallel = n_parallel

        self._neigx_gpu = self.xp.arange(x, dtype=self.xp.float32)
        self._neigy_gpu = self.xp.arange(y, dtype=self.xp.float32) 

        if topology == 'hexagonal':
            self._xx_gpu = self.xp.array(self._xx)
            self._yy_gpu = self.xp.array(self._yy)


        if topology == 'rectangular':
            neig_functions = {
                'gaussian': prepare_neig_func(
                    gaussian_rect, self._neigx_gpu, self._neigy_gpu, self._std_coeff),
                'mexican_hat': prepare_neig_func(
                    mexican_hat_rect, self._neigx_gpu, self._neigy_gpu, self._std_coeff),
                'bubble': prepare_neig_func(
                    bubble, self._neigx_gpu, self._neigy_gpu),
                'triangle': prepare_neig_func(
                    triangle, self._neigx_gpu, self._neigy_gpu),
            }
        elif topology == 'hexagonal':
            neig_functions = {
                'gaussian': prepare_neig_func(
                    gaussian_generic, self._xx_gpu, self._yy_gpu, self._std_coeff),
                'mexican_hat': prepare_neig_func(
                    mexican_hat_generic, self._xx_gpu, self._yy_gpu, self._std_coeff),
                'bubble': prepare_neig_func(
                    bubble, self._neigx_gpu, self._neigy_gpu),
            }
        else:
            neig_functions = {}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {
            'euclidean': euclidean_squared_distance_part,
            'euclidean_no_opt': euclidean_squared_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance,
        }

        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]

        # print("Using n_parallel = " + str(self._n_parallel))
        self._unravel_precomputed = self.xp.unravel_index(self.xp.arange(x*y, dtype=self.xp.int64), (x,y))
        self._weights_gpu = None

        self._normalizeWeights = normalize_weights

    def _activate(self, x_gpu):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x"""
        if len(x_gpu.shape) == 1:
            x_gpu = self.xp.expand_dims(x_gpu, axis=1)

        if self._sq_weights_gpu is not None:
            self._activation_map_gpu = self._activation_distance(
                    x_gpu, 
                    self._weights_gpu,
                    self._sq_weights_gpu,
                    xp=self.xp
            )
        else:
            self._activation_map_gpu = self._activation_distance(
                    x_gpu, 
                    self._weights_gpu,
                    xp=self.xp
            )

    def _winner(self, x_gpu):
        """Computes the coordinates of the winning neuron for the sample x"""
        if len(x_gpu.shape) == 1:
            x_gpu = self.xp.expand_dims(x_gpu, axis=1)

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
        
        g_gpu = self.neighborhood(wins, sig, xp=self.xp)*eta

        sum_g_gpu = self.xp.sum(g_gpu, axis=0)
        w_sum_g_gpu = self._weights_gpu * sum_g_gpu[:,:,self.xp.newaxis]

        sum_xg = self.xp.dot(
            g_gpu.reshape(g_gpu.shape[0],g_gpu.shape[1]*g_gpu.shape[2]).T, 
            x_gpu
        )

        self._numerator_gpu += sum_xg.reshape(w_sum_g_gpu.shape) - w_sum_g_gpu
        self._denominator_gpu += sum_g_gpu[:,:,self.xp.newaxis]


    def merge_updates(self):
        self._weights_gpu += self.xp.nan_to_num(
            self._numerator_gpu / self._denominator_gpu
        )
        
        if self._normalizeWeights:
            norms_gpu = self.xp.linalg.norm(self._weights_gpu, axis=2)
            self._weights_gpu = self.xp.nan_to_num(
                self._weights_gpu / norms_gpu[:,:,self.xp.newaxis]
            )


    def train(self, data, num_iteration, iter_beg=0, iter_end=None, verbose=False):
        if iter_end is None:
            iter_end = num_iteration

        # Copy arrays to device
        self._weights_gpu = self.xp.asarray(self._weights, dtype=self.xp.float32)
        data_gpu = self.xp.asarray(data, dtype=self.xp.float32)
        
        if verbose:
            print_progress(-1, num_iteration*len(data))

        for iteration in range(iter_beg, iter_end):
            try: # reuse already allocated memory
                self._numerator_gpu.fill(0)
                self._denominator_gpu.fill(0)
            except AttributeError: # whoops, I haven't allocated it yet
                self._numerator_gpu = self.xp.zeros(
                    self._weights_gpu.shape, 
                    dtype=self.xp.float32
                )
                self._denominator_gpu = self.xp.zeros(
                    (self._weights_gpu.shape[0], self._weights_gpu.shape[1],1),
                    dtype=self.xp.float32
                )

            if self._activation_distance in [
                    euclidean_squared_distance,
                    euclidean_squared_distance_part,
                    cosine_distance
            ]:
                self._sq_weights_gpu = (
                    self.xp.power(
                        self._weights_gpu.reshape(
                            -1, self._weights_gpu.shape[2]
                        ),
                        2
                    ).sum(axis=1, keepdims=True)
                )
            else:
                self._sq_weights_gpu = None

            eta = self._decay_function(self._learning_rate, self._learning_rateN, iteration, num_iteration)
            # sigma and learning rate decrease with the same rule
            sig = self._decay_function(self._sigma, self._sigmaN, iteration, num_iteration)

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
        if self.xp.__name__ == 'cupy':
            self._weights = self.xp.asnumpy(self._weights_gpu)
        else:
            self._weights = self._weights_gpu
        
        # free temporary memory
        del self._numerator_gpu
        del self._denominator_gpu
        del self._activation_map_gpu
        del self._sq_weights_gpu
        
        if verbose:
            print('\n quantization error:', self.quantization_error(data))

    def train_batch(self, data, num_iteration, verbose=False):
        return train(data, num_iteration, verbose=verbose)

    def train_random(self, data, num_iteration, verbose=False):
        print("WARNING: due to batch SOM algorithm, random order is not supported. Falling back to train_batch.")
        return train(data, num_iteration, verbose=verbose)

    def quantization(self, data_gpu):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data_gpu)
        winners_coords = self.xp.argmin(self._distance_from_weights(data_gpu), axis=1)
        return self._weights_gpu[self.xp.unravel_index(winners_coords,
                                           self._weights.shape[:2])]

    def _distance_from_weights(self, data_gpu):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        distances = []
        for start in range(0, len(data_gpu), self._n_parallel):
            end = start + self._n_parallel
            if end > len(data_gpu):
                end = len(data_gpu)
            
            distances.append(euclidean_distance(data_gpu[start:end], self._weights_gpu, xp=self.xp))
        return self.xp.vstack(distances)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)

        # load to GPU
        data_gpu = self.xp.array(data, dtype=self.xp.float32)
        self._weights_gpu = self.xp.array(self._weights)

        # recycle buffer
        data_gpu -= self.quantization(data_gpu) 

        # free no longer needed buffer
        del self._weights_gpu

        qe = self.xp.linalg.norm(data_gpu, axis=1).mean()
        
        # free no longer needed buffer
        del data_gpu

        return qe.item()

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not ajacent conunts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        total_neurons = np.prod(self._weights.shape)
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return np.nan

        # load to GPU
        data_gpu = self.xp.array(data, dtype=self.xp.float32)
        self._weights_gpu = self.xp.array(self._weights)

        distances = self._distance_from_weights(data_gpu) 

        # free no longer needed buffers
        del self._weights_gpu
        del data_gpu

        # b2mu: best 2 matching units
        b2mu_inds = self.xp.argsort(distances, axis=1)[:, :2]
        b2my_xy = self.xp.unravel_index(b2mu_inds, self._weights.shape[:2])
        if self.topology ==  'rectangular':
            b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
            diff_b2mu_x = self.xp.abs(self.xp.diff(b2mu_x))
            diff_b2mu_y = self.xp.abs(self.xp.diff(b2mu_y))
            return ((diff_b2mu_x > 1) | (diff_b2mu_y > 1)).mean().item()
        elif self.topology == 'hexagonal':
            b2mu_x = self._xx_gpu[b2my_xy[0], b2my_xy[1]]
            b2mu_y = self._yy_gpu[b2my_xy[0], b2my_xy[1]]
            dxdy = self.xp.hstack([self.xp.diff(b2mu_x), self.xp.diff(b2mu_y)])
            distance = self.xp.linalg.norm(dxdy, axis=1)
            return (distance > 1.5).mean().item()
