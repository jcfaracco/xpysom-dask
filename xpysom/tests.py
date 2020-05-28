from math import sqrt, ceil
import unittest

import numpy as np
import cupy as cp

from minisom import MiniSom

from .xpysom import XPySom
from .distances import cosine_distance, manhattan_distance, euclidean_squared_distance
from .neighborhoods import gaussian_generic, gaussian_rect, mexican_hat_generic, mexican_hat_rect, bubble, triangle, prepare_neig_func

import pickle
import os

class TestCupySom(unittest.TestCase):
    def setUp(self):
        self.som = XPySom(5, 5, 1, std_coeff=np.sqrt(np.pi))
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


    def test_unavailable_neigh_function(self):
        with self.assertRaises(ValueError):
            XPySom(5, 5, 1, neighborhood_function='boooom')

    def test_unavailable_distance_function(self):
        with self.assertRaises(ValueError):
            XPySom(5, 5, 1, activation_distance='ridethewave')

    def test_win_map(self):
        winners = self.som.win_map([[5.0], [2.0]])
        assert winners[(2, 3)][0] == [5.0]
        assert winners[(1, 1)][0] == [2.0]

    def test_labels_map(self):
        labels_map = self.som.labels_map([[5.0], [2.0]], ['a', 'b'])
        assert labels_map[(2, 3)]['a'] == 1
        assert labels_map[(1, 1)]['b'] == 1
        with self.assertRaises(ValueError):
            self.som.labels_map([[5.0]], ['a', 'b'])

    def test_activation_reponse(self):
        response = self.som.activation_response([[5.0], [2.0]])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_distance_from_weights(self):
        data = np.arange(-5, 5).reshape(-1, 1)
        weights = self.som._weights.reshape(-1, self.som._weights.shape[2])
        distances = self.som.distance_from_weights(data)
        for i in range(len(data)):
            for j in range(len(weights)):
                assert(distances[i][j] == np.linalg.norm(data[i] - weights[j]))

    def test_quantization_error(self):
        assert self.som.quantization_error([[5], [2]]) == 0.0
        assert self.som.quantization_error([[4], [1]]) == 1.0

    def test_topographic_error(self):
        # 5 will have bmu_1 in (2,3) and bmu_2 in (2, 4)
        # which are in the same neighborhood
        self.som._weights[2, 4] = 6.0
        # 15 will have bmu_1 in (4, 4) and bmu_2 in (0, 0)
        # which are not in the same neighborhood
        self.som._weights[4, 4] = 15.0
        self.som._weights[0, 0] = 14.
        assert self.som.topographic_error([[5]]) == 0.0
        assert self.som.topographic_error([[15]]) == 1.0


    def test_quantization(self):
        q = self.som.quantization(np.array([[4], [2]]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = XPySom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = XPySom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        # same initialization
        np.testing.assert_array_almost_equal(som1._weights, som2._weights)
        data = np.random.rand(100, 2)
        som1 = XPySom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data, 10)
        som2 = XPySom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        np.testing.assert_array_almost_equal(som1._weights, som2._weights)

    def test_train(self):
        som = XPySom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = np.array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10)
        assert q1 > som.quantization_error(data)

        data = np.array([[1, 5], [6, 7]])
        q1 = som.quantization_error(data)
        som.train(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = XPySom(2, 2, 2, random_seed=1)
        som.random_weights_init(np.array([[1.0, .0]]))
        for w in som._weights:
            np.testing.assert_array_equal(w[0], np.array([1.0, .0]))

    def test_pca_weights_init(self):
        som = XPySom(2, 2, 2)
        som.pca_weights_init(np.array([[1.,  0.], [0., 1.], [1., 0.], [0., 1.]]))
        expected = np.array([[[0., -1.41421356], [-1.41421356, 0.]],
                          [[1.41421356, 0.], [0., 1.41421356]]])
        np.testing.assert_array_almost_equal(som._weights, expected)

    def test_distance_map(self):
        som = XPySom(2, 2, 2, random_seed=1)
        som._weights = np.array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        np.testing.assert_array_equal(som.distance_map(), np.array([[1., 1.], [1., 1.]]))

        som = MiniSom(2, 2, 2, topology='hexagonal', random_seed=1)
        som._weights = np.array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        np.testing.assert_array_equal(som.distance_map(), np.array([[.5, 1.], [1., .5]]))

    def test_pickling(self):
        with open('som.p', 'wb') as outfile:
            pickle.dump(self.som, outfile)
        with open('som.p', 'rb') as infile:
            pickle.load(infile)
        os.remove('som.p')

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

        cs_gauss = cp.asnumpy(gaussian_rect(self.som._neigx, self.som._neigy, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_gauss = self.minisom._gaussian((x,y), 1)
            np.testing.assert_array_almost_equal(ms_gauss, cs_gauss[i])

    def test_mexican_hat(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(mexican_hat_rect(self.som._neigx, self.som._neigy, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._mexican_hat((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_bubble(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(bubble(self.som._neigx, self.som._neigy, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._bubble((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_triangle(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(triangle(self.som._neigx, self.som._neigy, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._triangle((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])


class TestCupySomHex(unittest.TestCase):
    def setUp(self):
        self.som = XPySom(5, 5, 1, topology='hexagonal', std_coeff=np.sqrt(np.pi))
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

        cs_gauss = cp.asnumpy(gaussian_generic(self.som._xx, self.som._yy, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_gauss = self.minisom._gaussian((x,y), 1)
            np.testing.assert_array_almost_equal(ms_gauss, cs_gauss[i])

    def test_mexican_hat(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(mexican_hat_generic(self.som._xx, self.som._yy, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._mexican_hat((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_bubble(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(bubble(self.som._neigx, self.som._neigy, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._bubble((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])            
