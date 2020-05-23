from math import sqrt, ceil
import unittest

import numpy as np
import cupy as cp

from perfsom.minisom import MiniSom

from .cupysom import CupySom
from .distances import cosine_distance, manhattan_distance, euclidean_squared_distance
from .neighborhoods import gaussian_generic, gaussian_rect, mexican_hat_generic, mexican_hat_rect, bubble, triangle, prepare_neig_func


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

        cs_gauss = cp.asnumpy(gaussian_rect(self.som._neigx_gpu, self.som._neigy_gpu, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_gauss = self.minisom._gaussian((x,y), 1)
            np.testing.assert_array_almost_equal(ms_gauss, cs_gauss[i])

    def test_mexican_hat(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(mexican_hat_rect(self.som._neigx_gpu, self.som._neigy_gpu, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._mexican_hat((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_bubble(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(bubble(self.som._neigx_gpu, self.som._neigy_gpu, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._bubble((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_triangle(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(triangle(self.som._neigx_gpu, self.som._neigy_gpu, c, 1))

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

        cs_gauss = cp.asnumpy(gaussian_generic(self.som._xx_gpu, self.som._yy_gpu, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_gauss = self.minisom._gaussian((x,y), 1)
            np.testing.assert_array_almost_equal(ms_gauss, cs_gauss[i])

    def test_mexican_hat(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(mexican_hat_generic(self.som._xx_gpu, self.som._yy_gpu, self.som._std_coeff, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._mexican_hat((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])

    def test_bubble(self):
        cx, cy = cp.meshgrid(cp.arange(5), cp.arange(5))
        c = (cx.flatten(), cy.flatten())        

        cs_mex = cp.asnumpy(bubble(self.som._neigx_gpu, self.som._neigy_gpu, c, 1))

        for i in range(len(c[0])):
            x = cp.asnumpy(c[0][i]).item()
            y = cp.asnumpy(c[1][i]).item()
            ms_mex = self.minisom._bubble((x,y), 1)
            np.testing.assert_array_almost_equal(ms_mex, cs_mex[i])            
