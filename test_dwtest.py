# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:23:15 2020

"""

import unittest
import dwtest as dw
import numpy as np

class TestDWtestMethods(unittest.TestCase):
    """"Unit tests for dwtest module"""    
    def test_dwtest_benchmark(self):
        # test normal approximation 
        drv = np.array([[1, 6, 9, 7, 1, 2, 3, 4, 2, 1], 
                [3, 4, 2, 9, -1, 2, 2, 5, 5, 0],
                [2, 9, 0, 3, 2, 4, 4, 1, 9, 7]],ndmin =2).T
        #Fit a linear regression to the data
        y = np.array([0, 9, 2, 3, 6, 4, 0, 1, 2, 8],ndmin =2).T
        n_obs = len(y)
        intercept = np.ones(shape=(n_obs, 1))
        x = np.concatenate([intercept, drv], axis=1)
        beta = np.linalg.lstsq(x, y,rcond=-1)[0]
        res = y - np.dot(x, beta)  
        pv_bench = 0.807628109619
        pv_norm = dw.dwtest(res, x, tail='right', method='normal')[0]
        self.assertFalse(np.any(np.round(pv_bench - pv_norm,8)))
        
    def test_method_similarity(self):
        # test similarity   
        drv = np.array([[1, 6, 9, 7, 1, 2, 3, 4, 2, 1], 
                [3, 4, 2, 9, -1, 2, 2, 5, 5, 0],
                [2, 9, 0, 3, 2, 4, 4, 1, 9, 7]],ndmin =2).T
        #Fit a linear regression to the data.
        y = np.array([0, 9, 2, 3, 6, 4, 0, 1, 2, 8],ndmin =2).T
        n_obs = len(y)
        intercept = np.ones(shape=(n_obs, 1))
        x = np.concatenate([intercept, drv], axis=1)
        beta = np.linalg.lstsq(x, y,rcond=-1)[0]
        res = y - np.dot(x, beta)  
        pv_pan = dw.dwtest_pan(res, x)[0]
        pv_norm = dw.dwtest(res, x, tail='right', method='normal')[0]
        self.assertFalse(np.any(np.round(pv_pan - pv_norm,1)))
      
    def test_inverse_equivalence(self):
        # test equivalence to matrix inverse   
        drv = np.array([[1, 6, 9, 7, 1, 2, 3, 4, 2, 1], 
                [3, 4, 2, 9, -1, 2, 2, 5, 5, 0],
                [2, 9, 0, 3, 2, 4, 4, 1, 9, 7]],ndmin =2).T
        #Fit a linear regression to the data.
        y = np.array([0, 9, 2, 3, 6, 4, 0, 1, 2, 8],ndmin =2).T
        n_obs = len(y)
        intercept = np.ones(shape=(n_obs, 1))
        x = np.concatenate([intercept, drv], axis=1)
        beta = np.linalg.lstsq(x, y,rcond=-1)[0]
        res = y - np.dot(x, beta)  
        pv_pan = dw.dwtest_pan(res, x, matrix_inverse = False)[0]
        pv_pan_inv = dw.dwtest_pan(res, x, matrix_inverse = True)[0]
        pv_norm = dw.dwtest(res, x, method='normal', matrix_inverse = False)[0]
        pv_norm_inv = dw.dwtest(res, x, method='normal', matrix_inverse = True)[0]
        self.assertFalse(np.any(np.round(pv_pan - pv_pan_inv,8)))
        self.assertFalse(np.any(np.round(pv_norm - pv_norm_inv,8)))
        
if __name__ == '__main__':
    unittest.main()
