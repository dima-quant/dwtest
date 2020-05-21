"""
Durbin-Watson statistic and p-value.
Free to use, copy, share, and modify.
Implemented by Dmytro Makogon. Inspired by lmtest/R/dwtest.R
"""
import numpy as np
from scipy.stats import norm
from numba import jit


def dwtest(res, X, tail='both', method='Pan', matrix_inverse = False, n = 15):
    """
    Calculate  Durbin-Watson statistic and p-value 
    References
    ----------
    J. Durbin and G. S. Watson (1950), "Testing for Serial Correlation in Least
    Squares Regression I". Biometrika, Vol. 37, 409-428.
    J. Durbin and G. S. Watson (1951), "Testing for Serial Correlation in Least 
    Squares Regression: II", Biometrika, Vol. 38, 159-177.
    J. Durbin and G. S. Watson (1971), "Testing for Serial Correlation in Least 
    Squares Regression: III", Biometrika, Vol. 58, 1-19.
    R. W. Farebrother (1980), "Algorithm AS 153: Pan's Procedure for the Tail Probabilities
    of the Durbin-Watson Statistic", Journal of the Royal Statistical Society, Series C
    (Applied Statistics), Vol. 29, 224-227.
    
    Parameters
    ----------
    res : ndarray (n_obs, 1)
        Array of residuals
    X : ndarray (n_obs, dim)
        Array of regressors
    tail : str
        test 'left', 'right', or 'both' for two-sided
    method : str  
        Calculation method. 'Pan' for Pan's procedure and 'normal' for normal approximation
    matrix_inverse : bool
        Using matrix inverse is faster but less stable
    n : int
        Parameter of the gradsol function  

    Returns
    -------
    p_val, DW : float64
        p value and Durbin-Watson statistic 
    """
    if method=='Pan':
        # use Pan's method
        (p_right, DW) = dwtest_pan(res, X, matrix_inverse = matrix_inverse, n = n)
        if (p_right<0) or (p_right>1):
            # warning: Pan's method did not work
            p_right = -1
    elif method=='normal':
        DW = dw_stat(res)
        p_right = -1
    else: 
        # error: no such method
        pass 
        
    
    if (p_right<0):
        # use normal approximation
        (dw_mean, dw_var) = dw_meanvar(X, matrix_inverse = matrix_inverse)
        p_right = norm.cdf((DW-dw_mean)/np.sqrt(dw_var))

    if tail == 'both':
        p_val = 2*min(p_right, 1 - p_right)
    elif tail == 'right':
        p_val = p_right
    elif tail == 'left':
        p_val = 1 - p_right
    else:
        pass # warning: no such test
    return p_val, DW

@jit(nopython=True, parallel=False)
def dw_stat(res):
    """Calculate Durbin-Watson statistic"""
    return np.sum(np.diff(res.T)**2) / np.sum(res**2)

@jit(nopython=True, parallel=False)    
def gradsol(x, a, m, n):
    """
    Translated from FORTRAN to Python by Dima
    -----------------------------------------
    TRANSLATION OF AMENDED VERSION OF APPLIED STATISTICS ALGORITHM
    AS 153 (AS R52), VOL. 33, 363-366, 1984.
    BY R.W. FAREBROTHER  (ORIGINALLY NAMED GRADSOL OR PAN)

    GRADSOL EVALUATES THE PROBABILITY THAT A WEIGHTED SUM OF
    SQUARED STANDARD NORMAL VARIATES DIVIDED BY X TIMES THE UNWEIGHTED
    SUM IS LESS THAN A GIVEN CONSTANT, I.E. THAT
    A1.U1**2 + A2.U2**2 + ... + AM.UM**2 <
    X*(U1**2 + U2**2 + ... + UM**2) + C
    WHERE THE U'S ARE STANDARD NORMAL VARIABLES.
    FOR THE DURBIN-WATSON STATISTIC, X = DW, AND
    A ARE THE NON-ZERO EIGENVALUES OF THE "M*A" MATRIX.
    
    ORIGINALLY FROM STATLIB.  REVISED 5/3/1996 BY CLINT CUMMINS
 
    Parameters
    ----------
    x : float
        The Durbin-Watson statistic
    a : ndarray (m)
        Sorted array of non-zero eigenvalues of MA matrix
    m : int
        Array size
    n : int
        The number of terms in the series, typically between 10 and 15

    Returns
    -------
    sum0 : float64
        p value 
    """
    nu = np.searchsorted(a, x, side = 'right')
    # index starts from 0, a[nu-1] <= x < a[nu]
    
    if nu == m:
        sum0 = 1.0
    elif nu == 0:
        sum0 = 0.0
    else:
        k = 1
        h = m - nu
    
        if h >= nu:
            d  = 2; h  = nu; k  = - k; 
            j1 = 0; j2 = 2; j3 = 3; j4 = 1
        else:
            d  = - 2; nu  = nu + 1; 
            j1 = m - 2; j2 = m - 1; j3 = m + 1; j4 = m
    
        pin = np.pi / (2*n)
        sum0 = (k + 1) / 2
        sgn0 = k / n
        n2  = 2*n - 1
    
        # first integral
        st = np.int32(h - 2*np.floor(h/2))
        for  l1 in range(st, -1, -1):
            # take into account that k = -np.sign(d)
            for l2 in range(j2, nu-k, d): 
                sum1 = a[j4-1]
                if l2 == 0:
                    prod0 = x
                else:
                    # FIX BY CLINT CUMMINS 5/3/96
                    # prod0 = a(j2)
                    prod0 = a[l2-1] 
                u = 0.5*(sum1 + prod0)
                v = 0.5*(sum1 - prod0)
                sum1 = 0.0
                for i in range(1,n2+2,2):
                    y = u - v*np.cos(i*pin)
                    num = y - x
                    prod0 = np.prod(num / (y - a[:j1]))
                    prod0 *= np.prod(num / (y - a[j3-1:]))
                    sum1 += np.sqrt(np.abs(prod0))
                sgn0 = -sgn0
                sum0 += sgn0*sum1
                j1 += d
                j3 += d
                j4 += d
            # second integral
            if d == 2:
                j3 = j3 - 1
            else:
                j1 = j1 + 1 
            j2 = 0
            nu  = 0
    return sum0

@jit(nopython=True, parallel=False)
def dwtest_pan(res, X, matrix_inverse = False, n = 15):
    """
    Calculate  Durbin-Watson statistic according to
    J. Durbin and G. S. Watson (1971), "Testing for Serial Correlation in Least 
    Squares Regression: III", Biometrika, Vol. 58, 1-19.
    
    Parameters
    ----------
    res : ndarray (n_obs, 1)
        Array of residuals
    X : ndarray (n_obs, dim)
        Array of regressors
    matrix_inverse : bool
        Using matrix inverse is faster but less stable
    n : int
        Parameter of the gradsol function  

    Returns
    -------
    p_right, DW : float64
        Cumulative probability and Durbin-Watson statistic 
    """
    DW = dw_stat(res)
    (n_obs, dim) = X.shape
    
    # construct A matrix
    upp_one = np.diag(np.ones(n_obs-1),1)    
    A =  2*np.eye(n_obs) - upp_one - upp_one.T   
    A[0, 0] = 1
    A[-1, -1] = 1
    # calculate X(X'X)^{-1}X'
    if matrix_inverse:
        R = np.linalg.inv(np.dot(X.T, X))
        QQt = np.dot(X.dot(R), X.T)
    else:
        # X = QR => QR (R'R)^-1 R'Q' = Q Q'
        Qm = np.linalg.qr(X)[0]
        QQt = Qm.dot(Qm.T)
    # add identity to prevent insignificant imaginary values    
    eigv = np.sort(np.real(np.linalg.eigvals(A - QQt.dot(A)+np.eye(n_obs))))-1
    eigv = eigv[dim:]
    eps = 1e-10
    eigv = eigv[eigv > eps]
    p_right = gradsol(DW, eigv, len(eigv), n)
    return p_right, DW

@jit(nopython=True, parallel=False)
def dw_meanvar(X, matrix_inverse = False):
    """
    Calculate mean and variance of Durbin-Watson statistic according to
    J. Durbin and G. S. Watson (1950), "Testing for Serial Correlation in Least
    Squares Regression I". Biometrika, 37, 409-428.
    J. Durbin and G. S. Watson (1951), "Testing for Serial Correlation in Least 
    Squares Regression: II", Biometrika, Vol. 38, 159-177.
    
    Parameters
    ----------
    X : ndarray (n_obs, dim)
        Array of regressors
    matrix_inverse : bool
        Using matrix inverse is faster but less stable

    Returns
    -------
    dw_mean, dw_var : float64
        Mean and variance of Durbin-Watson statistic 
    """
    (n_obs, dim) = X.shape
    AX = np.zeros(X.shape)
    AX[0,:] = X[0,:] - X[1,:]
    AX[-1,:] = X[-1,:] - X[-2,:]
    AX[1:-1,:] = -X[:-2,:] + 2*X[1:-1,:] - X[2:,:]
    if matrix_inverse:
        AXR = AX.dot(np.linalg.inv(np.dot(X.T, X)))
    else:
        AXR = np.linalg.lstsq(np.dot(X.T, X), AX.T, rcond=-1)[0].T
    B = np.dot(X.T, AXR)
    P = 2*(n_obs - 1) - np.trace(B)
    Q = 2*(3*n_obs - 4) - 2*np.trace(np.dot(AX.T, AXR)) + np.trace(B.dot(B))
    dw_mean = P/(n_obs - dim)
    dw_var = 2/((n_obs - dim)*(n_obs - dim + 2)) * (Q - P*dw_mean)
    return dw_mean, dw_var
