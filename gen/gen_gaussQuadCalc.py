'''
Author:      Kartikay Shukla
File:        gen_gaussQuadCalc.py
Created:     July 29, 2025 
LM:          August 2, 2025

DESCRIPTION
This file contains functions for computing Gauss points and Gauss weights for numerical integration using Gauss-Legendre Quadrature Rule.
'''

import numpy as np

def gLQ(NGQP):
    '''
    This function uses NumPy library to call Gauss-Legendre points and weights. For more information visit https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html.

    NGQP: Number of Guass Quadrature points
    '''
    gaussLeg = {'points': np.empty(NGQP), 'weights': np.empty(NGQP)}
    gaussLeg['points'], gaussLeg['weights'] = np.polynomial.legendre.leggauss(NGQP)

    return gaussLeg

def gaussLegQuad(NGQP=None):
    '''
    This function calculates the gaussian points and gaussian weights. Gaussian points are the roots of Legendre polynomials. 
    Gaussian weights are determined by the compairing the result of integration of basis function from -1 to +1 with Gauss quadrature rule. Depending on the number of Gauss Quadrature points desired, the function gives a dictionary with Gauss points for polynomials form 0 to NGQP th order and corresponding weights. 
    For more information visit https://en.wikipedia.org/wiki/Legendre_polynomials.

    NGQP: Number of Guass Quadrature points
    '''

    if NGQP is None:
        NGQP = 2
        print('Number of Guass quadrature points desired not defined. Default value of 2 is taken.')

    #---------------------Gauss Points-----------------------#
    # Intialising Legendre Polynomials
    polyLeg = {}
    polyLeg['P0'] = 1
    polyLeg['P1'] = np.polynomial.Polynomial([0,1])

    # creating higher order Legendre Polynomials using Bonnet's recursion formula
    for n in list(range(2,NGQP+1)):
        partI = (2*n-1)*polyLeg['P1']*polyLeg[f'P{n-1}']
        partII = (n-1)*polyLeg[f'P{n-2}']
        polyLeg[f'P{n}'] = (1/n)*(partI - partII)

    # Solving roots of NGQP th order Legendre Polynomial
    # Gauss points are the roots of Legendre polynomials
    gaussPoints = np.array(polyLeg[f'P{NGQP}'].roots())

    #---------------------Gauss Weights-----------------------#
    '''
    A set of basis functions will be assumed and they will be integrations with limits -1 to +1. This integration will be compared to Gauss Quadrature rule to solve for the Gaussian weights.
    Basis functions: 1, x, x**2, x**3 etc.

    Solutions of integration of basis functions: integration of f(x) from -1 to +1 where f(x) is a basis function. It can be verified that mathematically that basis functions with odd powers give 0 as a result of integration from -1 to +1 and even powers give the results as 2/(n+1) where n is the order of polynomial.

    See first equation at https://en.wikipedia.org/wiki/Gaussian_quadrature.
    '''

    # creating basis functions
    a = np.array([0,1]) # dummy array
    basisFunc = {f'bF{n}': np.polynomial.Polynomial([a[0]]*n+[a[1]]) for n in list(range(NGQP))}

    # Solution of integration
    b = np.array([2/(n+1) if np.mod(n,2)==0 else 0 for n in list(range(NGQP))])

    # computing coefficients of weights (matrix A in linear algebra relation Ax = b)
    A = np.empty((NGQP,NGQP))
    for n in list(range(NGQP)):
        A[n] = [basisFunc[f'bF{n}'](val) for val in gaussPoints]
    
    gaussWt = np.array(np.linalg.solve(A,b))

    return gaussPoints, gaussWt