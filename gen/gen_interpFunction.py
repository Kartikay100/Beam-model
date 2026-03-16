'''
Author:      Kartikay Shukla
File:        gen_interpFunction.py
Created:     July 24, 2025 
LM:          Aug 3, 2025

DESCRIPTION
This file contains functions for computing interpolation functions. Based on number of nodes, it assumes a polynomial and computes the interpolation function.
'''

import sympy as sy
import numpy as np
from scipy.special import legendre
from .gen_gaussQuadCalc import *

def spectralNodes(points:int) -> np.ndarray:
    '''
    This function creates an array of spectral nodal points (also called as Gauss-Lobatto-Legendre points) over the interval [-1,1].
    points: number of nodes. Defines the degree of polynomial.
    '''
    polyOrder = points-1
    dervPolyOrder = legendre(polyOrder).deriv()

    rootsDervPolyOrder = dervPolyOrder.roots
    rootsDervPolyOrder.sort()
    specNodes = np.insert(rootsDervPolyOrder, [0,len(rootsDervPolyOrder)],[-1,1])

    return specNodes

def interpLagGLQ(NNPEL, NGQP):
    '''
    This is a function that returns a numpy matrix (array of array) of Lagrangian interpolation functions calculated at Gauss points. It has been taken from the classical formula for Lagrange polynomials.
    More information about Lagrange polynomials can be found at https://en.wikipedia.org/wiki/Lagrange_polynomial.
    NNPEL: Number of nodes per element.
    NGQP: Number of Gauss Quadrature Points.
    '''

    #------------------Initialise-------------------------------#
    # calling domain values at nodes in natural coordinates
    domain = spectralNodes(NNPEL) # domain of natural coordinates, used for integration using Gauss Legendre Quadrature points, calls Gauss–Lobatto node
    gaussPoints = gLQ(NGQP)['points'] # NumPy library to call Gauss-Legendre
    # gaussWts = gLQ(NGQP)['weights']

    # Memory allocation
    funcLag = np.zeros(shape=(NNPEL,NGQP),dtype=np.float64)
    funcLagDiff = np.zeros(shape=(NNPEL,NGQP),dtype=np.float64)
    
    #-------------------Interpolation Function-----------------#
    for n in range(NGQP):
        x = gaussPoints[n]
        for j,val in enumerate(domain):
            phi=1 # empty variable for product
            for m in range(NNPEL):
                if m!=j:
                    phi = phi * (x-domain[m])/(val-domain[m])       
            
            funcLag[j][n] = phi
    
    #--------First Derivative of Interpolation Function--------#
    for n in range(NGQP):
        x = gaussPoints[n]
        for j,val in enumerate(domain):
            phiDiff = 0 # empty variable for summation
            for i in range(NNPEL):
                phi = 1 # empty variable for product
                for m in range(NNPEL):
                    if ((m!=j) and (m!=i)):
                        phi *= (x-domain[m])/(val-domain[m])
                if (i!=j):
                    phiDiff += (1/(val-domain[i]))*phi  
            
            funcLagDiff[j][n] = phiDiff
    
    return funcLag, funcLagDiff

def interpLag(xVal,x):
    '''
    This is a function that directly gives the interpolation function values based on order of polynomial and x values. It has been taken from the classical formula for Lagrange polynomials.
    Differentiated function may not be correct.
    xVal: 1D domain of the function. Array. Length of this array determines the order of polynomial. This array signifies an element in a Finite element mesh. 1 element with two nodes corredpond to 1st order polynomial, 3 nodes/element correspond to second order polynomial and so on.
    x: Value of interpolation function to be calculated at this point. Independent variable. 
    '''
    polyOrder = len(xVal)
    # Empty dictionaries to assign values
    lagInterp = np.zeros(shape=polyOrder, dtype=np.float64) # for interpolation function
    lagInterpDiff = np.zeros(shape=polyOrder, dtype=np.float64) # for first derivative of interpolation function
    for j,val in enumerate(xVal):
        phi = 1 # empty variable for product
        phiDiff = 0 # empty variable for summation
        for i in range(polyOrder):
            if i != j:
                phi = phi * (x-xVal[i])/(val-xVal[i])
                # phiDiff = phiDiff + 1/(x-xVal[i]) # incorrect function
            else:
                continue        
        lagInterp[j] = phi
        # lagInterpDiff[j] = lagInterp[j]*phiDiff

    return lagInterp, lagInterpDiff

# Functions using symbolic library.

def interpLagCalSym(ETYPE=None):
    '''
    This is a function that determines the interpolation function based on number of nodes in an element. Calculations has been done in symbolic form.
    '''
    if ETYPE is None:
        ETYPE=2
        print('Number of nodes per element not specified. Default value of 2 nodes/element is taken.')
    
    # create variables for coefficients
    coeffLagFunc = [sy.symbols(f'a{i}') for i in list(range(ETYPE))] 
    
    # create approximation function from the coefficients
    approxLagFunc = np.polynomial.Polynomial(coeffLagFunc) 
    '''
    Define nth order approximation polynomial. The order is defined by the number of nodes in the element; 2 nodes 1st order, 3 nodes 2nd orde etc. This approximation function is used to determine the solution (for e.g. in Eg3.2 we are trying to solve for temperature distribution). Polynomial defined as - e.g. a+bx or a+bx+cx^2 etc.
    '''

    # Defining Essential Boundary Conditions - to solve for coefficients of the approximation function
    xVal = [sy.symbols(f'x{i}') for i in list(range(ETYPE))] # domain locations
    yVal = [sy.symbols(f'y{i}') for i in list(range(ETYPE))] # generalised displacement (e.g. temperature) at domain locations
    
    # applying EBC to get a set of equations 
    approxEqn = [approxLagFunc(val)-yVal[i] for i,val in enumerate(xVal)]

    # solving the equations symbolically to get the values of coefficients
    coeffSoln = sy.solve(approxEqn, coeffLagFunc)

    # gets keys of the sympy solution dictionary
    coeffSolnkeys = list(coeffSoln.keys())
    
    # get values of coefficients from the keys in a list
    coeffSolnvals = [coeffSoln[coeffSolnkeys[i]] for i,val in enumerate(coeffSolnkeys)]
 
    # substituting solved coefficients in the approx Lagrangian Function
    x = sy.symbols('x') # symbolic variable
    
    # recreating approximation function with known coefficients in symbolic form
    approxLagFuncSoln = np.sum([coeffSolnvals[i]*x**i for i in list(range(ETYPE))])

    # Lagnrangian interpolation function is given as
    interpLagFunc = [sy.diff(approxLagFuncSoln,val) for val in yVal]

    # First derivative of Lagrangian interpolation function is given as
    interpLagFuncDiff = [sy.diff(val,x) for val in interpLagFunc]
    ''' 
    Simpler way would be to compare approxLagFuncSoln with finite element approximation function (which is of form e.g. L0*yVal[0] + L1*yVal[1] for a first order equation). Comparing terms in these two equations would give the interpolation function. However, this was not possible here. So a symbolic polynomial was created (approxLagFuncSoln) and differentiated with respect to yVals to get interpolation functions.
    '''
    return interpLagFunc, interpLagFuncDiff, xVal

