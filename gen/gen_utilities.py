'''
Author:      Kartikay Shukla
File:        gen_functions.py
Created:     September 23, 2025 
LM:          September 25, 2025

DESCRIPTION
This file contains general functions for computation in the Finite Element Model.
'''

from .gen_interpFunction import *
from .gen_gaussQuadCalc import *
import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation

#---------------------------------------------------------------------------------------#
#--------------------------------Permutation Symbol Functions---------------------------#
#---------------------------------------------------------------------------------------#

def permutation_symbol()-> np.ndarray:
    '''
    This function calculates the value of permutation operator.
    For indices i, j, k (with values 0,1,2 for each index), the value of ε  is as follows for corresponding index values
    ε = 1 for (012), (120), (201)
    ε = -1 for (021), (210), (102)
    ε = 0 for for all other index values

    Returns
    permutationSym = 3x3x3 array of values of permutation operator for index value i, j, k
    '''

    # memory allocation
    permutationSym = np.zeros(shape=[3,3,3], dtype=np.float64)
    
    # values of indices i, j, k
    indexVal = [0,1,2]

    for i,j,k in product(indexVal, indexVal, indexVal):
        permutationSym[i,j,k] = 0.5 * ((i-j) * (j-k) * (k-i))

    return permutationSym

def permutationSymbol(i,j,k: np.float64)-> np.float64:
    '''
    This function returns the value of permutation operator for given index values i, j, k.
    ε = 1 for (012), (120), (201)
    ε = -1 for (021), (210), (102)
    ε = 0 for for all other index values

    i: index i values range from 0,1,2
    j: index j values range from 0,1,2
    k: index k values range from 0,1,2

    Returns:
    permutationSym = value of permutation operator for index value i, j, k. np.float64
    '''
    indexVal = np.array([0,1,2], dtype=np.float64)

    if i in indexVal and j in indexVal and k in indexVal:
        permutationSym = 0.5 * ((i-j) * (j-k) * (k-i))
    else:
        print('Values of index should 0,1 or 2.')

    return permutationSym

#---------------------------------------------------------------------------------------#
#---------------------Rotation mapping functions----------------------------------------#
#---------------------------------------------------------------------------------------#

def skewSymmMat(vector: np.ndarray)-> np.ndarray:
    '''
    This function computes the skew symmeteric matrix from the given axial vector.

    input:
    vector: axial vector in space having three coordinates based on the axes x, y and z. np.ndarray
    output:
    skewMat: skew symmetrix matrix representation of the vector. np.ndarray
    '''

    # memory allocaiton

    skewMat = np.zeros(shape=[3,3], dtype=np.float64)

    skewMat[0,1], skewMat[0,2], skewMat[1,2] = -vector[2], vector[1], -vector[0]

    skewMat += -skewMat.T

    return skewMat

def rotationVector(skewMat: np.ndarray)->np.ndarray:
    '''
    This function returns a rotation vector from skew symmeteric rotation tensor.

    skewMat: skew symmetric matrix, second order tensor, matrix form

    Returns
    rotVec: numpy array of with terms for each rotation vector
    '''

    rotVec = np.zeros(shape=3, dtype=np.float64)
    rotVec[0], rotVec[1], rotVec[2] = -skewMat[1,2], skewMat[0,2], skewMat[0,1]

    return rotVec

def rotTensor(rotVec: np.ndarray)-> np.ndarray:
    '''
    This function uses SciPy function to calculate the rotation tensor from the given rotation vector.
    For n*pi angles that sine or cosine function returns zero value, the function returns a very small value of the order of E-16 and not zero exactly.

    rotVec: rotation vector. Its a numpy array (3,). The three terms denote rotation about three orthogonal coordinate axes.

    Returns:
    rotationTensor: This is an orthogonal rotation tensor. Its a numpy array (3,3).
    '''

    rotationTensor = Rotation.from_rotvec(rotVec).as_matrix()
    
    return rotationTensor

def rotVector(rotTensor: np.ndarray)-> np.ndarray:
    '''
    This function uses SciPy function to calculate the rotation vector from the given orthogonal rotation tensor.
    For n*pi angles that sine or cosine function returns zero value, the function returns a very small value of the order of E-16 and not zero exactly.

    rotTensor: rotation vector. Its a numpy array (3,3). The three terms denote rotation about three orthogonal coordinate axes.

    Returns:
    rotationVector: This is a rotation vector. Its a numpy array (3,1).
    '''

    rotationVector = Rotation.from_matrix(rotTensor).as_rotvec()
    
    return rotationVector

def rotationTensor_Mat(skewMat: np.ndarray)-> np.ndarray:
    '''
    This function computes the orthogonal rotation tensor from the given skew symmetric rotation tensor.
    It uses exponential map to calculate orthogonal rotation tensor from skew-symmetric matrix.
    Refer to Dr. Greg Payette's (Wells Research Engineer, XOM) notes on Dr. Juan Simo's Paper (Finite Strain beam Formulation Part I and II) or Appendix R - Mathematics of Finite Rotations of Nonlinear FE Method book by Dr. Carlos Fellipe (Retd., Professor).
    Cannot handle 0deg rotation - returns NAN values.
    For other n*pi angles that sine or cosine functions return zero value, the function returns a very small value of the order of E-16 and not zero exactly.

    skewMat: skew symmetric rotation matrix uses axial vector is the rotation vector itself. Its a numpy array (3,3).

    Returns:
    rotationTensor: This is an orthogonal rotation tensor. Its a numpy array (3,3).
    '''

    I = np.identity(3) # identity tensor
    rotVec = rotationVector(skewMat=skewMat) #  rotation vector from skew-symmetric matrix
    normRotVec = np.linalg.norm(rotVec, ord = 2) # Euclidean norm

    rotationTensor = I + (np.sin(normRotVec)/normRotVec)*skewMat + (2*np.square(np.sin(normRotVec/2)/normRotVec))*np.dot(skewMat,skewMat)

    return rotationTensor

def rotationTensor_Vec(rotVec: np.ndarray)-> np.ndarray:
    '''
    This function computes the orthogonal rotation tensor from the given rotation vector.
    It uses exponential map to calculate orthogonal rotation tensor from rotation vector.
    Refer to Dr. Greg Payette's (Wells Research Engineer, XOM) notes on Dr. Juan Simo's Paper (Finite Strain beam Formulation Part I and II) or Appendix R - Mathematics of Finite Rotations of Nonlinear FE Method book by Dr. Carlos Fellipe (Retd., Professor).
    Cannot handle 0deg rotation - returns NAN values. 
    For other n*pi angles that sine or cosine function returns zero value, the function returns a very small value of the order of E-16 and not zero exactly.

    rotVec: rotation vector. Its a numpy array (3,). The three terms denote rotation about three orthogonal coordinate axes.

    Returns:
    rotationTensor: This is an orthogonal rotation tensor. Its a numpy array (3,3).
    '''

    I = np.identity(3) # identity tensor
    skewMat = skewSymmMat(vector=rotVec) #  skew-symmetric matrix from rotation vector
    normRotVec = np.linalg.norm(rotVec, ord = 2) # Euclidean norm

    rotationTensor = I + (np.sin(normRotVec)/normRotVec)*skewMat + (2*np.square(np.sin(normRotVec/2)/normRotVec))*np.dot(skewMat,skewMat)

    return rotationTensor




#---------------------------------------------------------------------------------------#
#-------------Functions to assist with Stiffness matrix calculation---------------------#
#---------------------------------------------------------------------------------------#

def Jacobian(NNPEL:np.float64, NGQP: np.float64, elemGlobalCoord: np.ndarray)->np.ndarray:
    '''
    Function to calculate the jacobian of natural (spatial on the body) to local coordinates(coordinate ranging from -1 to 1 for numerical integration).
    NNPEL: number of nodes per element
    NGQP: number of Gauss Quadrature points    
    elemGlobalCoord: global coordinates of the element in consideration.

    returns:
    numpy array of calculated Jacobian function.
    '''

    interpDiff = interpLagGLQ(NNPEL, NGQP)[1]

    J = np.dot(elemGlobalCoord, interpDiff)

    return J

def outerInterpFunc(NNPEL: np.float64, NGQP:np.float64, n:np.float64, Jac: np.ndarray)-> np.ndarray:
    '''
    This function computes the outer product of Langrangian interpolation functions. This result is useful in calculating the stiffness matrix.
    NNPEl: number of nodes per element
    NGQP: number of Gauss Quadrature points
    n: index of Gauss Quadrature point to be used for evaluation
    jac: jacobian of the natural to local coordinate

    Returns:
    numpy arrays of outer products of Lagrangian Interpolation function.
    '''

    # interpolation functions
    interp, interpDiff = interpLagGLQ(NNPEL, NGQP)

    S0 = np.zeros(shape=NNPEL, dtype=np.float64)
    S1 = np.zeros(shape=NNPEL, dtype=np.float64)

    S00 = np.zeros(shape=[NNPEL,NNPEL], dtype=np.float64)
    S10 = np.zeros(shape=[NNPEL,NNPEL], dtype=np.float64)
    S01 = np.zeros(shape=[NNPEL,NNPEL], dtype=np.float64)
    S11 = np.zeros(shape=[NNPEL,NNPEL], dtype=np.float64)
    
    S0[:] = interp[:,n]
    S1[:] = interpDiff[:,n] * (1/Jac[n])
    
    S00[:] = np.outer(S0,S0)
    S10[:] = np.outer(S1,S0)
    S01[:] = S10.T
    S11[:] = np.outer(S1,S1)

    return S0, S1, S00, S10, S01, S11

def maptoLocal(NNPEL:np.float64, NGQP: np.float64, elemGlobalCoord: np.ndarray)->np.ndarray:
    '''
    This function maps natural coordinates to local coordinates using Lagrangian interpolation function.
    NNPEL: number of nodes per element
    NGQP: number of Gauss Quadrature points    
    elemGlobalCoord: global coordinates of the element in consideration.

    returns:
    numpy array of calculated local coordinates.
    '''

    interp = interpLagGLQ(NNPEL, NGQP)[0]

    func = np.dot(elemGlobalCoord, interp)

    return func

#---------------------------------------------------------------------------------------#
#-------------Input handling functions--------------------------------------------#
#---------------------------------------------------------------------------------------#

def applied(inApplied, ECON, DOF, NNPEL, NEL):

    applied = np.zeros(shape=[NEL, NNPEL, DOF//2], dtype=np.float64)
    
    for i, row in enumerate(ECON):
        for j, elemNode in enumerate(row):
            # input handling function for applied force
            if elemNode in inApplied['globalNode#']:
                index = np.where(inApplied['globalNode#']==elemNode)[0][0] # (0,0) because assumed global node number do not repeat
                for k, key in enumerate(list(inApplied.keys())[1:]):
                    applied[i,j,k] = inApplied[key][index]
    
    return applied
        

#---------------------------------------------------------------------------------------#
#----------------Error Calculator-------------------------------------------------------#
#---------------------------------------------------------------------------------------#         

def calcError(config, newConfig):
        '''
        This function calculates the error between new configuration and current configuration. This function will help determine the convergence of the solution.
        The function returns the cumulative sum, which represents the total normalized error between the two configurations.
        This value is often used to determine if a numerical solution has converged to a stable state
        config: dictionary of np.ndarray. Has keys from '0' to '5' with first 3 for translational DOF and remaining for rotational DOF.
        newConfig: dictionary of np.ndarray. Similar structure as config.

        error: np.float64
        '''
        error = 0.0

        for key in newConfig.keys():
            num = len(newConfig[key])
            error += np.linalg.norm(newConfig[key] - config[key]) / num
        
        return error
