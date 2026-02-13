'''
Author:      Kartikay Shukla
File:        gen_functions.py
Created:     September 23, 2025 
LM:          September 25, 2025

DESCRIPTION
This file contains general functions for computation in the Finite Element Model.
'''

import json
from tabnanny import check
from .gen_interpFunction import *
from .gen_gaussQuadCalc import *
import numpy as np
from itertools import product, permutations
from scipy.spatial.transform import Rotation
from pathlib import Path

#---------------------------------------------------------------------------------------#
#--------------------------------Permutation Symbol Functions---------------------------#
#---------------------------------------------------------------------------------------#

def permutation_symbol()-> np.ndarray:
    '''
    This function calculates the value of permutation operator. This operator is also known as Levi-Civita symbol.
    For indices i, j, k (with values 0,1,2 for each index), the value of ε is as follows for corresponding index values
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
    '''
    This function modifies the given input into a form that is suitable for calculation. 
    Handles a input data structure where values are defined for all the nodes and all the degrees of freedom in the same dictionary.
    '''
    applied = np.zeros(shape=[NEL, NNPEL, DOF//2], dtype=np.float64)
    
    for i, row in enumerate(ECON):
        for j, elemNode in enumerate(row):
            # input handling function for applied force
            if elemNode in inApplied['globalNode#']:
                index = np.where(inApplied['globalNode#']==elemNode)[0][0] # (0,0) because assumed global node number do not repeat
                for k, key in enumerate(list(inApplied.keys())[1:]):
                    applied[i,j,k] = inApplied[key][index]
    
    return applied

def count(ECON, inpNodes):
    '''
    This function counts the nodes for which the parameter has been given (input).
    This parameter can be boundary condition, applied force or applied moment.
    '''

    count = 0
    for row in ECON:
        for elemNode in row:
            for inpNode in inpNodes:
                if elemNode == inpNode:
                    count +=1

    return count

def values(NEL, ECON, parameter):
    '''
    This function modifies the given input parameter into an dictionary form which is sparsly populated.
    This parameter can be used for boundary conditions, applied force or applied moment.
    The input is a dictionary of form {'DOF':{'nodes':[], 'values':[]}}
    It also uses the count function defined above.
    NEL: number of elements
    ECON: element connectivity matrix
    inpNodes: array of nodes for which the value has been defined
    inpValues: array valus of a parameter at correspoding nodes.
    Returns 
    out: dictionary 
        'pointer': It is an array defined in such a way that the difference of the array value at enxt index with current index gives the total number of given values for the element (given by the current index). 
                    The value of the pointer at the current index is the index for 'nodes' and 'values' Its size if number of elements +1.
        'nodes': numpy array of size same as the number of nodes for which the parameter has been defined (given by count()). It stores the element level nodal number at a particular index number. This index number is the value of 'pointer' at the particular element.
        'values': numpy array of size given by count(). It stores the value of a parameter at a particular index number. This index number is the value of 'pointer' at the particular element.
    '''

    inpNodes, inpValues = parameter['nodes'], parameter['values']
    counter = count(ECON, inpNodes)

    # memory allocation
    out = {'pointer': np.zeros(NEL+1, dtype=np.int32),
            'nodes': np.zeros(counter, dtype=np.int32),
            'values': np.zeros(counter, dtype=np.float64)
            }

    cnt = 0

    for i, row in enumerate(ECON):
        out['pointer'][i+1] = out['pointer'][i]
        for j, elem_node in enumerate(row):
            for node, value in zip(inpNodes, inpValues):
                if elem_node == node:
                    out['pointer'][i+1] += 1 
                    out['nodes'][cnt] = j
                    out['values'][cnt] = value
                    cnt += 1 # increment counter
    
    return out        

def appGenForce(applied, DOF, NNPEL, NEL, ECON):
    '''
    This function modifies the given input into a form that is suitable for calculation. 
    Handles an input data structure similar to that processed by count() and values() above.
    The input is a dictionary of form {'DOF':{'nodes':[], 'values':{}}} which is handled by values() to return a dictionary of form {'pointer':[], 'nodes':[], 'values':[]}
    applied: input dictionary of form {'DOF':{'nodes':[], 'values':{}}}
    '''
    genDistForce = np.zeros(shape=[NEL, NNPEL, DOF//2], dtype=np.float64) # memory alocaiton

    distLoad = {f'{key}': values(NEL, ECON, applied[key]) for key in applied.keys()}
    
    for j in range(DOF//2):
        for i in range(NEL):
            numLoads = distLoad[f'{j}']['pointer'][i+1]-distLoad[f'{j}']['pointer'][i]
            if numLoads == 0:
                genDistForce[i,:,j] = 0.0
            else:
                for numLoad in range(numLoads):
                    cnt = distLoad[f'{j}']['pointer'][i] + numLoad
                    elemNode = distLoad[f'{j}']['nodes'][cnt]
                    genDistForce[i,elemNode,j] = distLoad[f'{j}']['values'][cnt]
    
    return genDistForce


#---------------------------------------------------------------------------------------#
#----------------Error Calculator-------------------------------------------------------#
#---------------------------------------------------------------------------------------#         

def calcErrorI(changeConfig, globalNodes):
        '''
        This function calculates the error based on the change in current configuraiton and normalises with respect to global number of nodes.
        This function will help determine the convergence of the solution.
        This value is often used to determine if a numerical solution has converged to a stable state
        changeConfig: dictionary of np.ndarray. Has keys from '0' to '5' with first 3 for translational DOF and remaining for rotational DOF.

        error: np.float64
        '''
        error = 0.0
        error = np.max([np.linalg.norm(changeConfig[key])/globalNodes for key in changeConfig.keys()])
        # error = np.linalg.norm(changeConfig['0'])/globalNodes

        return error

def calcErrorII(changeConfig, newConfig, direction):
        '''
        This function calculates the error based on the change in current configuraiton and new configuraiton along a specified direction.
        Direction can take values from 0, 1, 2 for translation along X, Y and Z axes or 3, 4, 5 for rotaiton about X, Y and Z axes.
        This value is often used to determine if a numerical solution has converged to a stable state
        changeConfig: dictionary of np.ndarray. Has keys from '0' to '5' with first 3 for translational DOF and remaining for rotational DOF.
        newConfig: dictionary of np.ndarray. Similar structure as config.
        direction: notation for identifying degree of freedom with respect to an axis.

        error: np.float64
        '''
        error = 0.0
        error = np.linalg.norm(changeConfig[f'{direction}'])/np.linalg.norm(newConfig[f'{direction}'])

        return error

def lengthCheck(globalNodes, config):
    '''
    This function calculates the length of the beam after deformation. This number can be compared with the initial length.
    Assumed linear interpolation between points.
    
    globalNodes: Total number of nodes. Tranlation vector is defined for each node.
    config: configuration of beam for each degree of freedom at each global node.

    Output:
    length: length of beam after deformation.
    '''
    length = 0
    for i in range(1, globalNodes):
        X = config['0'][i] - config['0'][i-1]
        Y = config['1'][i] - config['1'][i-1]
        Z = config['2'][i] - config['2'][i-1]

        transVec = np.array([X,Y,Z], dtype=np.float64)
        length += np.linalg.norm(transVec)

    return length

def rotVecCheck(rotVector):
    '''
    This function checks if the direction of axial vector will been flipped due to SciPy Rotation function.
    This happens when updating configuration of the rotational degree of freedom using previous configuratin and change in configuration.
    When the updated rotation vector has magnitude >= pi, the Rotation function reduces the magnitude to magnitude-pi and flips the direction of rotation vector. This way it represents the same way rotation but direction reversed.
    
    If direction will be flipped (due to magnitude of new vector>pi), this function corrects the direction and rotation angle.
    If no, the function returns the same rotation vector.
    
    rotVector: dictionary of numpy arrays, contains previous, change and new configuration of beam at a particular node for rotational degrees of freedom.

    Output:
    rotVectorCorr: numpy array of shape 3, contains rotation vector representing new configuration of beam at a particular node.
    '''
    rotVectorCorr = np.zeros(shape=3, dtype=np.float64) # rotation vector in new configuration
    normPrevRotVec = np.linalg.norm(rotVector['previous'])
    normChangeRotVec = np.linalg.norm(rotVector['change'])
    normNewRotVec = np.linalg.norm(rotVector['new'])
    
    normCheck = normPrevRotVec + normChangeRotVec # use to check if magnitude >pi or not.
    print('before modification','normPrevRotVec=',normPrevRotVec, 'normChangeRotVec=',normChangeRotVec, 'normNeewRotVec=',normNewRotVec, 'normCheck=',normCheck,'normCorrNewRotVec=')    
    if (normCheck >= np.pi) and (normCheck < 2*np.pi):
        
        rotVectorCorr[:] = -1 * (normCheck) * (rotVector['new']/normNewRotVec) # increase the magnitude by pi and flip the direction of axial vector.
        print('rotVector[new]=',rotVector['new'])
        print('after modification','normCorrNewRotVec=',np.linalg.norm(rotVectorCorr), 'rotVecCorr=', rotVectorCorr)
    else:
        rotVectorCorr[:] = rotVector['new'] # return same vector if new configuration magnitude less than pi.
    # rotVectorCorr[:] = rotVector['new'] # return same vector if new configuration magnitude less than pi.    

    return rotVectorCorr

def interpRotVec(Sarray, rotVecArray):
    '''
    This function interpolates net (total) rotation vectors using given interpolation functions. 
    It uses a reference rotation matrix to interpolate locally within an element and then uses the transpose of reference rotation matrix to get global interpolated rotation vectors.
    See paper by Crisfield and Jelenic, 1998 on 'Objectivity of strain measures in geometrically exact 3-D beam theory and it FE implementation.
    Also Algorithm 1 in paper 'Interpolation of rotational variables in nonlinear dynamics of 3D beams' by Jelenic, Crisfield 1998
    
    Sarray: numpy array of shape (NNPEL) containing interpolation function values at a particular Gauss point.
    rotVecArray: numpy array of shape (NNPEL, 3) containing rotation vectors at each node of the element.

    Output:
    rotVecGP: numpy array of shape (3) containing interpolated rotation vector at the Gauss point.
    '''

    NNPEL = rotVecArray.shape[0] # determine numnber of nodes in an element from the shape of input rotVecArray

    refRotMat = rotTensor(rotVecArray[0])

    # Memory allocations
    rotVecGP = np.zeros(shape=3, dtype=np.float64)
    rotMatGP = np.zeros(shape=[3,3], dtype=np.float64)
    rotVecNat = np.zeros(shape=[NNPEL,3], dtype=np.float64) # array to store rotation vectors at each node in an element in natural coordinates
    rotMatNat = np.zeros(shape=[NNPEL,3,3], dtype=np.float64) # array to store rotation tensors at each node in an element in natural coordinates
    
    for i in range(NNPEL):
        rotMatNat[i,:,:] = refRotMat.T @ rotTensor(rotVecArray[i,:])
        rotVecNat[i,:] = rotVector(rotMatNat[i])

    rotVecGPNat = np.dot(Sarray, rotVecNat)
    rotMatGP[:,:] = refRotMat @ rotTensor(rotVecGPNat)
    rotVecGP[:] = rotVector(rotMatGP)

    return rotVecGP, rotMatGP

def interpIncRotVec(Sarray, diffSarray, prevRotVecArray, newRotVecArray):
    '''
    This function interpolates incremental rotation vectors using given interpolation functions. 
    See paper by Crisfield and Jelenic, 1998 on 'Objectivity of strain measures in geometrically exact 3-D beam theory and it FE implementation.
    
    Sarray: numpy array of shape (NNPEL) containing interpolation function values at a particular Gauss point.
    Sarray: numpy array of shape (NNPEL) containing first derivative of interpolation function values at a particular Gauss point.
    prevRotVecArray: numpy array of shape (NNPEL, 3) containing rotation vectors at each node of the element, in previous configuration.
    newRotVecArray: numpy array of shape (NNPEL, 3) containing rotation vectors at each node of the element, in current configuration.


    Output:
    newRotVecGP: numpy array of shape (3) containing interpolated rotation vector at the Gauss point.
    '''

    NNPEL = prevRotVecArray.shape[0] # determine numnber of nodes in an element from the shape of input rotVecArray
    c = 0.5 # fixed constant for scaling reference rotation vector
    I = int(0.5 * (NNPEL+1)) - 1 # chosen node I for calculating reference rotation matrix, -1 at the end since pytho index values start from 0
    J = int(0.5 * (NNPEL+2)) - 1 # chosen node J for calculating reference rotation matrix, -1 at the end since pytho index values start from 0

    prevRotMatI = rotTensor(prevRotVecArray[I])
    prevRotMatJ = rotTensor(prevRotVecArray[J])
    prevRotVecIJ = rotVector(prevRotMatI.T @ prevRotMatJ)

    prevRefRotMat = prevRotMatI @ rotTensor(c * prevRotVecIJ)

    newRotMatI = rotTensor(newRotVecArray[I])
    newRotMatJ = rotTensor(newRotVecArray[J])
    newRotVecIJ = rotVector(newRotMatI.T @ newRotMatJ)

    newRefRotMat = newRotMatI @ rotTensor(c * newRotVecIJ)

    # Memory allocations
    newRotVecGP = np.zeros(shape=3, dtype=np.float64) # calculated rotation vector at gauss point in new/current configuration
    newRotMatGP = np.zeros(shape=[3,3], dtype=np.float64) # calculated rotation tensor at gauss point in new/current configuration
    diffNewRotVecGP = np.zeros(shape=3, dtype=np.float64) # calculated first derivative rotation vector at gauss point in new/current configuration
    diffNewRotMatGP = np.zeros(shape=[3,3], dtype=np.float64) # calculated first derivativerotation tensor at gauss point in new/current configuration
    
    prevRotMatNat = np.zeros(shape=[NNPEL,3,3], dtype=np.float64) # array to store rotation tensors at each node in an element in natural coordinates, in previous configuration
    prevRotVecNat = np.zeros(shape=[NNPEL,3], dtype=np.float64) # array to store rotation vectors at each node in an element in natural coordinates, in previous configuration
    newRotMatNat = np.zeros(shape=[NNPEL,3,3], dtype=np.float64) # array to store rotation tensors at each node in an element in natural coordinates, in current/new configuration
    changeRotMatNat = np.zeros(shape=[NNPEL,3,3], dtype=np.float64) # array to store incremental rotation tensors at each node in an element in natural coordinates
    changeRotVecNat = np.zeros(shape=[NNPEL,3], dtype=np.float64) # array to store incremental rotation vectors at each node in an element in natural coordinates
    
    # calculate local incremental rotation vectors
    for i in range(NNPEL):
        prevRotMatNat[i,:,:] = prevRefRotMat.T @ rotTensor(prevRotVecArray[i,:])
        newRotMatNat[i,:,:] = newRefRotMat.T @ rotTensor(newRotVecArray[i,:])
        
        changeRotMatNat[i,:,:] = newRotMatNat[i] @ prevRotMatNat[i].T
        changeRotVecNat[i,:] = rotVector(changeRotMatNat[i])

        prevRotVecNat[i,:] = rotVector(prevRotMatNat[i])
    # interpolate incremental rotation vector at Gauss point
    prevRotVecGPNat = np.dot(Sarray, prevRotVecNat)
    changeRotVecGPNat = np.dot(Sarray, changeRotVecNat)
    newRotMatGP[:,:] = newRefRotMat @ rotTensor(changeRotVecGPNat) @ prevRefRotMat.T @ rotTensor(prevRotVecGPNat)
    newRotVecGP[:] = rotVector(newRotMatGP)
    # interpolate first derivative of incremental rotation vector at Gauss point
    changeDiffRotVecGPNat = np.dot(diffSarray, changeRotVecNat)
    diffNewRotMatGP[:,:] = newRefRotMat @ rotTensor(changeDiffRotVecGPNat) @ prevRefRotMat.T @ rotTensor(prevRotVecGPNat)
    diffNewRotVecGP[:] = rotVector(diffNewRotMatGP)

    return newRotVecGP, diffNewRotVecGP

#---------------------------------------------------------------------------------------#
#----------------Write data to files----------------------------------------------------#
#---------------------------------------------------------------------------------------#    

def write_toJSON(data, name, direc):
    '''
    Write output results to JSON file for post-processing and manual checks for every loadstep and iteration.
    It has another function that converts numpy arrays to list for JSON serialization - code from ChatGPT.

    data: dictionary of numpy arrays containing results to be written to excel file.
    name: name of the output file
    direc: directory where results are to be written.
    
    '''
    outDir = Path(direc)
    outDir.mkdir(parents=True, exist_ok=True)
    outFile = outDir/f'{name}.json'

    def to_serializable(obj): 
        if isinstance(obj, np.ndarray): 
            return obj.tolist() 
        if isinstance(obj, (np.integer, np.floating)): 
            return obj.item() 
        if isinstance(obj, dict): 
            return {str(k): to_serializable(v) for k, v in obj.items()} 
        if isinstance(obj, (list, tuple)): 
            return [to_serializable(x) for x in obj] 
        return obj

    with outFile.open(mode="w", encoding="utf-8") as write_file:
        json.dump(to_serializable(data), write_file, indent=2)

#---------------------------------------------------------------------------------------#
#----------------Quaternion functions----------------------------------------------------#
#---------------------------------------------------------------------------------------#    

def quatMultiplcation(quat1: np.ndarray, quat2: np.ndarray)-> np.ndarray:
    '''
    This function multiplies two quaternions and returns the resulting quaternion.
    quat1: numpy array of shape (4,) containing the four components of the first quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.
    quat2: numpy array of shape (4,) containing the four components of the second quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.

    Returns:
    quatResult: numpy array of shape (4,) containing the four components of the resulting quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.
    '''
    quatResult = np.zeros(shape=4, dtype=np.float64)

    quatResult[0] = quat1[0]*quat2[0] - np.dot(quat1[1:], quat2[1:])
    quatResult[1:] = quat1[0]*quat2[1:] + quat2[0]*quat1[1:] + np.cross(quat1[1:], quat2[1:])

    return quatResult

def rotVecToQuat(rotVec: np.ndarray)-> np.ndarray:
    '''
    This function converts a given rotation vector to a quaternion.
    rotVec: numpy array of shape (3,) containing the three components of the rotation vector.
    Refer Simo's paper on 'A Three-Dimensional Finite Strain Rod Model. Part II: Computational Aspects' 1986 for the formula used in this function.

    Returns:
    quat: numpy array of shape (4,) containing the four components of the quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.
    '''
    quat = np.zeros(shape=4, dtype=np.float64)

    normRotVec = np.linalg.norm(rotVec, ord=2)
    if normRotVec > 1E-15:    
        quat[0] = np.cos(normRotVec/2)
        quat[1] = np.sin(normRotVec/2) * rotVec[0] / normRotVec
        quat[2] = np.sin(normRotVec/2) * rotVec[1] / normRotVec
        quat[3] = np.sin(normRotVec/2) * rotVec[2] / normRotVec
        
    else:
        quat[0] = 1.0
        quat[1:] = 0.0
    
    mask = np.abs(quat) < 1E-15
    if np.any(mask):
        quat[mask] = 0.0
    
    # # Hemisphere check
    # refQuat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # reference quaternion for hemisphere check
    # dotProduct = np.dot(quat, refQuat)
    # if dotProduct < 0:
    #     quat[:] = -1 * quat # flip the sign of quaternion to ensure it lies in the same hemisphere as the reference quaternion.
    # scalar identity check
    magQuat = np.linalg.norm(quat, ord=2)
    if not np.isclose(magQuat, 1.0):
        print(f'Warning: Magnitude of quaternion is not 1.0. It is {magQuat}')
    
    return quat

def quatToRotVec(quart: np.ndarray)-> np.ndarray:
    '''
    This function converts a given quaternion to a rotation vector.
    Refer Simo's paper on 'A Three-Dimensional Finite Strain Rod Model. Part II: Computational Aspects' 1986 for the formula used in this function.

    quart: numpy array of shape (4,) containing the four components of the quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.

    Returns:
    rotVec: numpy array of shape (3,) containing the three components of the rotation vector.
    '''
    rotVec = np.zeros(shape=3, dtype=np.float64)

    q0 = quart[0] # scalar part of quaternion
    qVec = np.array(quart[1:], dtype=np.float64) # vector part of quaternion

    normQVec = np.linalg.norm(qVec, ord=2)
    
    if normQVec > 1E-15:
        rotationAngle = 2 * np.arccos(q0)
        rotVec[:] = rotationAngle * (qVec/normQVec)
    else:
        rotVec[:] = 0.0
    
    mask = np.abs(rotVec) < 1E-15
    if np.any(mask):
        rotVec[mask] = 0.0
    
    return rotVec

def quatToRotMat(quat: np.ndarray)-> np.ndarray:
    '''
    This function converts a given quaternion to a orthogonal rotation tensor.
    Refer Simo's paper on 'A Three-Dimensional Finite Strain Rod Model. Part II: Computational Aspects' 1986 for the formula used in this function.
    
    quat: numpy array of shape (4,) containing the four components of the quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.
    
    Returns:
    rotMat: numpy array of shape (3,3) containing the components of the orthogonal rotation tensor.
    '''
    rotMat = np.zeros(shape=[3,3], dtype=np.float64)
    symmRotMat = np.zeros(shape=[3,3], dtype=np.float64)
    unsymmRotMat = np.zeros(shape=[3,3], dtype=np.float64)

    q0 = quat[0] # scalar part of quaternion
    qVec = np.array(quat[1:], dtype=np.float64) # vector part of quaternion

    # symmRotMat = np.outer(qVec, qVec) + (np.square(q0)-0.5) * np.identity(3)
    # print('symmRotMat=', symmRotMat)
    # unsymmRotMat[:,:] = q0 * skewSymmMat(qVec)
    # print('unsymmRotMat=', unsymmRotMat)
    # rotMat[:,:] = 2 * (symmRotMat + unsymmRotMat)

    rotMat[0,0] = q0**2 +qVec[0]**2 - 0.5
    rotMat[1,1] = q0**2 +qVec[1]**2 - 0.5
    rotMat[2,2] = q0**2 +qVec[2]**2 - 0.5

    rotMat[0,1] = qVec[0]*qVec[1] - q0*qVec[2]
    rotMat[1,0] = qVec[0]*qVec[1] + q0*qVec[2]

    rotMat[0,2] = qVec[0]*qVec[2] + q0*qVec[1]
    rotMat[2,0] = qVec[0]*qVec[2] - q0*qVec[1]

    rotMat[1,2] = qVec[1]*qVec[2] - q0*qVec[0]
    rotMat[2,1] = qVec[1]*qVec[2] + q0*qVec[0]

    rotMat[:,:] = 2 * rotMat

    mask = np.abs(rotMat) < 1E-15
    if np.any(mask):
        rotMat[mask] = 0.0
    
    detRotMat = np.linalg.det(rotMat)
    checkDet = np.abs(detRotMat)-1 < 1E-10 # check if determinant is close to 1 or not. It can be -1 for improper rotation but we are only interested in proper rotation with determinant 1.

    if detRotMat < 0 :
        print('Warning: Determinant of rotation matrix is negative. Check the input quaternion.')
        print('quat=',quat)
    elif not checkDet:
        print('Warning: Determinant of rotation matrix is not close to 1. Check the input quaternion.')

    checkIdentity = rotMat @ rotMat.T
    mask1 = np.abs(checkIdentity - np.identity(3)) < 1E-10
    if not np.all(mask1):
        print('Warning: Rotation matrix is not orthogonal. Check the input quaternion.')
    
        
    return rotMat

def rotMatToQuat(rotMat: np.ndarray)-> np.ndarray:
    '''
    This function converts a given orthogonal rotation tensor to a quaternion.
    It follows Spurrier's alhorithm for extracting quaternion from rotation matrix. It is a more robust algorithm than the one used in SciPy Rotation function for extracting quaternion from rotation matrix.
    It can handle cases rotation angle is very close to (2n-1)pi where n=1,2,3... and the sine or cosine functions return zero value.
    Refer Simo's paper on 'A Three-Dimensional Finite Strain Rod Model. Part II: Computational Aspects' 1986 for the formula used in this function.
    
    rotMat: numpy array of shape (3,3) containing the components of the orthogonal rotation tensor.
    
    Returns:
    quat: numpy array of shape (4,) containing the four components of the quaternion in the order (q0, q1, q2, q3) where q0 is the scalar part and (q1, q2, q3) are the vector part.
    '''
    quat = np.zeros(shape=4, dtype=np.float64)
    
    traceRotMat = np.trace(rotMat)
    checkList = np.array([traceRotMat, rotMat[0,0], rotMat[1,1], rotMat[2,2]])
    M = np.max(checkList)
    if M == traceRotMat:
        quat[0] = 0.5 * np.sqrt(1 + traceRotMat)
        quat[1] = (rotMat[2,1] - rotMat[1,2])/(4 * quat[0])
        quat[2] = (rotMat[0,2] - rotMat[2,0])/(4 * quat[0])
        quat[3] = (rotMat[1,0] - rotMat[0,1])/(4 * quat[0])
    else:
        index = np.argmax(checkList)
        if index == 1:
            quat[1] = np.sqrt(0.5 * M + 0.25 *(1-traceRotMat))
            quat[0] = (rotMat[2,1] - rotMat[1,2])/(4 * quat[1])
            quat[2] = (rotMat[0,1] + rotMat[1,0])/(4 * quat[1])
            quat[3] = (rotMat[0,2] + rotMat[2,0])/(4 * quat[1])
        elif index == 2:
            quat[2] = np.sqrt(0.5 * M + 0.25 *(1-traceRotMat))
            quat[0] = (rotMat[0,2] - rotMat[2,0])/(4 * quat[2])
            quat[1] = (rotMat[0,1] + rotMat[1,0])/(4 * quat[2])
            quat[3] = (rotMat[1,2] + rotMat[2,1])/(4 * quat[2])
        elif index == 3:
            quat[3] = np.sqrt(0.5 * M + 0.25 *(1-traceRotMat))
            quat[0] = (rotMat[1,0] - rotMat[0,1])/(4 * quat[3])
            quat[1] = (rotMat[0,2] + rotMat[2,0])/(4 * quat[3])
            quat[2] = (rotMat[1,2] + rotMat[2,1])/(4 * quat[3])
    
    # # Hemisphere check
    # refQuat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # reference quaternion for hemisphere check
    # dotProduct = np.dot(quat, refQuat)
    # if dotProduct < 0:
    #     quat[:] = -1 * quat # flip the sign of quaternion to ensure it lies in the same hemisphere as the reference quaternion.
    
    # scalar identity check
    magQuat = np.linalg.norm(quat, ord=2)
    if not np.isclose(magQuat, 1.0):
        print(f'Warning: Magnitude of quaternion is not 1.0. It is {magQuat}')
    return quat


def interpQuat(Sarray, quatArray):
    '''
    This function will interpolate quaternions at Gauss points using given interpolation functions.
    It returns rotation vectors at Gauss points by converting interpolated quaternions to rotation vectors.
    It uses a reference rotation matrix to interpolate locally within an element and then uses the transpose of reference rotation matrix to get global interpolated rotation vectors.
    See paper by Crisfield and Jelenic, 1998 on 'Objectivity of strain measures in geometrically exact 3-D beam theory and it FE implementation.
    Also Algorithm 1 in paper 'Interpolation of rotational variables in nonlinear dynamics of 3D beams' by Jelenic, Crisfield 1998

    Sarray: numpy array of shape (NNPEL) containing interpolation function values at a particular Gauss point.
    quatArray: numpy array of shape (NNPEL, 4) containing quaternion parameters at each node of the element.

    Returns
    rotVecGP: numpy array of shape (3) containing interpolated rotation vector at the Gauss point.
    rotMatGP: numpy array of shape (3,3) containing interpolated rotation matrix at the Gauss point.
    '''
    NNPEL = quatArray.shape[0] # determine numnber of nodes in an element from the shape of input quatArray
    
    rotVecGP = np.zeros(shape=3, dtype=np.float64)
    quatGP = np.zeros(shape=4, dtype=np.float64) # interpolated quaternion at Gauss point
    rotMatGP = np.zeros(shape=[3,3], dtype=np.float64)
    
    rotVecNat = np.zeros(shape=[NNPEL,3], dtype=np.float64) # array to store rotation vectors at each node in an element in natural coordinates
    quatNat = np.zeros(shape=[NNPEL,4], dtype=np.float64) # array to store rotation vectors at each node in an element in natural coordinates
    rotMatNat = np.zeros(shape=[NNPEL,3,3], dtype=np.float64) # array to store rotation tensors at each node in an element in natural coordinates
    
    refQuat = quatArray[0] # reference quaternion at node I
    refRotMat = quatToRotMat(refQuat)

    for i in range(NNPEL):  
        rotMatNat[i,:,:] = refRotMat.T @ quatToRotMat(quatArray[i,:])
        quatNat[i,:] = rotMatToQuat(rotMatNat[i])
        # check = np.dot(refQuat, quatNat[i]) # hemisphere check
        # if check < 0: quatNat[i] = -quatNat[i]
        rotVecNat[i,:] = quatToRotVec(quatNat[i])

    rotVecGPNat = np.dot(Sarray, rotVecNat)
    quatGPNat = rotVecToQuat(rotVecGPNat) # scalar identity check in the function itself. Ensures good conditioning to avoid numerical issues.

    # rotMatGP[:,:] = refRotMat @ quatToRotMat(quatGPNat)
    # quatGP[:] = rotMatToQuat(rotMatGP)
    # rotVecGP[:] = quatToRotVec(quatGP)

    rotMatGP[:,:] = refRotMat @ rotTensor(rotVecGPNat)
    rotVecGP[:] = rotVector(rotMatGP)
    return rotVecGP, rotMatGP
            
    

    
