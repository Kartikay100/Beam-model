'''
Author:      Kartikay Shukla
File:        gen_glMatAssy.py
Created:     July 21, 2025 
LM:          Aug 26, 2025

DESCRIPTION
This is an FEM Model for heat transfer problem Example 4.5 from the book Introduction to Linear Finite Element Method, Second Edition.
'''

import numpy as np
import scipy as sc
from Eg4_5_elemMat import *
from gen.gen_mesh1D import *
from gen.gen_compCost import *

def globMat(input, NEL, NNPEL, NGQP, DOF = 1, SOEL=None):
    ''''
    Function for assembling global matrix. This function calls for element matrices for each element and assembles them in the global matrix. It does not call/solve for all the element matrices together.
    This is a sparse matrix solver.

    input: output of the function in input_Eg3_2.py. It returns an array of array which needs to be segregated. Check the return of the function to determine the location of different properties.
    q_0 = heat generated inside the body. It is an input.
    NEPL: number of elements . It is an input.
    NNPEL: number of nodes per element. It is an input.
    DOF: Degree of freedom per node
    NGQP: Number of Gauss Quadrature points.
    dispL = transverse displacement of the beam at x=L
    '''

    #-------------------INTIALISE-------------------------------#
    # initialise inputs
    geomProp = input[0] # first element of the input array.
    L = geomProp['L']

    # calling returns of mesh1D function 
    ECON, elemGlobalCoord, h = mesh1D(L, NEL, NNPEL, SOEL)
    dofCON = DOFCON(DOF, NEL, NNPEL, ECON)

    #----------------Global matrix assembly--------------------#
    
    shapeGM = DOF * (NEL * NNPEL - NEL + 1) # total number of FE nodes
    shapeElemMat = DOF * NNPEL # shape of Element matrices, square matrix
    sizeElemMat = shapeElemMat ** 2 # size of Element matrices shapeElemMat x shapeElemMat
    sizeSparGM = sizeElemMat * NEL # size of sparse matrices, size = (shape,1) since vector
    
    # variables for storing vectorised coefficient matrix and column vector
    sparSMG = np.zeros(shape=sizeSparGM, dtype=np.float64)
    SMG = np.zeros(shape=[shapeGM,shapeGM], dtype=np.float64)
    CVG =  np.zeros(shape=shapeGM, dtype=np.float64)
    
    # variables for storing index locations of coefficient matrix
    sparIIrow = np.zeros(shape=sizeSparGM, dtype=np.float64)
    sparJJcol = np.zeros(shape=sizeSparGM, dtype=np.float64)
    
    # counter, for index position of vectorised matrix.
    m = 0 
    
    # Assembling element matrices to form global matrices
    for i in range(NEL): # loop over elements

        SME, CVE = elemMatBeamRIE(input, NEL, NNPEL, DOF, NGQP, i+1, SOEL)
        # Global stiffness matrix assembly
        sparSMG[m:m+sizeElemMat] = SME.flatten() # value
        sparIIrow[m:m+sizeElemMat] = np.repeat(dofCON[i],shapeElemMat) # row index
        sparJJcol[m:m+sizeElemMat] = np.tile(dofCON[i],shapeElemMat) # column index
        m += sizeElemMat
        
        # global column vector assembly
        CVG[dofCON[i]] += CVE
   
    # create global sparse stiffness matrix from vectorised matrix
    SMG = sc.sparse.coo_array((sparSMG, (sparIIrow,sparJJcol)), shape=(shapeGM,shapeGM)).tocsr()
    
    # solve the linear algebra problem using sparse solver
    solution = sc.sparse.linalg.spsolve(SMG, CVG)

    return elemGlobalCoord, solution

