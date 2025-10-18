'''
Author:      Kartikay Shukla
File:        gen_mesh1D.py
Created:     July 25, 2025 
LM:          July 25, 2025

DESCRIPTION
This file contains functions for generating mesh for the given 1D geometry. The code is mainly suitable for uniform mesh size but could also be expanded to accomodate for non uniform sizes. It also has a function for DOF connectivity matrix
'''

import numpy as np

def mesh1D(L, NEL, NNPEL, DOFPN=None):
    '''
    This function creates meshes with local coordinates and global coordinates.
    Also building element connectivity array.
    L = length of beam, ft
    NEL: number of elements
    NNPEL: number of nodes per element
    DOFPN = Degree of freedom per node
    

    Returns a numpy array of array of
    ECON: Element connectivity matrix
    elemGlobalCoord: global nodal coordinates per element
    h: Size of elements. Numpy float64 for uniform element size.
    '''
    
    # creating dictionaries for elements with local node numbers and global coordinates. 
    ECON = np.zeros(shape=[NEL,NNPEL],dtype=np.int32)
    elemGlobalCoord = np.zeros(shape=[NEL,NNPEL],dtype=np.float64)

    h = np.float64(L/NEL) # size of mesh - uniform, float
    dist_bw_nodes = h/(NNPEL-1)

    for i in range(NEL):
        if i==0:
            ECON[i]=np.array(range(NNPEL))
        else:
            ECON[i] = ECON[i-1]+(NNPEL-1)
        elemGlobalCoord[i] = ECON[i]*dist_bw_nodes # uniform mesh
    
    return ECON, elemGlobalCoord

def DOFCON (DOF, NEL, NNPEL, ECON):
    '''
    This function builds the degree of freedom connectivity matrix with respect to elements.
    DOF: Degree of freedom per node
    NEL: Number of elements
    NNPEL: Number of nodes per element
    ECON: Element connectivity matrix

    Returns a numpy array of array of Degree of freedom connectivity matrix
    '''
    eqns_p_elem = NNPEL * DOF
    DOFCON = np.zeros(shape=(NEL,eqns_p_elem), dtype=np.int32)

    # for i in range(NEL):
    #     if i==0:
    #         DOFCON[i] = np.array(range(NNPEL*DOF))
    #     else:
    #         DOFCON[i] = DOFCON[i-1]+(NNPEL-1)*DOF

    for i in range(NEL):
        for j in range(DOF):
            DOFCON[i][j:eqns_p_elem:DOF] = (DOF * ECON[i,:]) + j
           
    return DOFCON

