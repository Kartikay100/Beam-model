'''
Author:      Kartikay Shukla
File:        gen_mesh1D.py
Created:     July 25, 2025 
LM:          July 25, 2025

DESCRIPTION
This file contains functions for generating mesh for the given 1D geometry. The code is mainly suitable for uniform mesh size. It also has a function for DOF connectivity matrix
'''

import numpy as np
from gen.gen_interpFunction import spectralNodes

def mesh1DLGL(L, NEL, NNPEL, NGQP=None):
    '''
    This function generates mesh of global coordinates (NEL vs NNPEL array) based on a Gauss-Lobatto-Legendre spectral points.
    This meshing scheme places nodes (and calculates its global coordinates) in an element which are unequally spaced.
    L = length of beam , m
    NEL = number of elements
    NNPEL = number of nodes per elemnt
    NGQP = number of Gauss quadrature points, initial value defined none if NNPEL=NGQP.
    
    Returns a numpy array of array of
    ECON: Element connectivity matrix
    elemGlobalCoord: global nodal coordinates per element
    h: Size of elements. Numpy float64 for uniform element size.
    '''
    if NGQP == None:
        NGQP = NNPEL
    
    # memory allocation for element connectivity matrix and global coordinate matrix
    ECON = np.zeros(shape=[NEL,NNPEL],dtype=np.int32)
    elemEndCoord = np.zeros(shape=[NEL,2],dtype=np.float64) # coordinates of end points of an element
    elemGlobalCoord = np.zeros(shape=[NEL,NNPEL],dtype=np.float64)
    
    domain = spectralNodes(NGQP) # Location of nodes in natural coordinates for spectral basis
    h = np.float64(L/NEL) # size of element - uniform meshing, np.float64
   
    # create element connectivity matrix and elemGlobalCoord matrix
    for i in range(NEL):
        if i==0:
            elemEndCoord[i,0] = 0.0
            elemEndCoord[i,1] = h
            ECON[i,:]=np.array(range(NNPEL))
        else:
            elemEndCoord[i,0] = elemEndCoord[i-1,1]
            elemEndCoord[i,1] = elemEndCoord[i,0] + h
            ECON[i,:] = ECON[i-1]+(NNPEL-1)
        for j in range(NNPEL):
            elemGlobalCoord[i,j] = 0.5 * ( (1-domain[j])*elemEndCoord[i,0] + (1+domain[j])*elemEndCoord[i,1])
    
    return ECON, elemGlobalCoord


def mesh1D(L, NEL, NNPEL, DOFPN=None):
    '''
    This function creates meshes with local coordinates and global coordinates. This function distributes the nodes uniformly over an element.
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
            DOFCON[i,j:eqns_p_elem:DOF] = (DOF * ECON[i,:]) + j
           
    return DOFCON

