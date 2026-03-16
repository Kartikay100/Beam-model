'''
Author:      Kartikay Shukla
File:        boundary.py
Created:     October 8, 2025 
LM:          February 20, 2026

Description
This file contains function for handling boundary input from JSON file and applying boundary conditions to the element matrices.
'''
import numpy as np
from gen.gen_utilities import rotTensor, rotVector
 
class boundary():
    '''
    This class contains functions to handle and prepare boundary conditions to be applied to stiffness matrix or column vector.
    The input is of form 
        inputEBC = {'globalNode#': [[0,8], [2,2]], # [node number], [no of boundary values at node]
            'Values': [[0, 0],[1,0],[0,0],[1,0]] # [local DOF #, value]
        }
        inputMBC = {'globalNode#': [[4], [1]], # [node number], [no of boundary values at node]
        'Values': [[1, 0, 0, 0]], # [DOF#, refU, beta, betaU]
        }
        where for 'globalNode#' contains array with global node # and number of DOF with boundary conditions at those nodes respectively. 'Values' store local DOF # and value sequentially for all DOF with boundary values in order.
        This same input form is used for natural BC and mixed BC.
        Mixed BC has a slightly different form – this BC is of form (beta + betaU*U)*(U-refU). 'Values' stores local DOF #, refU, beta and betaU in order.
    Once the input has been read in this form, initialise boundary class object, call boundaryHandler(). It returns a dictionary which is used by applyEBC(), applyNBC() to apply boundary conditions to matrices and vectors respectively.
    applyMBC() uses the dictionary in a slightly different form and returns values which is used in element level calculations.
    '''
    def __init__(self, DOFPN, ECON, boundaryCond):
        '''
        Class constructor.
        ECON: Element connectivity matrix
        DOF: Degree of freedom per node
        inputBoundary: input boundary condition. In the form of array of global nodes and boundary values.
            Global nodes has an array of global node numbers and an array of number of boundary values at each global node. 
            Boundary values has an array of local DOF number and boundary value for that DOF at specified nodes sequentially.
        '''
        self.DOF = DOFPN
        self.ECON = ECON
        self.boundaryCond = boundaryCond

    def boundaryHandler(self, type:str) -> dict[str, float]:
        '''
        This function reads boundary conditions in terms of array of global nodes and boundary values and creates another array of Elem #, local node number and boundary value.
        Returns a dictionary of boundary conditions with element number, local DOF number and corresponding boundary value.

        type: type of boundary condition it is processing – essential, natural or mixed. 
        '''
        BC = {'Elem':[],
            'localDOF#':[],
            'Values':[]}
        
        nodes = np.array(self.boundaryCond['globalNode#'])[0] # node numbers
        numOfValues = np.array(self.boundaryCond['globalNode#'])[1] # number of boundary values at each node
        values = np.array(self.boundaryCond['Values']) # boundary values at specififed DOF
        
        count = 0 # counter to keep track of number of boundary values assigned at each node, used for indexing values array
        for n, node in enumerate(nodes):
            index = np.where(self.ECON==node) # returns tuple of arrays - array of row indices and column indices
            elemNum = index[0]
            localNodeNum = np.array(index[1])
            numbers = numOfValues[n] # number of boundary values at node n
            for value in values[count : count+numbers]: # loop over number of boundary values at node n
                localNumDOF = localNodeNum * self.DOF + value[0] # calculating local DOF number based on local node number and DOF per node.
                boundVal = value[1:] # boundary value for this DOF
                BC['Elem'].extend(elemNum.tolist())
                BC['localDOF#'].extend(localNumDOF.tolist())
                if type == 'NBC':
                    boundVal = [i/len(elemNum) for i in boundVal]
                BC['Values'].extend([boundVal] * len(elemNum))
            count += numbers
        BC['Elem'] = np.array(BC['Elem'])
        BC['localDOF#'] = np.array(BC['localDOF#'])
        BC['Values'] = np.array(BC['Values'])

        return BC
    
    def applyEBC(self, i, iter, SME, CVE, EBC):
        '''
        Apply essential boundary condition to element level stiffness matrices and column vector.
        i = element #.
        iter= iteration #.
        Returns
        SME = element level stiffness matrix, np.ndarray.
        CVE = element level column/ force vector, np.ndarray
        '''          

       #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if i in EBC['Elem']: # '0' for array of global nodes with boundary condition
            index = np.where(EBC['Elem']==i)[0] # [0] because index is a 1D array and np.where returns a tupple
            for k in index:
                indexNode = EBC['localDOF#'][k]
                # code for applying essential boundary condition for first iteration
                if iter==0:
                    value = SME[indexNode][indexNode]
                    SME[indexNode,:] = 0.0
                    CVE[:] = CVE - EBC['Values'][k] * SME[:,indexNode]
                    SME[:,indexNode] = 0.0
                    SME[indexNode,indexNode] = value
                    CVE[indexNode] = EBC['Values'][k] * value
                else:
                    value = SME[indexNode][indexNode]
                    SME[indexNode,:] = 0.0
                    CVE[:] = CVE - 0.0 * SME[:,indexNode]
                    SME[:,indexNode] = 0.0
                    SME[indexNode,indexNode] = value
                    CVE[indexNode] = 0.0

        return SME, CVE
    
    def applyNBC(self, i, CVE, NBC, loadfactor):
        '''
        Apply natural boundary condition to element level stiffness matrices and column vector.
        i = current element number.
        CVE = column/ force vector of an element
        NBC = input natural boundary conditions
        loadfactor = ratio of total NBC to be applied in this load step calculation.
        Returns
        CVE = element level column/ force vector, np.ndarray
        '''          

        #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if i in NBC['Elem']:
            index = np.where(NBC['Elem']==i)[0] # [0] because index could be a 1D array and np.where returns a tupple
            for k in index:
                indexNode =  NBC['localDOF#'][k]        
                CVE[indexNode] += NBC['Values'][k] * loadfactor
        
        return CVE

    def applyMBC(self, i, MBC):
        '''
        This function returns mixed boundary conditions for applying to stiffness matrix and force vector.
        i: element number
        MBC: mixed boundary condition
        '''
        
        index = np.where(MBC['Elem']==i)[0] # [0] because index could be a 1D array and np.where returns a tupple
        count = np.size(index)
        indexNode = np.zeros(shape=count, dtype=np.int32)
        refU = np.zeros(shape=count, dtype=np.float64)
        betaO = np.zeros(shape=count, dtype=np.float64)
        beta = np.zeros(shape=count, dtype=np.float64)
        for n, k in enumerate(index):
            indexNode[n] =  MBC['localDOF#'][k]
            refU[n] = MBC['Values'][k][0]
            betaO[n] = MBC['Values'][k][1]
            beta[n] = MBC['Values'][k][2]
        
        return indexNode, refU, betaO, beta