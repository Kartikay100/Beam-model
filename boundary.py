'''
Author:      Kartikay Shukla
File:        boundary.py
Created:     October 8, 2025 
LM:          October 10, 2025

Description
This file contains function for handling boundary input from JSON file and applying boundary conditions to the element matrices.
Refer to Dr. Greg Payette's notes and Dr Simo's paper (part I and II) for details on formula for calculation.
'''
import numpy as np

class boundary():
    def __init__(self, DOFPN, NNPEL, NEL, ECON, boundaryCond):
        '''
        Class constructor
        '''
        self.DOF = DOFPN
        self.NNPEL = NNPEL
        self.NEL = NEL
        self.ECON = ECON
        self.boundaryCond = boundaryCond
        self.keys = boundaryCond.keys()

    def countBC(self):
        '''
        This function counts the number of element level essential boundary conditions specifed by the nodes array (input).
        '''

        count = 0
        for row in self.ECON:
            for elemNode in row:
                for inpNode in self.boundaryCond['globalNode#']:
                    if elemNode == inpNode:
                        count +=1

        return count

    def valueBC(self, count):
        '''
        This function returns an array of element level boundary condition form the input boundary condition
        count = number of nodes with EBC, output of function countEBC
        Returns
        boundary: dictionary of input boundary condition with keys as
            'Elem': Stores element # that has nodes with boundary conditions, np.array.
            'Nodes': Index location of node in ECON with boundary condition, np.array.
            'Values': Value of boundary condition for every degree of freedom, np.ndarray. Count x DOF. Count is the total number of EBC.
        '''

        boundary = {'Elem': np.zeros(shape=count, dtype=np.int32),
                    'Nodes': np.zeros(shape=count, dtype=np.int32),
                    'Values': np.zeros(shape=[count, self.DOF], dtype=np.float64) }
        
        cnt = 0
        for i, row in enumerate(self.ECON):
            # boundary['Elem'][i] = 0.0
            for j, elemNode in enumerate(row):
                if elemNode in self.boundaryCond['globalNode#']:
                    index = np.where(self.boundaryCond['globalNode#']==elemNode)[0][0] # (0,0) because assumed global node number do not repeat
                    boundary['Elem'][cnt] = i
                    boundary['Nodes'][cnt] = j
                    for k, key in enumerate(list(self.keys)[1:]):
                        boundary['Values'][cnt,k] = self.boundaryCond[key][index]
                    cnt += 1
        
        return boundary
    
    def applyEBC(self, SME, CVE, i, boundary, iter):
        '''
        Apply essential boundary condition to element level stiffness matrices and column vector.
        SME = element level stiffness matrix, np.ndarray.
        CVE = element level column/ force vector, np.ndarray
        i = current element number
        boundary = input boundary condition. Output of essentialBC function. It has three numpy arrays: Elem, Nodes and Values.
            Elem: index denotes the element number and value at the index denotes the number of nodes in that element have an essential boundary condition.
            Nodes: index of node in ECON matrix that has a boundary condition.
            Values: value of boundary condition. Size depends on number of degrees of freedom.
        iter: Current iteration number for a non linear solver.
        '''
        #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if boundary['Elem'][i] != 0:
            if iter == 0:
                for k, nodeNum in enumerate(boundary['Nodes']): 
                    indexNode =  nodeNum*self.DOF
                    for j in range(self.DOF):    
                        # code for applying essential boundary condition
                        value = SME[indexNode+j][indexNode+j]
                        SME[indexNode+j,:] = 0.0
                        CVE = CVE - boundary['Values'][k,indexNode+j]*SME[:,indexNode+j]
                        SME[:,indexNode+j] = 0.0
                        SME[indexNode+j,indexNode+j] = value
                        CVE[indexNode+j] = boundary['Values'][k,indexNode+j]*value
            else:
                for k, nodeNum in enumerate(boundary['Nodes']): 
                    indexNode =  nodeNum*self.DOF
                    for j in range(self.DOF):    
                        # code for applying essential boundary condition
                        value = SME[indexNode+j][indexNode+j]
                        SME[indexNode+j,:] = 0.0
                        CVE = CVE - 0*SME[:,indexNode+j]
                        SME[:,indexNode+j] = 0.0
                        SME[indexNode+j,indexNode+j] = value
                        CVE[indexNode+j] = 0*value
       
        return SME, CVE 

class boundaryPDOF():

    def __init__(self, DOFPN, NNPEL, NEL, ECON):
        '''
        Class constructor
        '''
        self.DOF = DOFPN
        self.NNPEL = NNPEL
        self.NEL = NEL
        self.ECON = ECON

    def applyEBCs(self, SME, CVE, i, iter, boundary):
        '''
        This function applies essential boundary conditions to the given stiffness matrix and column vector based on the out dictionary, which is the output of boundaryCs function.
        It only handles one degree of freedom. Need to call this again and again to apply BC for multiple DOF.
        i : current element number
        j : current degree of freedom
        '''

        for key in boundary.keys():
            out = boundary[f'{key}']
            numberBCs = out['pointer'][i+1]-out['pointer'][i]
            if numberBCs != 0:
                j = int(key)
                if iter == 0:
                    for numberBC in range(numberBCs):
                        cnt = out['pointer'][i] + numberBC
                        elemNodeNum = out['nodes'][cnt] * self.DOF + j
                        # code for applying essential boundary condition
                        value = SME[elemNodeNum][elemNodeNum]
                        SME[elemNodeNum,:] = 0.0
                        CVE = CVE - out['values'][cnt]*SME[:,elemNodeNum]
                        SME[:,elemNodeNum] = 0.0
                        SME[elemNodeNum,elemNodeNum] = value
                        CVE[elemNodeNum] = out['values'][cnt]*value
                else:
                    for numberBC in range(numberBCs):
                        cnt = out['pointer'][i] + numberBC
                        elemNodeNum = out['nodes'][cnt] * self.DOF + j
                        # code for applying essential boundary condition
                        value = SME[elemNodeNum][elemNodeNum]
                        SME[elemNodeNum,:] = 0.0
                        SME[:,elemNodeNum] = 0.0
                        SME[elemNodeNum,elemNodeNum] = value
                        CVE[elemNodeNum] = 0.0
        
        return SME, CVE 
    
    def applyNBCs(self, SME, CVE, i, iter, boundary, loadFactor):
        '''
        This function applies natural boundary conditions to the given stiffness matrix and column vector based on the out dictionary, which is the output of boundaryCs function.
        It only handles one degree of freedom. Need to call this again and again to apply BC for multiple DOF.

        '''
        for key in boundary.keys():
            out = boundary[f'{key}']
            numberBCs = out['pointer'][i+1]-out['pointer'][i]
            if numberBCs != 0:
                if iter == 0:
                    j = int(key)
                    for numberBC in range(numberBCs):
                        cnt = out['pointer'][i] + numberBC
                        elemNodeNum = out['nodes'][cnt] * self.DOF + j
                        CVE[elemNodeNum] += out['values'][cnt] * loadFactor

        return SME, CVE 