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
        This function counts the number of essential boundary conditions specifed by the input.
        '''

        count = 0
        for row in self.ECON:
            for elemNode in row:
                for node in self.boundaryCond['globalNode#']:
                    if elemNode == node:
                        count +=1

        return count
    
    def countElemBC(self, i):
        '''
        This function counts the number of essential boundary conditions specifed by the input in a specified element.

        i: It is the element number in computation.
        '''
        nodeElem = []
        countElem = 0
        for elemNode in self.ECON[i]:
            for node in self.boundaryCond['globalNode#']:
                if elemNode == node:
                    countElem +=1
                    nodeElem.append(elemNode)

        return countElem, nodeElem

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






