'''
Author:      Kartikay Shukla
File:        solver.py
Created:     September 26, 2025 
LM:          October 6, 2025

Description
This file contains function for calculating, assemblying and solving stiffness matrices and force vector.
Refer to Dr. Greg Payette's notes and Dr Simo's paper (part I and II) for details on formula for calculation.
'''

import numpy as np

from itertools import product
from scipy.sparse import coo_array
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from gen.gen_interpFunction import *
from gen.gen_utilities import *
from gen.gen_gaussQuadCalc import *
from gen.gen_mesh1D import *

class FEMSolver:
    '''
    Class to solve finite strain beam finite element problem.
    '''

    def __init__(self, DOFPN, NNPEL, NEL, NGQP, ECON, elemGlobalCoord, boundaryE, boundaryN, appForce, appMoment, inMatModF, inMatModM):
        '''
        Class constructor
        DOFPN: degree of freedom per node, np.float64.
        NNPEL: Number of nodes per element, np.float64.
        NEL: Number of elements, np.float64.
        NGQP: number of Gauss Quadrature points, np.float64.
        ECON = Element connectivity array, np.ndarray of size NEL x NNPEL.
        elemGlobalCoord: global coordinates of each node of an element, np.ndarray. NEL x NNPEL array.
        boundaryE: dictionary of input essential boundary condition with keys as
            'Elem': Stores element # that has nodes with essential boundary conditions, np.array.
            'Nodes': Index location of node in ECON with boundary condition, np.array.
            'Values': Value of boundary condition for every degree of freedom, np.ndarray. Count x DOF. Count is the total number of EBC.
        boundaryN: dictionary of input natural boundary condition. Similar structure as boundaryE.
        appForce: Applied force at each node of a global element, np.ndarray. NEL x NNPEL x DOF/2. Output of applied function in utilities file.
        appMoment: Applied moment at each node of a global element, np.ndarray. NEL x NNPEL x DOF/2. Output of appFM function in utilities file.
        inMatModF: initial material model relating normal stress/strain, np.ndarray. 3 x 3 for DOF=6.
        inMatModM: initial material model relating shear stress/strain, np.ndarray. 3 x 3 for DOF=6.
        pointForce: point load applied at the node, np.ndarray. NEL x NNPEL x DOF/2. Output of applied function in utilities file.
        pointMoment: point moment applied at the node, np.ndarray. NEL x NNPEL x DOF/2. Output of applied function in utilities file.
        '''

        self.DOF = DOFPN
        self.NNPEL = NNPEL
        self.NEL = NEL
        self.NGQP = NGQP
        self.ECON = ECON
        self.elemGlobalCoord = elemGlobalCoord
        self.boundaryE = boundaryE
        self.boundaryN = boundaryN
        self.appForce = appForce
        self.appMoment = appMoment
        self.inMatModF = inMatModF
        self.inMatModM = inMatModM

        #-------------------------------calling functions----------------------------#
        # interpolation functions
        self.interp, self.interpDiff = interpLagGLQ(NNPEL, NGQP)
        # calling gauss weights from gaussLegQuad function
        self.gaussWt = gaussLegQuad(NGQP)[1]
        # permutation operator from utilities function
        self.permutation = permutation_symbol()

        #----------------------------Initial Calculations----------------------------------------------#
        # Jacobian
        self.J = np.dot(elemGlobalCoord, self.interpDiff)
        # calling returns of DOFCON function 
        self.dofCON = DOFCON(self.DOF, self.NEL, self.NNPEL, self.ECON)
        # calculations for size of matrices for global assembly
        self.eqns_p_elem = self.DOF * self.NNPEL
        self.shapeGM = self.DOF * (self.NEL * self.NNPEL - self.NEL + 1) # total number of FE nodes, same as DOF * global number of nodes.
        self.shapeElemMat = self.DOF * self.NNPEL # shape of Element matrices, square matrix, same as eqs_p_elem
        self.sizeElemMat = self.shapeElemMat ** 2 # size of Element matrices shapeElemMat x shapeElemMat
        self.sizeSparGM = self.sizeElemMat * self.NEL # size of sparse matrices, size = (shape,1) since vector
        self.globalNodes = (self.NEL * self.NNPEL - self.NEL + 1) # Total number of nodes in global assembly, same as DOF * globalNodes = shapeGM

        #----------------------------MEMORY ALLOCATION for element level functions-------------------------------------------#
        # memory allocation for interpolation functions and there outer products
        self.S = {f'S{i}':np.zeros(shape=self.NNPEL,dtype=np.float64) for i in range(2)} 
        self.outerS = {f'S{i}{j}':np.zeros(shape=[self.NNPEL, self.NNPEL],dtype=np.float64) for i in range(2) for j in range(2)}
        # configuration update variables - reusable memory allocations
        self.netRotVec = np.zeros(shape=[self.DOF//2,self.NNPEL], dtype=np.float64) # total rotation vector of a node, defining the total rotation from initial configuration to current configuration.
        self.nu = np.zeros(shape=[self.DOF//2, self.NNPEL],dtype=np.float64) # incremental change in rotation vector of a node.
        self.beta = np.zeros(shape=[self.DOF//2], dtype=np.float64)
        self.xi = np.zeros(shape=[self.DOF//2], dtype=np.float64)
        self.dphi_oE = np.zeros(shape=[self.DOF//2], dtype=np.float64) # derivative of initial or previous translational configuration of beam wrt to length of beam
        self.nuGP = np.zeros(shape=self.DOF//2,dtype=np.float64) # incremental change in rotation vector of a Gauss point.
        self.diffNuGP = np.zeros(shape=self.DOF//2,dtype=np.float64) # spatial derivative of incremental change in rotation vector of a node, self.nu
        self.netRotVecGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # total rotation vector of a Gauss point, defining the total rotation from initial configuration to current configuration.
        self.netRotMatGP = np.zeros(shape=[self.DOF//2,self.DOF//2], dtype=np.float64) # numpy 3x3 array for storing orthogonal rotation tensor at each node
        self.incRotMatGP = np.zeros(shape=[self.DOF//2,self.DOF//2], dtype=np.float64) # numpy 3x3 array for storing incremental orthogonal rotation tensor at each Gauss point
        # internal strain, material model and internal stress measurements
        self.gammaGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # axial strain measurement of an element
        self.omegaGP = np.zeros(shape=[self.NGQP,self.DOF//2], dtype=np.float64) # curvature strain measurement of an element
        self.prevOmegaGP = np.zeros(shape=[self.NGQP,self.DOF//2], dtype=np.float64) # curvature strain measurement of an element
        self.matModFGP = np.zeros(shape=[3,3], dtype=np.float64) # constitutive relations between normal stress-strain compoenents of an element evaluated at Gauss point
        self.matModMGP = np.zeros(shape=[3,3], dtype=np.float64) # constitutive relations between shear stress-strain compoenents of an element evaluated at Gauss point
        self.forceGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # Internal stress measurement for translational DOF of an element evaluated at Gauss point
        self.momentGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # Internal stress measurement for rotational DOF of an element evaluated at Gauss point
        self.appForceGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # External/applied stress measurement for translational DOF of an element evaluated at Gauss point
        self.appMomentGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # External/applied stress measurement for rotational DOF of an element evaluated at Gauss point
        # element level stiffness and force vector.
        self.SME = np.zeros(shape=[self.shapeElemMat,self.shapeElemMat], dtype=np.float64)
        self.coeffSME = {f'K{i}{j}':np.zeros(shape=[self.NNPEL,self.NNPEL],dtype=np.float64) for i,j in product(range(self.DOF), repeat=2)} # memory allocation for constants of elemental stiffness matrices
        self.CVE = np.zeros(shape=self.shapeElemMat, dtype=np.float64)
        self.coeffCVE = {f'F{i}':np.zeros(shape=self.NNPEL,dtype=np.float64) for i in range(self.DOF)} # memory allocation for constants of elemental force vector

        #----------------------------MEMORY ALLOCATION for global level functions-------------------------------------------#
        # variables for storing vectorised coefficient matrix and column vector
        self.sparSMG = np.zeros(shape=self.sizeSparGM, dtype=np.float64)
        self.SMG = csr_matrix((self.shapeGM,self.shapeGM), dtype=np.float64)
        self.CVG =  np.zeros(shape=self.shapeGM, dtype=np.float64)
        # variables for storing index locations of coefficient matrix
        self.sparIIrow = np.zeros(shape=self.sizeSparGM, dtype=np.float64)
        self.sparJJcol = np.zeros(shape=self.sizeSparGM, dtype=np.float64)
        # array variable for storing solution
        self.solution = np.zeros(shape=self.shapeGM, dtype=np.float64)

        #----------------------------MEMORY ALLOCATION for confugration update functions-------------------------------------------#
        # memory allocation for configuration update at every global node
        self.previousConfig = {f'{i}':np.zeros(shape=self.globalNodes, dtype=np.float64) for i in range(self.DOF)}
        self.changeConfig = {f'{i}':np.zeros(shape=self.globalNodes, dtype=np.float64) for i in range(self.DOF)}
        self.newConfig = {f'{i}':np.zeros(shape=self.globalNodes, dtype=np.float64) for i in range(self.DOF)}
        # memory allocation for translational location at every global node
        self.phi_o = np.zeros(shape=[self.globalNodes,self.DOF//2], dtype=np.float64) # phi_o vector at each global node 
        # initialise phi_o vector for every global node. phi_o stores coordinates of every global node. 
        # it has been defined as a curve called the line of centroids. The rotation vector from the newConfig dictionary helps define the rotation of planes at each global node which can be used to determine the spatial coordinate of the other points on the plane.
        for j,row in enumerate(self.ECON):
            for k, node in enumerate(row):
                self.phi_o[node,-1] = elemGlobalCoord[j,k]
        # memory allocation for rotation update variables
        shapeVec = self.DOF//2 # = 3, shape of rotation vector
        shapeMat = [shapeVec, shapeVec] # = [3,3] shape of rotation tensor
        rotKeys = ['previous', 'change', 'new'] # keys for dictionary of rotation vector and rotation tensor
        self.rotVec = {i:np.zeros(shape=shapeVec, dtype=np.float64) for i in rotKeys}
        self.rotMat = {i:np.zeros(shape=shapeMat, dtype=np.float64) for i in rotKeys}
    
    def _elemMatComput(self, i):
        '''
        This function computes the element matrices.
        Should be run after varProcessing as it generates the inputs for this function.
        appForce = applied force, spatial form, np.ndarray. It is a NGQP x 3 vector.
        appMoment = applied moment, spatial form, np.ndarray. It is a NGQP x 3 vector.
        All of the above parameters are evaluated at gauss points and not at nodes.
        '''

        # Re-zero memory allocations
        self.S['S0'][:] = 0.0
        self.S['S1'][:] = 0.0
        for key in self.outerS.keys():
            self.outerS[key][:,:] = 0.0
        self.SME[:,:] = 0.0
        for key in self.coeffSME.keys():
            self.coeffSME[key][:,:] = 0.0
        self.CVE[:] = 0.0
        for key in self.coeffCVE.keys():
            self.coeffCVE[key][:] = 0.0
        self.netRotVec[:,:] = 0.0
        self.nu[:,:] = 0.0

        # configuration update initialisation
        self.prevOmegaGP[:,:] = np.copy(self.omegaGP[:,:])
        self.netRotVec[:,:] = np.array([self.newConfig['3'][self.ECON[i]], 
                                        self.newConfig['4'][self.ECON[i]], 
                                        self.newConfig['5'][self.ECON[i]]], dtype=np.float64)
        self.nu[:,:] = np.array([self.changeConfig['3'][self.ECON[i]], 
                                 self.changeConfig['4'][self.ECON[i]], 
                                 self.changeConfig['5'][self.ECON[i]]], dtype=np.float64) # incremental rotation vector
        
        for GP in range(self.NGQP):
            # Re-zero memory allocation
            self.beta[:] = 0.0
            self.xi[:] = 0.0
            self.dphi_oE[:] = 0.0
            self.nuGP[:] = 0.0
            self.diffNuGP[:] = 0.0
            self.netRotVecGP[:] = 0.0
            self.netRotMatGP[:,:] = 0.0
            self.incRotMatGP[:,:] = 0.0
            self.gammaGP[:] = 0.0
            self.matModFGP[:,:] = 0.0
            self.matModMGP[:,:] = 0.0
            self.forceGP[:] = 0.0
            self.momentGP[:] = 0.0
            self.appForceGP[:] = 0.0
            self.appMomentGP[:] = 0.0
            # initial calculations
            effecWt = self.J[i,GP] * self.gaussWt[GP]
            # interpolation functions
            self.S['S0'][:] = self.interp[:,GP]
            self.S['S1'][:] = self.interpDiff[:,GP] * (1 / self.J[i,GP])
            # outer product of interpolation functions
            self.outerS['S00'][:,:] = np.outer(self.S['S0'],self.S['S0'])
            self.outerS['S01'][:,:] = np.outer(self.S['S0'],self.S['S1'])
            self.outerS['S10'][:,:] = self.outerS['S01'].T
            self.outerS['S11'][:,:] = np.outer(self.S['S1'],self.S['S1'])

            #---------------------Configuration Update Procedures---------------------------------#
            # strain update at an element at a gauss point
            self.dphi_oE[:] = np.dot(self.S['S1'],self.phi_o[self.ECON[i],:])
            print('self.S[S1]=',self.S['S1'])
            print('self.phi_o[self.ECON[i],:]=',self.phi_o[self.ECON[i],:])
            print('self.dphi_oE=',self.dphi_oE)
            self.nuGP[:] = np.dot(self.S['S0'],self.nu.T) # incremental rotation vector
            self.diffNuGP[:] = np.dot(self.S['S1'],self.nu.T) # derivative of incremental rotation vector at Gauss point
            self.netRotVecGP[:] = np.dot(self.S['S0'],self.netRotVec.T)
            self.netRotMatGP[:,:] = rotTensor(self.netRotVecGP[:]) # orthogonal rotation tensor at Gauss point
            self.incRotMatGP[:,:] = rotTensor(self.nuGP[:]) # incremental orthogonal rotation tensor at Gauss point
            
            normNu = np.linalg.norm(self.nuGP[:], ord=2)
            self.xi[:] = self.incRotMatGP[:,:] @ self.prevOmegaGP[GP, :]
            if normNu==0.0:
                self.beta[:] = 0.0
            else:
                self.beta[:] = np.sin(normNu)/normNu * self.diffNuGP[:] \
                        + (1 - (np.sin(normNu) / normNu)) * (np.dot(self.nuGP[:],self.diffNuGP[:]) / normNu) * self.nuGP[:] / normNu \
                        + (2 * np.sin(0.5 * normNu) * np.sin(0.5 * normNu)) / normNu**2 * np.cross(self.nuGP[:], self.diffNuGP[:])
            self.omegaGP[GP,:] = self.beta[:] + self.xi[:]
            self.gammaGP[:] = self.dphi_oE - np.dot(self.netRotMatGP[:,:],np.array([0,0,1]))

            # material model update 
            self.matModFGP[:,:] = self.netRotMatGP[:,:] @ self.inMatModF @ self.netRotMatGP[:,:].T
            self.matModMGP[:,:] = self.netRotMatGP[:,:] @ self.inMatModM @ self.netRotMatGP[:,:].T
            # internal stress measurement update
            self.forceGP[:] = self.matModFGP[:,:] @ self.gammaGP[:]
            self.momentGP[:] = self.matModMGP[:,:] @ self.omegaGP[GP,:]

            # external stress measurement update
            self.appForceGP[:] = np.dot(self.S['S0'], self.appForce[i,:,:])
            self.appMomentGP[:] = np.dot(self.S['S0'], self.appMoment[i,:,:])
            #---------------Element Matrix Calculation---------------------------------#
            for a in range(self.DOF//2): # a=alpha, b=beta as notes in reference material
                for b in range(self.DOF//2):
                    self.coeffSME[f'K{a}{b}'][:,:] += self.matModFGP[a,b] * self.outerS['S11'] * effecWt

                    self.coeffSME[f'K{a}{b+self.DOF//2}'][:,:] += (np.dot(np.dot(self.dphi_oE, self.permutation[:,b,:]), self.matModFGP[a,:].T) - np.dot(self.forceGP[:], self.permutation[:,b,a])) * self.outerS['S10'] * effecWt
                    
                    self.coeffSME[f'K{a+self.DOF//2}{b}'][:,:] += (np.dot(np.dot(self.dphi_oE, self.permutation[:,a,:]), self.matModFGP[b,:].T) + np.dot(self.forceGP[:], self.permutation[:,b,a])) * self.outerS['S01'] * effecWt
                    if a==2 and b==2:
                        print('self.forceGP=',self.forceGP)
                    self.coeffSME[f'K{a+self.DOF//2}{b+self.DOF//2}'][:,:] += ((np.dot(np.dot(np.dot(self.dphi_oE, self.permutation[:,a,:]), self.matModFGP[:,:]), np.dot(self.dphi_oE, self.permutation[:,b,:]).T)\
                                                        + np.dot(np.dot(self.permutation[a,:,:],np.dot(self.forceGP[:], self.permutation[:,b,:]).T), self.dphi_oE)) * self.outerS['S00'] \
                                                        + (- np.dot(self.momentGP[:], self.permutation[:,b,a])) * self.outerS['S10'] \
                                                        + (self.matModMGP[a,b]) * self.outerS['S11']) * effecWt
                    
                self.coeffCVE[f'F{a}'][:] += (self.appForceGP[a] * self.S['S0'] + (- self.forceGP[a]) * self.S['S1']) * effecWt
                
                self.coeffCVE[f'F{a+self.DOF//2}'][:] += (((np.dot(np.dot(self.permutation[a,:,:], self.forceGP[:]), self.dphi_oE)) + self.appMomentGP[a]) * self.S['S0'] + (- self.momentGP[a]) * self.S['S1']) * effecWt
        
        for j in range(self.DOF):
            for k in range(self.DOF):
                self.SME[j:self.eqns_p_elem:self.DOF, k:self.eqns_p_elem:self.DOF] = self.coeffSME[f'K{j}{k}']
            self.CVE[j:self.eqns_p_elem:self.DOF] = self.coeffCVE[f'F{j}']
        print('K=',self.coeffSME)
        # print('F=',self.coeffCVE)
        print('SME=',self.SME)
        # print('CVE=',self.CVE)
        # return self.SME, self.CVE

    def _applyEBC(self, i, iter):
        '''
        Apply essential boundary condition to element level stiffness matrices and column vector.
        SME = element level stiffness matrix, np.ndarray.
        CVE = element level column/ force vector, np.ndarray
        '''          

       #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if i in self.boundaryE['Elem']:
            index = np.where(self.boundaryE['Elem']==i)[0] # [0] because index could be a 1D array and np.where returns a tupple
            for k in index:
                indexNode =  self.boundaryE['Nodes'][k] * self.DOF
                if iter == 0:
                    for j in range(self.DOF):    
                        # code for applying essential boundary condition for first iteration
                        value = self.SME[indexNode+j][indexNode+j]
                        self.SME[indexNode+j,:] = 0.0
                        self.CVE[:] = self.CVE - self.boundaryE['Values'][k,j] * self.SME[:,indexNode+j]
                        self.SME[:,indexNode+j] = 0.0
                        self.SME[indexNode+j,indexNode+j] = value
                        self.CVE[indexNode+j] = self.boundaryE['Values'][k,j] * value                        
                else:
                    for j in range(self.DOF):    
                        # code for applying homogeneous essential boundary condition
                        value = self.SME[indexNode+j][indexNode+j]
                        self.SME[indexNode+j,:] = 0.0
                        self.CVE[:] = self.CVE - 0 * self.SME[:,indexNode+j]
                        self.SME[:,indexNode+j] = 0.0
                        self.SME[indexNode+j,indexNode+j] = value
                        self.CVE[indexNode+j] = 0 * value  
        
        # return self.SME , self.CVE 
    
    def _applyNBC(self, i, iter):
        '''
        Apply natural boundary condition to element level stiffness matrices and column vector.
        Returns
        SME = element level stiffness matrix, np.ndarray.
        CVE = element level column/ force vector, np.ndarray
        '''          

       #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if i in self.boundaryN['Elem']:
            index = np.where(self.boundaryN['Elem']==i)[0] # [0] because index could be a 1D array and np.where returns a tupple
            for k in index:
                indexNode =  self.boundaryN['Nodes'][k] * self.DOF
                if iter == 0:
                    for j in range(self.DOF):    
                        # code for applying natural boundary condition for first iteration
                        self.CVE[indexNode+j] += self.boundaryN['Values'][k,j]
                else:
                    for j in range(self.DOF):    
                        # code for applying homogeneous natural boundary condition
                        self.CVE[indexNode+j] += 0 
        
        return self.SME , self.CVE 

    def _globalSolve(self, iter):
        '''
        Function to assemble and solve global matrices. It uses elemMatComput function to compute individual element matrices and then assembles them to form global matrix.
        It also uses sparse solver to solve for the change in displacements.
        '''
        # Re-zero memory allocations
        self.sparSMG[:] = 0.0
        self.SMG[:,:] = 0.0
        self.CVG[:] = 0.0
        self.sparIIrow[:] = 0.0
        self.sparJJcol[:] = 0.0
        self.solution[:] = 0.0
        self.SME[:,:]=0.0
        self.CVE[:]=0.0
        
        # counter, for index position of vectorised matrix.
        m = 0 

        for i in range(self.NEL): # loop over elements
            if (i % 1 == 0):
                print(i)

            self._elemMatComput(i)
            self._applyEBC(i, iter)
            self._applyNBC(i, iter)

            print('after boundary SME=',self.SME)
            print('after boundary CVE=',self.CVE)
            # Global stiffness matrix assembly
            self.sparSMG[m:m+self.sizeElemMat] = self.SME.flatten() # value
            self.sparIIrow[m:m+self.sizeElemMat] = np.repeat(self.dofCON[i],self.shapeElemMat) # row index
            self.sparJJcol[m:m+self.sizeElemMat] = np.tile(self.dofCON[i],self.shapeElemMat) # column index
            m += self.sizeElemMat
            
            # global column vector assembly
            self.CVG[self.dofCON[i]] += self.CVE

        # create global sparse stiffness matrix from vectorised matrix
        self.SMG[:,:] = coo_array((self.sparSMG, (self.sparIIrow,self.sparJJcol)), shape=(self.shapeGM,self.shapeGM)).tocsr()
    
        # solve the linear algebra problem using sparse solver
        self.solution[:] = spsolve(self.SMG, self.CVG)
        print('det SMG=',np.linalg.det(self.SMG.todense()))
        print('rank SMG=',np.linalg.matrix_rank(self.SMG.todense()))
        # print('CVG=',self.CVG)
        print('solution=',self.solution)
        # return self.solution

    def _deltaConfig(self):
        '''
        This function separates translation and rotation response of each node into an array in dictionary.
        solution: latest solution of response of beam, np.ndarray. Output of the globMat.solve() which calculates the incremental change in solution. A one-dimensional array.
        '''
        for key in self.changeConfig.keys():
            self.changeConfig[key] = 0.0

        for i in range(self.DOF):
            self.changeConfig[f'{i}'] = self.solution[i::self.DOF]
        
        # return self.changeConfig
    
    def _updateConfig(self):
        '''
        This function updates the response of the beam with latest solution.
        This function should be called after running deltaConfig.
        iter: number of iteration. Initial configuration is the previous configuration for first iteration.
        initialConfig: initial configuration of the beam. Dictionary of np.ndarray
        changeConfig: change in configuration of the beam. Output of  deltaConfig function. Dictionary of np.ndarray.

        returns
        newConfig: dictionary of np.ndarray.
        '''
        # Re-zero memory allocations

        for key in self.newConfig.keys():
            self.previousConfig[key][:] = self.newConfig[key][:] # previous configuration of beam.
            self.newConfig[key][:] = 0.0
        for key in self.rotMat.keys():
            self.rotMat[key][:,:] = 0.0
        for key in self.rotVec.keys():
            self.rotVec[key][:] = 0.0

        # update translational movement
        for i in range(self.DOF//2):
            self.newConfig[f'{i}'][:] = self.previousConfig[f'{i}'][:] + self.changeConfig[f'{i}'][:]

        # update rotational movement
        for i in range(self.globalNodes):
            # previous configuration
            # self.rotVec['previous'][:] = np.array(previousConfig[f'{j+3}'][i] for j in range(self.DOF // 2))
            self.rotVec['previous'][0] = self.previousConfig['3'][i]
            self.rotVec['previous'][1] = self.previousConfig['4'][i]
            self.rotVec['previous'][2] = self.previousConfig['5'][i]
            self.rotMat['previous'][:,:] = rotTensor(self.rotVec['previous'])

            # change in configuration
            self.rotVec['change'][0] = self.changeConfig['3'][i]
            self.rotVec['change'][1] = self.changeConfig['4'][i]
            self.rotVec['change'][2] = self.changeConfig['5'][i]
            self.rotMat['change'][:,:] = rotTensor(self.rotVec['change'])
            print('self.rotVec[change]=',self.rotVec['change'])

            # new configuration
            self.rotMat['new'][:,:] = self.rotMat['change'] @ self.rotMat['previous']
            print('rotMat[prev]=',self.rotMat['previous'])
            print('rotMat[change]=',self.rotMat['change'])
            print('rotMat[new]=',self.rotMat['new'])

            self.rotVec['new'][:] = rotVector(self.rotMat['new'])

            # update configuration dictionary
            self.newConfig['3'][i] = self.rotVec['new'][0]
            self.newConfig['4'][i] = self.rotVec['new'][1]
            self.newConfig['5'][i] = self.rotVec['new'][2]

        self.phi_o[:,0] = self.newConfig['0'][:]
        self.phi_o[:,1] = self.newConfig['1'][:]
        self.phi_o[:,2] = self.newConfig['2'][:]

        return self.previousConfig, self.newConfig #self.phi_o
    
    def FEMSolve(self, iter):
        '''
        This function calls other functions in this class returns the error between configurations.
        iter: current iteration number. 
        '''
        self._globalSolve(iter)
        self._deltaConfig()
        self._updateConfig()
        error = calcError(self.previousConfig, self.newConfig)

        return error, self.solution

    

