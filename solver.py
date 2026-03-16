'''
Author:      Kartikay Shukla
File:        solver.py
Created:     September 26, 2025 
LM:          February 20, 2026

Description
This file contains function for calculating, assemblying and solving stiffness matrices and force vector. Newton Raphson method is used to compute solutions for nonlinear problem.
Refer to Simo's paper (part I and II) for details on formula for calculation.
'''

import numpy as np

from itertools import product
from scipy.sparse import coo_array
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from gen.gen_interpFunction import interpLagGLQ
from gen.gen_utilities import permutation_symbol, rotTensor, rotVector, calcErrorI, lengthCheck, interpRotVec
from gen.gen_gaussQuadCalc import gLQ
from gen.gen_mesh1D import DOFCON

class FEMSolver:
    '''
    Class to solve finite strain beam finite element problem.
    '''

    def __init__(self, NNPEL, NEL, ECON, elemGlobalCoord, boundaryE, boundaryN, appForce, appMoment, inMatModF, inMatModM, NGQP = None):
        '''
        Class constructor
        DOFPN: degree of freedom per node, np.float64.
        NNPEL: Number of nodes per element, np.float64.
        NEL: Number of elements, np.float64.
        ECON = Element connectivity array, np.ndarray of size NEL x NNPEL.
        elemGlobalCoord: global coordinates of each node of an element, np.ndarray. NEL x NNPEL array.
        boundaryE: dictionary of input essential boundary condition with keys as
            'Elem': Stores element # that has nodes with essential boundary conditions, np.array.
            'Nodes': Index location of node in ECON with boundary condition, np.array.
            'Values': Value of boundary condition for every degree of freedom, np.ndarray. Count x DOF. Count is the total number of EBC.
        boundaryN: dictionary of input natural boundary condition. Similar structure as boundaryE.
        appForce: Applied force at each node of a global element, np.ndarray. NEL x NNPEL x DOF/2 x 2. Output of genForceHandler() function in utilities file. Handles UDL and UVL.
        appMoment: Applied moment at each node of a global element, np.ndarray. NEL x NNPEL x DOF/2 x 2. Output of genForceHandler() function in utilities file. Handles UDL and UVL.
        inMatModF: initial material model relating normal stress/strain, np.ndarray. 3 x 3 for DOF=6.
        inMatModM: initial material model relating shear stress/strain, np.ndarray. 3 x 3 for DOF=6.
        '''

        self.DOF = 6
        self.NNPEL = NNPEL
        self.NEL = NEL
        self.ECON = ECON
        self.elemGlobalCoord = elemGlobalCoord
        self.boundaryE = boundaryE
        self.boundaryN = boundaryN
        self.appForceVariation = appForce
        self.appMomentVariation = appMoment
        self.inMatModF = inMatModF
        self.inMatModM = inMatModM
        if NGQP != None:
            self.NGQP = NGQP
        elif NNPEL<3:
            self.NGQP = NNPEL-1 # reduced order integration
        else:
            self.NGQP = NNPEL # number of gauss points same as number of nodes per element to ensure full integration.
        #-------------------------------calling functions----------------------------#
        # interpolation functions
        self.interp, self.interpDiff = interpLagGLQ(self.NNPEL, self.NGQP)
        # calling gauss weights from gaussLegQuad function
        self.gaussWt = gLQ(self.NGQP)['weights']
        # permutation operator from utilities function
        self.permutation = permutation_symbol()
        #----------------------------Initial Calculations----------------------------------------------#
        # Jacobian
        self.J = np.dot(elemGlobalCoord, self.interpDiff)
        # calling returns of DOFCON function 
        self.dofCON = DOFCON(self.DOF, self.NEL, self.NNPEL, self.ECON)
        self.E3 = np.array([0,0,1],dtype=np.float64) # E3 unit basis vector in reference configuration, defined along the length of the beam
        # calculations for size of matrices for global assembly
        self.eqns_p_elem = self.DOF * self.NNPEL
        self.globalNodes = self.NEL * self.NNPEL - self.NEL + 1 # Total number of nodes in global assembly, same as DOF * globalNodes = shapeGM
        self.shapeGM = self.DOF * self.globalNodes # total number of FE nodes
        self.shapeElemMat = self.DOF * self.NNPEL # shape of Element matrices, square matrix, same as eqs_p_elem
        self.sizeElemMat = self.shapeElemMat ** 2 # size of Element matrices shapeElemMat x shapeElemMat
        self.sizeSparGM = self.sizeElemMat * self.NEL # size of sparse matrices, size = (shape,1) since vector
        #----------------------------MEMORY ALLOCATION for element level functions-------------------------------------------#
        # compute Jacobian in the current configuration
        self.currJ = 0.0 # Jacobian in current configuration for an element at Gauss point
        # memory allocation for interpolation functions and their outer products
        self.S = {f'S{i}':np.zeros(shape=self.NNPEL,dtype=np.float64) for i in range(2)} 
        self.outerS = {f'S{i}{j}':np.zeros(shape=[self.NNPEL, self.NNPEL],dtype=np.float64) for i in range(2) for j in range(2)}
        # configuration update variables - reusable memory allocations
        self.netRotVec = np.zeros(shape=[self.NNPEL, self.DOF//2], dtype=np.float64) # total rotation vector of a node in current configuration, defining the total rotation from initial configuration to current configuration.
        self.nu = np.zeros(shape=[self.NNPEL,self.DOF//2],dtype=np.float64) # incremental change in rotation vector of a node.
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
        self.omegaGP = np.zeros(shape=[self.NEL,self.NGQP,self.DOF//2], dtype=np.float64) # curvature strain measurement of an element
        self.prevOmegaGP = np.zeros(shape=[self.NEL,self.NGQP,self.DOF//2], dtype=np.float64) # curvature strain measurement of an element in previous configuration
        self.matModFGP = np.zeros(shape=[3,3], dtype=np.float64) # constitutive relations between normal stress-strain compoenents of an element evaluated at Gauss point
        self.matModMGP = np.zeros(shape=[3,3], dtype=np.float64) # constitutive relations between shear stress-strain compoenents of an element evaluated at Gauss point
        self.forceGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # Internal stress measurement for translational DOF of an element evaluated at Gauss point
        self.momentGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # Internal stress measurement for rotational DOF of an element evaluated at Gauss point
        self.forceGPW = np.zeros(shape=[self.NEL, self.NGQP, self.DOF//2], dtype=np.float64) # Internal stress measurement for translational DOF of an element evaluated at Gauss point
        self.momentGPW = np.zeros(shape=[self.NEL, self.NGQP, self.DOF//2], dtype=np.float64) # Internal stress measurement for rotational DOF of an element evaluated at Gauss point
        self.appForce = np.zeros(shape=[NEL, NNPEL, self.DOF//2], dtype=np.float64) # External/applied distributed stress measurement for translational DOF of an element, calculated at nodes, can handle UVL along Z axis.
        self.appMoment = np.zeros(shape=[NEL, NNPEL, self.DOF//2], dtype=np.float64) # External/applied distributed stress measurement for rotational DOF of an element, calcaluted at nodes, can handle UVL along Z axis.
        self.appForceGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # External/applied distributed stress measurement for translational DOF of an element evaluated at Gauss point
        self.appMomentGP = np.zeros(shape=self.DOF//2, dtype=np.float64) # External/applied distributed stress measurement for rotational DOF of an element evaluated at Gauss point
        # element level stiffness and force vector.
        self.SME = np.zeros(shape=[self.shapeElemMat,self.shapeElemMat], dtype=np.float64) # element level stiffness matrix.
        self.coeffSME = {f'K{i}{j}':np.zeros(shape=[self.NNPEL,self.NNPEL],dtype=np.float64) for i,j in product(range(self.DOF), repeat=2)} # memory allocation for constants of elemental stiffness matrices
        self.CVE = np.zeros(shape=self.shapeElemMat, dtype=np.float64)
        self.coeffCVE = {f'F{i}':np.zeros(shape=self.NNPEL,dtype=np.float64) for i in range(self.DOF)} # memory allocation for constants of elemental force vector
        #----------------------------MEMORY ALLOCATION for global level functions-------------------------------------------#
        # variables for storing vectorised coefficient matrix and column vector
        self.sparSMG = np.zeros(shape=self.sizeSparGM, dtype=np.float64)
        self.SMG = csr_matrix((self.shapeGM,self.shapeGM), dtype=np.float64)
        self.SMGdense = np.zeros(shape=[self.shapeGM,self.shapeGM], dtype=np.float64) # for debugging purposes only, store dense matrix
        self.CVG =  np.zeros(shape=self.shapeGM, dtype=np.float64)
        self.genF =  np.zeros(shape=self.shapeGM, dtype=np.float64)
        # variables for storing index locations of coefficient matrix
        self.sparIIrow = np.zeros(shape=self.sizeSparGM, dtype=np.int32)
        self.sparJJcol = np.zeros(shape=self.sizeSparGM, dtype=np.int32)
        # array variable for storing solution
        self.changeSolution = np.zeros(shape=self.shapeGM, dtype=np.float64)
        #----------------------------MEMORY ALLOCATION for configration update functions-------------------------------------------#
        # memory allocation for configuration update at every global node
        self.previousConfig = {f'{i}':np.zeros(shape=self.globalNodes, dtype=np.float64) for i in range(self.DOF)}
        self.changeConfig = {f'{i}':np.zeros(shape=self.globalNodes, dtype=np.float64) for i in range(self.DOF)}
        self.newConfig = {f'{i}':np.zeros(shape=self.globalNodes, dtype=np.float64) for i in range(self.DOF)}        
        self.changeRotation = np.zeros(shape=[self.globalNodes,3], dtype=np.float64) # change in rotational vector at each global node at every iteration
        self.phi_o = np.zeros(shape=[self.globalNodes,3], dtype=np.float64) # translational vector at each global node 
        self.rotation = np.zeros(shape=[self.globalNodes,3], dtype=np.float64) # rotational vector at each global node 
        self.rotationMat = np.zeros(shape=[self.globalNodes,3,3], dtype=np.float64) # rotational matrix at each global node 
        self.rotationMat[:] = np.eye(3)
        
        # initialise phi_o vector for every global node. phi_o stores coordinates of every global node. 
        # it has been defined as a curve called the line of centroids. The rotation vector from the newConfig dictionary helps define the rotation of planes at each global node which can be used to determine the spatial coordinate of the other points on the plane.
        for j,row in enumerate(self.ECON):
            for k, node in enumerate(row):
                self.phi_o[node,-1] = elemGlobalCoord[j,k]
                self.appForce[j,k,:] = self.appForceVariation[j,k,:,0] + self.appForceVariation[j,k,:,1] * elemGlobalCoord[j,k]
                self.appMoment[j,k,:] = self.appMomentVariation[j,k,:,0] + self.appMomentVariation[j,k,:,1] * elemGlobalCoord[j,k]
        # memory allocation for rotation update variables
        shapeVec = self.DOF//2 # = 3, shape of rotation vector
        shapeMat = [shapeVec, shapeVec] # = [3,3] shape of rotation tensor
        rotKeys = ['previous', 'change', 'new'] # keys for dictionary of rotation vector and rotation tensor
        self.rotVec = {i: np.zeros(shape=shapeVec, dtype=np.float64) for i in rotKeys}
        self.rotMat = {i: np.zeros(shape=shapeMat, dtype=np.float64) for i in rotKeys}
    
    def _elemMatComput(self, i, ratioLoadStep):
        '''
        This function computes the element matrices.
        i = number of element.
        All the parameters are evaluated at gauss points and not at nodes.
        '''

        # Re-zero memory allocations
        self.SME[:,:] = 0.0
        for key in self.coeffSME.keys():
            self.coeffSME[key][:,:] = 0.0
        self.CVE[:] = 0.0
        for key in self.coeffCVE.keys():
            self.coeffCVE[key][:] = 0.0
        self.netRotVec[:,:] = 0.0
        self.nu[:,:] = 0.0

        # configuration update initialisation
        self.prevOmegaGP[i,:,:] = self.omegaGP[i,:,:]
        self.omegaGP[i,:,:] = 0.0

        self.netRotVec[:,:] = self.rotation[self.ECON[i],:] # total rotation vector at nodes of an element in current configuration, defining the total rotation from initial configuration to current configuration.
        self.nu[:,:] = self.changeRotation[self.ECON[i],:] # incremental rotation vector
        
        for GP in range(self.NGQP):
            # Re-zero temporary memory allocations
            self.S['S0'][:] = 0.0
            self.S['S1'][:] = 0.0
            for key in self.outerS.keys():
                self.outerS[key][:,:] = 0.0
            self.dphi_oE[:] = 0.0
            self.nuGP[:] = 0.0
            self.diffNuGP[:] = 0.0
            self.netRotVecGP[:] = 0.0
            self.netRotMatGP[:,:] = 0.0
            self.incRotMatGP[:,:] = 0.0
            self.beta[:] = 0.0
            self.xi[:] = 0.0
            self.gammaGP[:] = 0.0
            self.matModFGP[:,:] = 0.0
            self.matModMGP[:,:] = 0.0
            self.forceGP[:] = 0.0
            self.momentGP[:] = 0.0
            self.appForceGP[:] = 0.0
            self.appMomentGP[:] = 0.0
            # compute Jacobian and other constant terms at gauss point
            self.currJ = self.J[i,GP]
            
            effecWt = self.currJ * self.gaussWt[GP]
            # interpolation functions
            self.S['S0'][:] = self.interp[:,GP]
            self.S['S1'][:] = self.interpDiff[:,GP] * (1 / self.currJ)
            # outer product of interpolation functions
            self.outerS['S00'][:,:] = np.outer(self.S['S0'],self.S['S0'])
            self.outerS['S01'][:,:] = np.outer(self.S['S0'],self.S['S1'])
            self.outerS['S10'][:,:] = self.outerS['S01'].T
            self.outerS['S11'][:,:] = np.outer(self.S['S1'],self.S['S1'])

            #---------------------Configuration Update Procedures---------------------------------#
            # strain update at an element at a gauss point
            self.dphi_oE[:] = np.dot(self.S['S1'],self.phi_o[self.ECON[i],:])
            
            self.nuGP[:] = np.dot(self.S['S0'],self.nu) # incremental rotation vector at Gauss point
            self.incRotMatGP[:,:] = rotTensor(self.nuGP) # incremental orthogonal rotation tensor at Gauss point
            self.diffNuGP[:] = np.dot(self.S['S1'],self.nu) # derivative of incremental rotation vector at Gauss point  
            self.netRotVecGP[:], self.netRotMatGP[:,:] = interpRotVec(self.S['S0'], self.netRotVec) # rotation vector and orthogonal rotation tensor at Gauss point
            # self.netRotVecGP[:] = np.dot(self.S['S0'],self.netRotVec) # incremental rotation vector at Gauss point
            # self.netRotMatGP[:,:] = rotTensor(self.netRotVecGP) # incremental orthogonal rotation tensor at Gauss point
                        
            normNu = np.linalg.norm(self.nuGP, ord=2)
            self.xi[:] = self.incRotMatGP[:,:] @ self.prevOmegaGP[i, GP, :]
            if normNu != 0.0:
                self.beta[:] = np.sin(normNu)/normNu * self.diffNuGP \
                        + (1 - (np.sin(normNu) / normNu)) * (np.dot(self.nuGP,self.diffNuGP) / normNu) * self.nuGP / normNu \
                        + (2 * np.sin(0.5 * normNu) ** 2) / normNu**2 * np.cross(self.nuGP, self.diffNuGP)
            else:
                self.beta[:] = 0.0
            
            self.omegaGP[i,GP,:] = self.beta + self.xi
            self.gammaGP[:] = self.dphi_oE - np.dot(self.netRotMatGP,self.E3)
            
            # material model update 
            self.matModFGP[:,:] = self.netRotMatGP @ self.inMatModF @ self.netRotMatGP.T
            self.matModMGP[:,:] = self.netRotMatGP @ self.inMatModM @ self.netRotMatGP.T
            
            # internal stress measurement update
            self.forceGP[:] = self.matModFGP @ self.gammaGP
            self.momentGP[:] = self.matModMGP @ self.omegaGP[i, GP,:]
            self.forceGPW[i,GP,:] = self.forceGP
            self.momentGPW[i,GP,:] = self.momentGP

            # external distributed stress measurement update
            self.appForceGP[:] = np.dot(self.S['S0'], self.appForce[i,:,:]) * ratioLoadStep
            self.appMomentGP[:] = np.dot(self.S['S0'], self.appMoment[i,:,:]) * ratioLoadStep
          
            #---------------Element Matrix Calculation---------------------------------#
            for a in range(self.DOF//2): # a=alpha, b=beta as notes in reference material
                for b in range(self.DOF//2):
                    self.coeffSME[f'K{a}{b}'][:,:] += self.matModFGP[a,b] * self.outerS['S11'] * effecWt
                    
                    self.coeffSME[f'K{a}{b+self.DOF//2}'][:,:] += ((self.dphi_oE @ self.permutation[:,b,:]) @ self.matModFGP[:,a] - self.forceGP @ self.permutation[:,b,a]) * self.outerS['S10'] * effecWt
                    
                    self.coeffSME[f'K{a+self.DOF//2}{b}'][:,:] += ((self.dphi_oE @ self.permutation[:,a,:]) @ self.matModFGP[:,b] - self.forceGP @ self.permutation[:,a,b]) * self.outerS['S01'] * effecWt                        
                    
                    self.coeffSME[f'K{a+self.DOF//2}{b+self.DOF//2}'][:,:] += self.matModMGP[a,b] * self.outerS['S11'] * effecWt \
                                                                            + (self.momentGP @ self.permutation[:,a,b]) * self.outerS['S10'] * effecWt \
                                                                            + (- self.dphi_oE @ self.permutation[:,a,:] @ self.matModFGP @ self.permutation[:,b,:] @ self.dphi_oE) * self.outerS['S00'] * effecWt \
                                                                            + (self.dphi_oE @ self.permutation[:,a,:] @ self.permutation[:,b,:] @ self.forceGP) * self.outerS['S00'] * effecWt
                
                self.coeffCVE[f'F{a}'][:] += ((-self.forceGP[a] * self.S['S1']) + (self.appForceGP[a] * self.S['S0'])) * effecWt
                self.coeffCVE[f'F{a+self.DOF//2}'][:] += (((self.forceGP @ self.permutation[:,a,:] @ self.dphi_oE )+ self.appMomentGP[a]) * self.S['S0'] + (- self.momentGP[a]) * self.S['S1']) * effecWt
    
        for j in range(self.DOF):
            for k in range(self.DOF):
                self.SME[j:self.eqns_p_elem:self.DOF, k:self.eqns_p_elem:self.DOF] = self.coeffSME[f'K{j}{k}']
            self.CVE[j:self.eqns_p_elem:self.DOF] = self.coeffCVE[f'F{j}']
    
    def _applyEBC(self, i, iter):
        '''
        Apply essential boundary condition to element level stiffness matrices and column vector.
        i = element #.
        iter= iteration #.
        '''          
       #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if i in self.boundaryE['Elem']: # '0' for array of global nodes with boundary condition
            index = np.where(self.boundaryE['Elem']==i)[0] # [0] because index is a 1D array and np.where returns a tupple
            for k in index:
                indexNode = self.boundaryE['localDOF#'][k]
                # code for applying essential boundary condition for first iteration
                if iter==0:
                    value = self.SME[indexNode][indexNode]
                    self.SME[indexNode,:] = 0.0
                    self.CVE[:] = self.CVE - self.boundaryE['Values'][k] * self.SME[:,indexNode]
                    self.SME[:,indexNode] = 0.0
                    self.SME[indexNode,indexNode] = value
                    self.CVE[indexNode] = self.boundaryE['Values'][k] * value
                else:
                    value = self.SME[indexNode][indexNode]
                    self.SME[indexNode,:] = 0.0
                    self.CVE[:] = self.CVE - 0.0 * self.SME[:,indexNode]
                    self.SME[:,indexNode] = 0.0
                    self.SME[indexNode,indexNode] = value
                    self.CVE[indexNode] = 0.0

    def _applyNBC(self, i, loadfactor):
        '''
        Apply natural boundary condition to element level stiffness matrices and column vector.
        i = current element number.
        loadfactor = ratio of total NBC to be applied in this load step calculation.
        '''          
        #---------------Applying boundary condition----------------#
        # Nested If loops to handle where to apply boundary condition
        if i in self.boundaryN['Elem']:
            index = np.where(self.boundaryN['Elem']==i)[0] # [0] because index could be a 1D array and np.where returns a tupple
            for k in index:
                indexNode =  int(self.boundaryN['localDOF#'][k])
                self.CVE[indexNode] += self.boundaryN['Values'][k] * loadfactor

    def _globalSolver(self, iter, ratioLoadStep):
        '''
        Function to assemble and solve global matrices. It uses _elemMatComput() function to compute individual element matrices and then assembles them to form global matrix.
        It also uses sparse solver to solve for the change in displacements. It calls _elemMatComput, _appleEBC and _appleNBC sequentially.
        iter = iteration number.
        loadFactor = ratio of total NBC to be applied in this load step calculation.
        Returns
        changeSolution : incremental change in deformation. Needs to be added to previous configuration to get total deformtion.
        '''
        # Re-zero memory allocations
        self.sparSMG[:] = 0.0
        self.sparIIrow[:] = 0.0
        self.sparJJcol[:] = 0.0
        self.SMG[:,:] = 0.0
        self.CVG[:] = 0.0
        self.changeSolution[:] = 0.0
        
        # counter, for index position of vectorised matrix.
        m = 0 
        for i in range(self.NEL): # loop over elements
            self._elemMatComput(i, ratioLoadStep)
            self._applyEBC(i, iter)
            self._applyNBC(i, ratioLoadStep)
            # Global stiffness matrix assembly
            self.sparSMG[m:m+self.sizeElemMat] = self.SME.flatten() # value
            self.sparIIrow[m:m+self.sizeElemMat] = np.repeat(self.dofCON[i],self.shapeElemMat) # row index
            self.sparJJcol[m:m+self.sizeElemMat] = np.tile(self.dofCON[i],self.shapeElemMat) # column index
            m += self.sizeElemMat
            
            # global column vector assembly
            self.CVG[self.dofCON[i]] += self.CVE

        # create global sparse stiffness matrix from vectorised matrix
        self.SMG[:,:] = coo_array((self.sparSMG, (self.sparIIrow,self.sparJJcol)), shape=(self.shapeGM,self.shapeGM), dtype=np.float64).tocsr()
        # self.SMGdense[:,:] = self.SMG.todense() # for debugging purposes only, store dense form of global stiffness matrix
        # solve the linear algebra problem using sparse solver
        self.changeSolution[:] = spsolve(self.SMG, self.CVG)

        self.genF[:] = self.SMGdense @ self.changeSolution - self.CVG

    def _deltaConfig(self):
        '''
        This function separates translation and rotation response of each node into an array in dictionary.
        This function should be run after _globalSolver.
        '''
        for key in self.changeConfig.keys():
            self.changeConfig[key][:] = 0.0

        for i in range(self.DOF):
            self.changeConfig[f'{i}'][:] = self.changeSolution[i::self.DOF]
    
    def _updateConfig(self, iter):
        '''
        This function updates the response of the beam with latest solution.
        It should be called after running deltaConfig.
        iter: number of iteration. Initial configuration is the previous configuration for first iteration.
        returns
        newConfig: dictionary of np.ndarray.
        '''
        # Re-zero memory allocations
        if iter!=0:
            for key in self.newConfig.keys():
                self.previousConfig[key][:] = self.newConfig[key][:] # previous configuration of beam.
                self.newConfig[key][:] = 0.0
        else:
            self.previousConfig['0'][:] = self.phi_o[:,0]
            self.previousConfig['1'][:] = self.phi_o[:,1]
            self.previousConfig['2'][:] = self.phi_o[:,2]
        
        for key in self.rotMat.keys(): 
            self.rotMat[key][:,:] = 0.0
            self.rotVec[key][:] = 0.0 # same key
        self.phi_o[:,:] = 0.0
        self.changeRotation[:,:] = 0.0
        self.rotation[:,:] = 0.0
        self.rotationMat[:,:,:] = 0.0

        # update translational movement
        for i in range(self.DOF//2):
            self.newConfig[f'{i}'][:] = self.previousConfig[f'{i}'] + self.changeConfig[f'{i}']

        # update rotational movement
        for i in range(self.globalNodes):
            # previous configuration
            self.rotVec['previous'][0] = self.previousConfig['3'][i]
            self.rotVec['previous'][1] = self.previousConfig['4'][i]
            self.rotVec['previous'][2] = self.previousConfig['5'][i]
            self.rotMat['previous'][:,:] = rotTensor(self.rotVec['previous'])
            
            # change in configuration
            self.rotVec['change'][0] = self.changeConfig['3'][i]
            self.rotVec['change'][1] = self.changeConfig['4'][i]
            self.rotVec['change'][2] = self.changeConfig['5'][i]
            self.rotMat['change'][:,:] = rotTensor(self.rotVec['change'])
            self.changeRotation[i,:] = self.rotVec['change'] # store change in rotation vector at global node for use in next iteration

            # new configuration
            self.rotMat['new'][:,:] = self.rotMat['change'] @ self.rotMat['previous']
            self.rotationMat[i,:,:] = self.rotMat['new']
            self.rotVec['new'][:] = rotVector(self.rotMat['new'])

            # update configuration dictionary
            self.newConfig['3'][i] = self.rotVec['new'][0]
            self.newConfig['4'][i] = self.rotVec['new'][1]
            self.newConfig['5'][i] = self.rotVec['new'][2]

        self.phi_o[:,0] = self.newConfig['0'][:]
        self.phi_o[:,1] = self.newConfig['1'][:]
        self.phi_o[:,2] = self.newConfig['2'][:]
        self.rotation[:,0] = self.newConfig['3'][:]
        self.rotation[:,1] = self.newConfig['4'][:]
        self.rotation[:,2] = self.newConfig['5'][:]

    def _genForces(self):
        '''
        This function calculates generalised force (force and moments) from the given generalised displacements (translation and rotation).
        It uses geometric relations and constitutive relations to calculate generalised forces.

        Returns:
        force: dictionary with keys X, Y and Z of size (globalNodes,) storing forces at each global node.
        moment: dictionary with keys X, Y and Z of size (globalNodes,) storing moments at each global node.
        '''

        y = self.NEL * self.NGQP
        force = {f'{i}':np.zeros(shape=y, dtype=np.float64) for i in ['X', 'Y', 'Z']}
        moment = {f'{i}':np.zeros(shape=y, dtype=np.float64) for i in ['X', 'Y', 'Z']}
        Zaxis = np.zeros(shape=y, dtype=np.float64)

        for i in range(self.NEL):
            for GP in range(self.NGQP):
                index = i*self.NGQP + GP
                force['X'][index] = self.forceGPW[i,GP,0]
                force['Y'][index] = self.forceGPW[i,GP,1]
                force['Z'][index] = self.forceGPW[i,GP,2]
                moment['X'][index] = self.momentGPW[i,GP,0]
                moment['Y'][index] = self.momentGPW[i,GP,1]
                moment['Z'][index] = self.momentGPW[i,GP,2]
                Zaxis[index] = self.interp[:,GP] @ self.elemGlobalCoord[i]

        return force, moment, Zaxis
    
    def FEMSolve(self, iterCount, ratioLoadStep):
        '''
        This function calls other functions in this class to compute configurations and returns the error between configurations.
        '''
        
        self._globalSolver(iterCount, ratioLoadStep)
        self._deltaConfig()
        self._updateConfig(iterCount)

        error = calcErrorI(self.changeConfig, self.globalNodes)
        length = lengthCheck(self.globalNodes, self.newConfig)
        
        if self.SMGdense.all() == self.SMGdense.T.all():
            symmTan = 0
        else:
            symmTan = 1
        force, moment, Zaxis = self._genForces()
        return self.newConfig, error, length, symmTan, force, moment, Zaxis

    

