'''
Author:      Kartikay Shukla
File:        main.py
Created:     October 6, 2025 
LM:          October 8, 2025 

Description
This file contains code for solving Finite strain beam formulation using FE method.
Refer to Dr. Greg Payette's notes and Dr Simo's paper (part I and II) for details on formula for calculation.
'''
from gen.gen_mesh1D import *
from gen.gen_utilities import *
from solver import *
from input.input import *
from boundary import *

NEL=2
NNPEL=2
DOFPN = 6
NGQP = 1
iterations = 100
convergence = 1E-6

# intialise input
geom, material, boundaryCondE, boundaryCondN, inAppForce, inAppMoment= inputJson()
inMatModF = np.identity(DOFPN//2) * material['E']
inMatModM = np.identity(DOFPN//2) * material['G']

ECON, elemGlobalCoord = mesh1D(geom['L'], NEL, NNPEL)

# initialising boundary class object and preparing boundary conditions for applying
boundaryEClass = boundary(DOFPN, NNPEL, NEL, ECON, boundaryCondE)
countE = boundaryEClass.countBC()
EBC = boundaryEClass.valueBC(countE)
boundaryNClass = boundary(DOFPN, NNPEL, NEL, ECON, boundaryCondN)
countN = boundaryNClass.countBC()
NBC = boundaryNClass.valueBC(countN)
# preparing applied force and moment inout for calculation
appForce = applied(inAppForce, ECON, DOFPN, NNPEL, NEL)
appMoment = applied(inAppMoment, ECON, DOFPN, NNPEL, NEL)


# initialising class object
solverObj = FEMSolver(DOFPN, NNPEL, NEL, NGQP, ECON, elemGlobalCoord, EBC, NBC, appForce, appMoment, inMatModF, inMatModM)

for iter in range(iterations):
    
    error , solution = solverObj.FEMSolve(iter)
    if error<=convergence:
        break
    else:
        continue