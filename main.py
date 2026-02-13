'''
Author:      Kartikay Shukla
File:        main.py
Created:     October 6, 2025 
LM:          January 21, 2026

Description
This file contains code for solving Finite strain beam formulation using FE method.
Refer to Dr. Greg Payette's notes and Dr Simo's paper (part I and II) for details on formula for calculation.
'''

from gen.gen_compCost import *
start = tic()
from gen.gen_mesh1D import *
from gen.gen_utilities import *
from solver_messy import *
from input.input import *
from boundary import * 
from postpro import *
from gen.gen_plot import *

# Finite element parameters
NEL = 5
NNPEL = 2
DOFPN = 6
iterations = 10
convergence = 1E-5
loadsteps = 2

# intialise input
geom, material, boundaryCondE, boundaryCondN, inAppForce, inAppMoment = inputJson()

inMatModF = np.diag([material['G'] * geom['A1'] * material['Ks'], 
                     material['G'] * geom['A2'] * material['Ks'], 
                     material['E'] * geom['A3']
                     ])

inMatModM = np.diag([material['E'] * geom['I1'], 
                     material['E'] * geom['I2'], 
                     material['G'] * geom['J'] 
                     ])

ECON, elemGlobalCoord = mesh1DLGL(geom['L'], NEL, NNPEL)
# print('ECON=',ECON)

# initialising boundary class object and preparing boundary conditions for applying
boundaryEClass = boundary(DOFPN, NNPEL, NEL, ECON, boundaryCondE)
countE = boundaryEClass.countBC()
EBC = boundaryEClass.valueBC(countE)
boundaryNClass = boundary(DOFPN, NNPEL, NEL, ECON, boundaryCondN)
countN = boundaryNClass.countBC()
NBC = boundaryNClass.valueBC(countN)
# print('NBC=',NBC)

# preparing applied force and moment inout for calculation
appForce = applied(inAppForce, ECON, DOFPN, NNPEL, NEL)
appMoment = applied(inAppMoment, ECON, DOFPN, NNPEL, NEL)

# initialising class object
solverObj = FEMSolver(DOFPN, NNPEL, NEL, ECON, elemGlobalCoord, EBC, NBC, appForce, appMoment, inMatModF, inMatModM)

loadstep_conv = 0
for loadstep in range(loadsteps):
        print('loadstep=',loadstep)
        ratio = (loadstep+1)/loadsteps
        for iter in range(iterations):
                print('iter=',iter)
                solution, error, length = solverObj.FEMSolve(iter, ratio)
                print('error=',error, 'at loadstep=',loadstep+1, 'iter=',iter+1, 'at ratioLoadstep=',ratio)
                print('solution=',solution)
                print('solution [Tx]=',solution['0'][-1],
                        '[Ty]=',solution['1'][-1],
                        '[Tz]=',solution['2'][-1],
                        '[Rx]=',solution['3'][-1],
                        '[Ry]=',solution['4'][-1],
                        '[Rz]=',solution['5'][-1], 'at loadstep=',loadstep+1, 'iter=',iter+1, 'at ratioLoadstep=',ratio)
                if np.isclose(length, geom['L']):
                        print('Length of beam is close before and after deformation. Current length is ', length)
                else: 
                        print('Length of beam is different before and after deformation. Current length is ', length)                     
                if error < convergence:
                        print('Solution convereged in ', iter+1, 'iterations')
                        # print('solution [Tz]=', f'{solution['2'][-1]}')
                        # print('solution [Rz]=', solution['5'][-1])
                        loadstep_conv += 1
                        break
                else:
                        continue

print('Loadsteps converged = ',loadstep_conv)
result = postprocessing(elemGlobalCoord, solution)
label3DDefl = ['Deflection of beam', 'Deflection along Z', 'Deflection along X', 'Deflection along Y']
legDefl = 'Defllection of Centerline of beam'
labelSlp = ['Slope along the beam', 'Length along the beam', 'Slope']
labelDefl = ['Deflection along the beam', 'Length along the beam', 'Deflection']

plot3D(result[1]['Z'], result[1]['X'], result[1]['Y'],  legDefl, label=label3DDefl)
plotGen(result[1]['Z'], result[1]['X'], 'X wrt Z Coordinates', labelDefl)
# plotGen(result[1]['Z'], result[1]['Y'], 'Y wrt Z Coordinates', labelDefl)
# plotGen(result[0], result[1]['X'], 'X', labelDefl)
# plotGen(result[0], result[1]['Y'], 'Y', labelDefl)
# plotGen(result[0], result[1]['Z'], 'Z', labelDefl)
# plotGen(result[0], result[2]['X'], 'X', labelSlp)
plotGen(result[0], result[2]['Y'], 'Y', labelSlp)
# plotGen(result[0], result[2]['Z'], 'Z', labelSlp)

time = toc(start)
# plt.show()