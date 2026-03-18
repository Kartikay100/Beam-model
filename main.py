'''
Author:      Kartikay Shukla
File:        main.py
Created:     October 6, 2025 
LM:          March 11, 2026

Description
This file contains code for solving Finite strain beam formulation using FE method.
'''

from gen.gen_compCost import *
start = tic()
from gen.gen_mesh1D import *
from gen.gen_utilities import *
from solver import *
from input.input import *
from boundary import * 
from postpro import *
from gen.gen_plot import *

# intialise input
cond, geom, material, boundaryCondE, boundaryCondN, inAppForce, inAppMoment = inputJson()
 
# Finite element parameters
NEL = cond['NEL']
NNPEL = cond['NNPEL']
DOFPN = cond['DOFPN']
iterations = cond['iterations']
convergence = cond['convergence']
loadsteps = cond['loadsteps']
follower = cond['follower']
postprocess = cond['postprocess']

inMatModF = np.diag([material['G'] * geom['A1'] * material['Ks'], 
                     material['G'] * geom['A2'] * material['Ks'], 
                     material['E'] * geom['A3']
                     ])

inMatModM = np.diag([material['E'] * geom['I1'], 
                     material['E'] * geom['I2'], 
                     material['G'] * geom['J'] 
                     ])

ECON, elemGlobalCoord = mesh1DLGL(geom['L'], NEL, NNPEL)


# initialising boundary class object and preparing boundary conditions for applying
boundaryEClass = boundary(DOFPN, ECON, boundaryCondE)
EBC = boundaryEClass.boundaryHandler('EBC')
boundaryNClass = boundary(DOFPN, ECON, boundaryCondN)
NBC = boundaryNClass.boundaryHandler('NBC')

# preparing applied force and moment inout for calculation
appForce = genForceHandler(DOFPN, NNPEL, NEL, ECON, inAppForce)
appMoment = genForceHandler(DOFPN, NNPEL, NEL, ECON, inAppMoment)

# initialising class object
solverObj = FEMSolver(NNPEL, NEL, ECON, elemGlobalCoord, EBC, NBC, appForce, appMoment, inMatModF, inMatModM)

loadstep_conv = 0
symmTang  = np.zeros(shape=[loadsteps, iterations])
for loadstep in range(loadsteps):
        print('loadstep=',loadstep)
        ratio = (loadstep+1)/loadsteps
        for iter in range(iterations):
                print('iter=',iter)
                solution, error, length, symmTang[loadstep, iter], force, moment, Zaxis= solverObj.FEMSolve(iter, ratio)
                print('error=',error, 'at loadstep=',loadstep+1, 'iter=',iter+1, 'at ratioLoadstep=',ratio)
                print('solution=',solution)
                if np.isclose(length, geom['L']):
                        print('Length of beam is close before and after deformation. Current length is ', length,'.')
                else: 
                        print('Length of beam is different before and after deformation. Current length is ', length,'.')                     
                if error < convergence:
                        print('Solution convereged in ', iter+1, 'iterations')
                        print('solution=',solution)
                        loadstep_conv += 1
                        break
                else:
                        continue

print('Loadsteps converged = ',loadstep_conv)
tanSymm = np.where(symmTang==1)
if tanSymm[0].size >0:
    print('Tangent matrix was not symmetric at loadstep=',tanSymm[0],'iteration=',tanSymm[1], 'respectively.')
else:
    print('Tangent matrix was symmetric in all iterations.')

result = postprocessing(elemGlobalCoord, solution)
label3DDefl = ['Deflection of beam', 'Deflection along Z', 'Deflection along X', 'Deflection along Y']
legDefl = 'Defllection of Centerline of beam'
labelSlp = ['Slope along the beam', 'Length along the beam', 'Slope']
labelDefl = ['Deflection along the beam', 'Length along the beam', 'Deflection']

# plot3D(result[1]['Z'], result[1]['X'], result[1]['Y'],  legDefl, label=label3DDefl)
# plotGen(result[1]['Z'], result[1]['X'], 'X wrt Z Coordinates', labelDefl)
# plotGen(result[1]['Z'], result[1]['Y'], 'Y wrt Z Coordinates', labelDefl)
# plotGen(result[0], result[1]['X'], 'X', labelDefl)
# plotGen(result[0], result[1]['Y'], 'Y', labelDefl)
# plotGen(result[0], result[1]['Z'], 'Z', labelDefl)
# plotGen(result[0], result[2]['X'], 'X', labelSlp)
# plotGen(result[0], result[2]['Y'], 'Y', labelSlp)
# plotGen(result[0], result[2]['Z'], 'Z', labelSlp)


if postprocess == True:
        x = 1 # random factor
        labelForce = ['Force along the beam', 'Length along the beam', 'Force']
        labelMoment = ['Moment along the beam', 'Length along the beam', 'Moment']
        print('Sum of forces along X axis=', np.sum(force['X'])/ x) 
        print('Sum of forces along Y axis=', np.sum(force['Y'])/ x)
        print('Sum of forces along Z axis=', np.sum(force['Z'])/ x)
        print('Sum of moments about X axis=', np.sum(moment['X'])/ x)
        print('Sum of moments about Y axis=', np.sum(moment['Y'])/ x)
        print('Sum of moments about Z axis=', np.sum(moment['Z'])/ x)
        plotGen(Zaxis, force['X']/ x, 'X', labelForce)
        plotGen(Zaxis, force['Y']/ x, 'Y', labelForce)
        plotGen(Zaxis, force['Z']/ x, 'Z', labelForce)
        plotGen(Zaxis, moment['X']/ x, 'X', labelMoment)
        plotGen(Zaxis, moment['Y']/ x, 'Y', labelMoment)
        plotGen(Zaxis, moment['Z']/ x, 'Z', labelMoment)
        
time = toc(start)
plt.show()