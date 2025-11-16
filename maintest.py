'''
Author:      Kartikay Shukla
File:        main.py
Created:     October 6, 2025 
LM:          November 5, 2025 

Description
This file contains code for solving Finite strain beam formulation using FE method.
Refer to Dr. Greg Payette's notes and Dr Simo's paper (part I and II) for details on formula for calculation.
'''
from gen.gen_mesh1D import *
from gen.gen_utilities import *
from solver import *
from input.input import *
from postpro import *
from gen.gen_plot import *

# Finite element parameters
NEL=10
NNPEL=2
DOFPN = 6
iterations = 10
convergence = 1E-5
loadSteps = 1

# Load stepping parameters
loadStep = {'Steps':loadSteps, 
        'Iterations': iterations, 
        'Convergence': convergence}

# intialise input
geom, material, boundaryCondE, boundaryCondN, inAppForce, inAppMoment = inputJson()

ECON, elemGlobalCoord = mesh1D(geom['L'], NEL, NNPEL)

EBC = {f'{key}': values(NEL, ECON, boundaryCondE[key]) for key in boundaryCondE.keys()}
NBC = {f'{key}': values(NEL, ECON, boundaryCondN[key]) for key in boundaryCondN.keys()}

inMatModF = np.diag([material['G'] * geom['A1'] * material['Ks'], 
                     material['G'] * geom['A2'] * material['Ks'], 
                     material['E'] * geom['A3']
                     ])
inMatModM = np.diag([material['E'] * geom['I1'], 
                     material['E'] * geom['I2'], 
                     material['G'] * geom['J'] 
                     ])

# preparing applied force and moment inout for calculation
appForce = appGenForce(inAppForce, DOFPN, NNPEL, NEL, ECON)
appMoment = appGenForce(inAppMoment, DOFPN, NNPEL, NEL, ECON)

# # initialising class object
solverObj = FEMSolver(DOFPN, NNPEL, NEL, ECON, elemGlobalCoord, EBC, NBC, appForce, appMoment, inMatModF, inMatModM)
solution = solverObj.FEMSolve(loadStep=loadStep)

result = postprocessing(elemGlobalCoord, solution)

label3DDefl = ['Deflection of beam', 'Deflection along Z', 'Deflection along X', 'Deflection along Y']
legDefl = 'Defllection of Centerline of beam'
labelSlp = ['Slope along the beam', 'Length along the beam', 'Slope']
labelDefl = ['Deflection along the beam', 'Length along the beam', 'Deflection']

plot3D(result[1]['Z'], result[1]['X'], result[1]['Y'],  legDefl, label=label3DDefl)

plotGen(result[1]['Z'], result[1]['X'], 'X wrt Z Coordinates', labelDefl)
plotGen(result[0], result[1]['X'], 'X', labelDefl)
plotGen(result[0], result[1]['Y'], 'Y', labelDefl)
plotGen(result[0], result[1]['Z'], 'Z', labelDefl)
plotGen(result[0], result[2]['X'], 'X', labelSlp)
plotGen(result[0], result[2]['Y'], 'Y', labelSlp)
plotGen(result[0], result[2]['Z'], 'Z', labelSlp)

plt.show()