'''
Author:      Kartikay Shukla
File:        input.py
Created:     Sept 22, 2025 
LM:          Oct 21, 2025

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for verification #L2: Cantilever beam with rectangular cross section with point load along Z direction (axial compression).
'''

import json
import numpy as np 

g = 9.81 

inputCond = {'NEL': 2,
        'NNPEL': 2,
        'DOFPN': 6,
        'iterations': 1,
        'convergence': 1E-5,
        'loadsteps': 1,
        'follower': False,
        'postprocess': False
        }

inputGeom = {'L': 1,   # length of beam, m
        'b': 0.05, # height of beam, m,
        'h': 0.05, # heaight of beam m
        'A1': 0.05, # area of beam in m^2
        'A2': 0.05, # area of beam in m^2
        'A3': 0.0025, # cross-sectional area of beam in m^2
        'I1': 5.20833E-7, # moment of inertia m^4, b*h^3/12
        'I2': 5.20833E-7, # moment of inertia m^4,  inertia about E2xE2
        'J': 1.04166E-6, # polar moment of inertia m^4, pi*(D^4-d^4)/32, inertia about E3xE3, J=I1+I2
        }

inputMatProp = {'E': 210E9, # Youngs modulus, Pa
                'rho': 7850, # density of steel, kg/m^3
                'G': 80E9, # shear modulus, Pa
                'Ks': 1.2E4 # shear correction coefficient, unitless
                }

inputEBC = {'globalNode#': [[0],[6]],
        'Values': [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]]
        }
inputNBC = {'globalNode#': [[2],[1]],
        'Values': [[2,-5000]],
        }

inputForce = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
        'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }
inputMoment = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
        'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }


with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    