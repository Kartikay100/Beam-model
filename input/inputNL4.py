'''
Author:      Kartikay Shukla
File:        input.py
Created:     Sept 22, 2025 
LM:          Oct 21, 2025

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for verification # 10- Cantilever beam with point torque load (about Z direction, twisting of beam, non-linear)
'''

import json
import numpy as np 

g = 9.81 
twist = 95.74373063110957 # T = theta * G * J / L for theta = 3600 deg = 10 wraps

inputCond = {'NEL': 100,
        'NNPEL': 2,
        'DOFPN': 6,
        'iterations': 10,
        'convergence': 1E-8,
        'loadsteps': 1,
        'follower': False,
        'postprocess': False
        }

inputGeom = {'L': 1,   # length of beam, m
        'D': 3.73216E-3, # diameter of rod, m
        'A1': 3.73216E-3, # area of beam in m^2, of plane E1xE1
        'A2': 3.73216E-3, # area of beam in m^2, of plane E2xE2
        'A3': 1.09477E-5, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': 9.52380952E-12, # moment of inertia m^4, pi*(D^4-d^4)/64, inertia about E1xE1
        'I2': 9.52380952E-12, # moment of inertia m^4, pi*(D^4-d^4)/64, inertia about E2xE2
        'J': 1.904761E-11, # polar moment of inertia m^4, pi*(D^4-d^4)/32, inertia about E3xE3, J=I1+I2
        }

inputMatProp = {'E': 210E9, # Youngs modulus, Pa
                'rho': 7850, # density of steel, kg/m^3
                'G': 80E9, # shear modulus, Pa
                'Ks': 1, # shear correction factor, unitless
                }

inputEBC = {'globalNode#': [[0], [6]], # [node number], [no of boundary values at node]
            'Values': [[0, 0],[1,0],[2,0],[3,0],[4,0],[5,0]] # [DOF #, value]
        }
inputNBC = {'globalNode#': [[100], [1]], # [node number], [no of boundary values at node]
            'Values': [[5, twist]] # [DOF #, value], 5=Rz
        }

inputForce = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx
        }
inputMoment = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }


with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    