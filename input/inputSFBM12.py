'''
Author:      Kartikay Shukla
File:        input.py
Created:     March 08, 2026
LM:          March 08, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for SFBM verification # 12- SSB with point force and point moment load.
'''

import json
import numpy as np 

x = 9964.016384 # conversion number for tonnesF to Newton
g = 9.81 

inputCond = {'NEL': 8,
        'NNPEL': 11,
        'DOFPN': 6,
        'iterations': 10,
        'convergence': 1E-15,
        'loadsteps': 1,
        'follower': False,
        'postprocess': True
        }

inputGeom = {'L': 8,   # length of beam, m
        'b': 0.01, # width of rod, m
        'h': 0.01, # height of rod, m
        'A1': 0.08, # area of beam m^2, of plane E1xE1
        'A2': 0.08, # area of beam m^2, of plane E2xE2
        'A3': 1E-4, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': 1/12E-8, # moment of inertia m^4, bh^3/12, inertia about E1xE1
        'I2': 1/12E-8, # moment of inertia m^4, b^3h/12, inertia about E2xE2
        'J': 1/6E-8, # polar moment of inertia m^4, inertia about E3xE3, J=I1+I2
        }

inputMatProp = {'E': 210E9, # Youngs modulus, Pa
                'rho': 7850, # density of steel, kg/m^3
                'G': 80E9, # shear modulus, Pa
                'Ks': 1, # shear correction factor, unitless
                }

inputEBC = {'globalNode#': [[0, 60], [3,2]], # [node number], [no of boundary values at node]    
        'Values': [[0,0],[1,0],[2,0],[0,0],[1,0]] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }
inputNBC = {'globalNode#': [[40, 80], [1, 1]], 
            'Values': [[4, -10*x], [0, -20*x]] 
        }

inputForce = {'globalNode#': [np.array(range(0, 41)).tolist(), np.tile(np.array([1]), 41).tolist()], # [node number], [no of boundary values at node]
            'Values': np.tile([0, -4*x, 0], (41,1)).tolist() # [DOF #, value], 0=Tx
        }
inputMoment = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }

with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    