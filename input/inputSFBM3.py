'''
Author:      Kartikay Shukla
File:        input.py
Created:     March 03, 2026
LM:          March 03, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for SFBM verification # 3- Pin-pin beam with UDL.
'''

import json
import numpy as np 

g = 9.81 

inputCond = {'NEL': 10,
        'NNPEL': 10,
        'DOFPN': 6,
        'iterations': 10,
        'convergence': 1E-5,
        'loadsteps': 1,
        'follower': False,
        'postprocess': True
        }

inputGeom = {'L': 100,   # length of beam, in
        'b': 1, # width of rod, in
        'h': 1, # height of rod, in
        'A1': 100, # area of beam in in^2, of plane E1xE1
        'A2': 100, # area of beam in m^2, of plane E2xE2
        'A3': 1, # cross-sectional area of beam in m^2, of plane E3xE3
        'I1': 1/12, # moment of inertia in^4, bh^3/12, inertia about E1xE1
        'I2': 1/12, # moment of inertia in^4, b^3h/12, inertia about E2xE2
        'J': 1/6, # polar moment of inertia in^4, inertia about E3xE3, J=I1+I2
        }

inputMatProp = {'E': 30E6, # Youngs modulus, psi
                'rho': 0.28359929045986, # density of steel, kg/m^3
                'G': 12E6, # shear modulus, psi
                'Ks': 1, # shear correction factor, unitless
                }

inputEBC = {'globalNode#': [[0,90], [3,3]], # [node number], [no of boundary values at node]    
        'Values': [[0,0],[1,0],[2,0],[0,0],[1,0],[2,0]] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }
inputNBC = {'globalNode#': [[0], [0]], 
            'Values': [[0, 0]] 
        }

inputForce = {'globalNode#': [np.array(range(91)).tolist(), np.tile(np.array([1]), 91).tolist()], # [node number], [no of boundary values at node]
            'Values': np.tile([1, 1, 0], (91,1)).tolist() # [DOF #, value], 0=Tx
        }
inputMoment = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }

with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    