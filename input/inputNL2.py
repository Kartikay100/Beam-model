'''
Author:      Kartikay Shukla
File:        input.py
Created:     Sept 22, 2025 
LM:          Feb 20, 2026

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for verification #7 - Simply supported beam under distributed load. Eg. 5.2.1 in NL-FEM-Reddy book 2nd edition. Half beam analysed.
'''

import json

inputCond = {'NEL': 8,
        'NNPEL': 2,
        'DOFPN': 6,
        'iterations': 10,
        'convergence': 1E-3,
        'loadsteps': 10,
        'follower': False,
        'postprocess': False
        }

inputGeom = {'L': 50,   # length of beam, in
        'b': 1, # inner diameter of beam, in
        'h': 1, # outer diameter of beam, in
        'A1': 50, # area of beam in in^2, of plane E1xE1
        'A2': 50, # area of beam in in^2, of plane E2xE2
        'A3': 1, # cross-sectional area of beam in in^2, of plane E3xE3
        'I1': 1/12, # moment of inertia m^4, inertia about E1xE1
        'I2': 1/12, # moment of inertia m^4, inertia about E2xE2
        'J': 1/6, # polar moment of inertia in^4, inertia about E3xE3, J=I1+I2
        }

inputMatProp = {'E': 30E6, # Youngs modulus, psi
                'G': 12E6, # shear modulus, psi
                'Ks': 1, # unitless
                }

# Need to apply 6 EBCs always.
inputEBC = {'globalNode#': [[0,8], [4,2]], # [node number], [no of boundary values at node]
            'Values': [[0,0],[1,0],[3,0],[5,0],[2,0],[4,0]] # [DOF #, value] # 1=Ty, 0=Tx, 4=Ry
        }
inputNBC = {'globalNode#': [[0], [0]], # [node number], [no of boundary values at node]
            'Values': [[0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }

inputForce = {'globalNode#': [[0,1,2,3,4,5,6,7,8], [1,1,1,1,1,1,1,1,1]], # [node number], [no of boundary values at node]
            'Values': [[0, 10, 0],[0, 10, 0],[0, 10, 0],[0, 10, 0],[0, 10, 0],[0, 10, 0],[0, 10, 0],[0, 10, 0],[0, 10, 0]] # [DOF #, value], 0=Tx
        }
inputMoment = {'globalNode#': [[0], [1]], # [node number], [no of boundary values at node]
            'Values': [[0, 0, 0]] # [DOF #, value], 0=Tx, 1=Ty, 3=Rx
        }


with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputCond, inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    