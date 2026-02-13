'''
Author:      Kartikay Shukla
File:        input.py
Created:     Sept 22, 2025 
LM:          Oct 21, 2025

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for verification #L3: Cantilever beam with rectangular cross section with point load along X and Y direction.
'''

import json
import numpy as np 

g = 9.81 

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

inputEBC = {'globalNode#': [0],
        'Tx': [0], # T for translation
        'Ty': [0], 
        'Tz': [0],
        'Rx': [0], # R for rotation
        'Ry': [0],
        'Rz': [0]
        }
inputNBC = {'globalNode#': [2],
        'Tx': [-5000], # T for translation, force, in Newtons
        'Ty': [-5000], 
        'Tz': [0],
        'Rx': [0], # R for rotation, moment
        'Ry': [0],
        'Rz': [0]
        }

inputForce = {'globalNode#': [0, 1, 2, 3, 4],
                 'X': [0, 0, 0, 0, 0],
                 'Y': [0, 0, 0, 0, 0],
                 'Z': [0, 0, 0, 0, 0]
                 }

inputMoment = {'globalNode#': [0, 1, 2, 3, 4],
                 'X': [0, 0, 0, 0, 0],
                 'Y': [0, 0, 0, 0, 0],
                 'Z': [0, 0, 0, 0, 0]
                 }


with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    