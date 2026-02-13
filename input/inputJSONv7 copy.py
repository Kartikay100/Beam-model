'''
Author:      Kartikay Shukla
File:        input.py
Created:     Sept 22, 2025 
LM:          Oct 21, 2025

DESCRIPTION
This file creates JSON file from the input dictionary.
Input data for verification #7- Cantilever beam with point moment load about Y direction for full circular deformation.
'''

import json
import numpy as np 

g = 9.81 
moment = 4*np.pi
Im = 7850 * 0.0000109477 * 1**3 / 3

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
                'Im': Im, # mass moment of inertia of rod about its end mL^2/3
                }

inputEBC = {'globalNode#': [0],
        'Tx': [0], # T for translation
        'Ty': [0], 
        'Tz': [0],
        'Rx': [0], # R for rotation
        'Ry': [0],
        'Rz': [0]
        }

inputNBC = {'globalNode#': [5],
        'Tx': [0], # T for translation, force, in Newtons
        'Ty': [0], 
        'Tz': [0],
        'Rx': [0], # R for rotation, moment
        'Ry': [moment],
        'Rz': [0]
        }

inputForce = {'globalNode#': [0, 1, 2, 3, 4, 5],
                 'X': [0, 0, 0, 0, 0, 0],
                 'Y': [0, 0, 0, 0, 0, 0],
                 'Z': [0, 0, 0, 0, 0, 0]
                 }

inputMoment = {'globalNode#': [0, 1, 2, 3, 4, 5],
                 'X': [0, 0, 0, 0, 0, 0],
                 'Y': [0, 0, 0, 0, 0, 0],
                 'Z': [0, 0, 0, 0, 0, 0]
                 }


with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    