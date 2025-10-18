'''
Author:      Kartikay Shukla
File:        input.py
Created:     Sept 22, 2025 
LM:          Sept 22, 2025

DESCRIPTION
This is an FEM Model for Timoshenko beam problem Example 4.5 from the book Introduction to Linear Finite Element Method, Second Edition.

This file creates JSON file from the input dictionary.
'''

import json
import numpy as np

rho = 7850
g = 9.81 

inputGeom = {'L': 10,   # length of beam, m
        'I': 1.3465775404794298e-05,# moment of inertia m^4, pi*(D^4-d^4)/64
        'Ri': 0.1016, # inner radius of pipe, m
        'Ro': 0.1397, # outer radius of pipe, m
        'A': 0.0288823, # cross-sectional area of beam in m^2
        }

inputMatProp = {'E': 210E9, # Youngs modulus, Pa
                'rho': rho, # density of steel, kg/m^3
                'k': 10E5, # spring stiffness, lbf/ft
                'G': 0, # shear modulus, psi
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
        'Tx': [-500], # T for translation, force, in Newtons
        'Ty': [0], 
        'Tz': [0],
        'Rx': [0], # R for rotation, moment
        'Ry': [0],
        'Rz': [0]
        }

inputForce = {'globalNode#': [0, 1, 2],
                 'X': [0, 0, 0],
                 'Y': [0, 0, 0],
                 'Z': [0, 0, 0]
                 }

inputMoment = {'globalNode#': [0, 1, 2],
                 'X': [0, 0, 0],
                 'Y': [0, 0, 0],
                 'Z': [0, 0, 0]
                 }


with open("inputJSON.json", mode="w", encoding="utf-8") as write_file:
    json.dump([inputGeom, inputMatProp, inputEBC, inputNBC, inputForce, inputMoment], write_file, indent=2)
    