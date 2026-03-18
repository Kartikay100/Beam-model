'''
Author:      Kartikay Shukla
File:        input.py
Created:     October 6, 2025 
LM:          March 03, 2026

DESCRIPTION
This file has a single function with all the input parameters compiled. Output of the function returns an array of array.
'''

import numpy as np
import json

def inputJson():
    with open('input/inputJSON.json', mode='r', encoding='utf-8') as read_file:
        inputs  = json.load(read_file)
    
    cond = inputs[0]
    
    geom = inputs[1]
    
    material = inputs [2]

    boundaryCondE = inputs[3]

    boundaryCondN = inputs[4]

    appForce = inputs[5]
    
    appMoment = inputs[6]

    return cond, geom, material, boundaryCondE, boundaryCondN, appForce, appMoment

