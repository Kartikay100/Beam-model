'''
Author:      Kartikay Shukla
File:        input.py
Created:     October 6, 2025 
LM:          October 6, 2025

DESCRIPTION
This is an FEM Model for Timoshenko beam problem Example 4.5 from the book Introduction to Linear Finite Element Method, Second Edition.

This file has a single function with all the input parameters compiled. Output of the function returns an array of array. First position is geometry array, second for boundary condition array and third for material property array.
'''

# from .inputGeom import *
# from .inputBC import *
# from .inputMaterial import *
import numpy as np
import json

def inputJson():
    with open('input/inputJSON.json', mode='r', encoding='utf-8') as read_file:
        inputs  = json.load(read_file)
    
    geom = inputs[0]
    
    material = inputs [1]

    boundaryCondE = inputs[2]

    boundaryCondN = inputs[3]

    appForce = inputs[4]
    
    appMoment = inputs[5]

    return geom, material, boundaryCondE, boundaryCondN, appForce, appMoment


def inputPython(dispL):
    '''
    This function reads input from individual input python files and compiles them into a output.
    '''
    geom = geomBeamFEM()

    material = matBeamFEM()

    boundaryCond = boundaryBeamFEM(geom['L'], material['k'], dispL)

    return geom, boundaryCond, material
