'''
Author:      Kartikay Shukla
File:        postpro.py
Created:     Oct 21, 2025 
LM:          Oct 22, 2025

DESCRIPTION
This file has a function that stores the exact solution of problem which can be used to compare FEM solutions.
It also has the post processing functions. It takes the solution from FEM function and prepares them for plotting. 
'''
import numpy as np


def postprocessing(elemGlobalCoord, solution):
    #---------------------Initialise----------------------#
    # x-axis values for plotting
    Zaxis = np.unique(elemGlobalCoord.flatten())
    
    # y-axis values for plotting
    Defl = {'X': solution['0'],
            'Y': solution['1'],
            'Z': solution['2']
            }
    Slope = {'X': solution['3'],
             'Y': solution['4'],
             'Z': solution['5']
             }


    return Zaxis, Defl, Slope

