'''
Author:      Kartikay Shukla
File:        gen_interpFunction.py
Created:     July 31, 2025 
LM:          Aug 5, 2025

DESCRIPTION
This file contains functions for determining time elapsed during the computation time.
'''

import time

def tic():
    '''
    Function returns the floating point number representing high-resolution monotonic wall clock time in seconds.
    It should be used at the beginning to program to record wall clock time when the program starts running.
    Output used in the toc() defined below.
    '''
    return time.perf_counter()

def toc(start):
    '''
    Function prints and returns time in seconds. It should be used at the end of the program to record the wall clock time elapsed to run the program.
    
    start: seconds in floating point number. Output of tic() defined above.
    '''
    elapsed_time = time.perf_counter() - start
    print('Computation time of this program is:',elapsed_time, 'seconds')
    return elapsed_time