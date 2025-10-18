'''
Author:      Kartikay Shukla
File:        FEM-HT_Problem_Eg3.2
Created:     July 24, 2025 
LM:          July 24, 2025

DESCRIPTION
This is an FEM Model for heat transfer problem Example 3.2 from the book Introduction to Linear Finite Element Method, Second Edition. 

This script contains functions for plotting different solutions on the same plot. It is only good for plotting 9 solutions. Plotting more than 9 curves would cause the colors to repeat and create confusion.
'''

from Eg4_5_postpro import *
import matplotlib.pyplot as plt
import itertools as iter

def plotExact(): #for plotting exact solution
    exactsoln = exact() # calling exact solution function
    fig, ax = plt.subplots() # this allows us to use the same axis on the figure to plot multiple times
    ax.plot(exactsoln[0], exactsoln[1], label='Exact', color='red', linewidth=2, marker='o')

    # Add titles and labels
    ax.set_title('Eg 4.5')
    ax.set_xlabel('x')
    ax.set_ylabel('Deflection along the beam')
    ax.minorticks_on()

    # Show major and minor grid
    ax.grid(True)
    ax.grid(True, which='minor')
    

    return fig, ax

plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Above color palette is same as Tableau 10 color palette except red. The colors are presented in hexadecimal code. 
    # There corresponding description is [Muted Blue, Orange, Green, Purple, Brown, Pink, Gray, Olive, Teal]. 
    # This color palette is chosen for accessibility (colorblind freindly), high contrast between each color and well-suited for categorical data.

plotColor = iter.cycle(plotColors)

def plotResult(outputPro, NEPL, plotFig=None, plotAxes=None):
    # outputPro:This is the processed output. It is returned by the postprocessing function as an array.
    # NEPL: number of elements. This is an input.
    # plotFig: This is the figure. It is the output of the plotExact() function above.
    # plotAxes: This is the axes. It is the output of the plotExact() function above.

    if plotFig is None or plotAxes is None:
        plotFig, plotAxes = plotExact()

    plotAxes.plot(outputPro[0], outputPro[1], label=f'{NEPL} Element', color=next(plotColor), linewidth=2, marker='o')

    # Show legend
    plotAxes.legend()
    
    return plotFig, plotAxes

def plotGen(x, y, legend, label, plotGenFig=None, plotGenAx=None):
    # x: Array of values for x axis.
    # y: Array of values for y axis.
    # legend: number of elements. This is an input. Name for different curves apearing in legend.
    # plotGenFig: This is the figure. It is the output of the plotExact() function above.
    # plotGenAx: This is the axes. It is the output of the plotExact() function above.
    # label = array with labels for title, x axis and yaxis in order.

    if plotGenFig is None or plotGenAx is None:
        plotGenFig, plotGenAx = plt.subplots() # this allows us to use the same axis on the figure to plot multiple times
    
    plotGenAx.plot(x, y, label=f'{legend} Element', color=next(plotColor), linewidth=2, marker='o')

    # Add titles and labels
    plotGenAx.set_title(label[0])
    plotGenAx.set_xlabel(label[1])
    plotGenAx.set_ylabel(label[2])
    plotGenAx.minorticks_on()

    # Show major and minor grid
    plotGenAx.grid(True)
    plotGenAx.grid(True, which='minor')

    # Show legend
    plotGenAx.legend()
    
    return plotGenFig, plotGenAx



