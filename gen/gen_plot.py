'''
Author:      Kartikay Shukla
File:        FEM-HT_Problem_Eg3.2
Created:     July 24, 2025 
LM:          July 24, 2025

DESCRIPTION
This script contains functions for plotting different solutions on the same plot. 
It is only good for plotting 9 solutions. Plotting more than 9 curves would cause the colors to repeat and create confusion.
'''


import matplotlib.pyplot as plt
import itertools as iter


plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Above color palette is same as Tableau 10 color palette except red. The colors are presented in hexadecimal code. 
    # There corresponding description is [Muted Blue, Orange, Green, Purple, Brown, Pink, Gray, Olive, Teal]. 
    # This color palette is chosen for accessibility (colorblind freindly), high contrast between each color and well-suited for categorical data.

plotColor = iter.cycle(plotColors)

def plotGen(x, y, legend, label, plotGenFig=None, plotGenAx=None):
    '''
    x: Array of values for x axis.
    y: Array of values for y axis.
    legend: number of elements. This is an input. Name for different curves apearing in legend.
    plotGenFig: This is the figure. It could be used to plot the curves on the same figure.
    plotGenAx: This is the axes. It could be used to plot the curves on the same axes.
    label = array with labels for title, x axis and yaxis in order.
    '''

    if plotGenFig is None or plotGenAx is None:
        plotGenFig, plotGenAx = plt.subplots() # this allows us to use the same axis on the figure to plot multiple times
    
    plotGenAx.plot(x, y, label=f'{legend} Axis', color=next(plotColor), linewidth=2, marker='o')

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


def plot3D(x, y, z, legend, label, plotGenFig=None, plotGenAx=None):
    '''
    # x: Array of values for x axis.
    # y: Array of values for y axis.
    # z: Array of values of z axis.
    # legend: Name for different curves apearing in legend.
    # plotGenFig: This is the figure. It could be used to plot the curves on the same figure.
    # plotGenAx: This is the axes. It could be used to plot the curves on the same axes.
    # label = array with labels for title, x axis, yaxis and zaxis in order.
    '''

    if plotGenFig is None or plotGenAx is None:
        plotGenFig = plt.figure()
        plotGenAx = plotGenFig.add_subplot(111, projection='3d') # this allows us to use the same axis on the figure to plot multiple times
    
    plotGenAx.plot(z, x, y, label=legend, color=next(plotColor), linewidth=2, marker='.')

    # Add titles and labels
    plotGenAx.set_title(label[0])
    plotGenAx.set_xlabel(label[1])
    plotGenAx.set_ylabel(label[2])
    plotGenAx.set_zlabel(label[3])
    plotGenAx.minorticks_on()

    # Show major and minor grid
    plotGenAx.grid(True)
    plotGenAx.grid(True, which='minor')

    # Show legend
    plotGenAx.legend()
    
    return plotGenFig, plotGenAx