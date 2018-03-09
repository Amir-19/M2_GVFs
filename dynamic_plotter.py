'''
A simple plotting library that plots lines in real time
Author: Jaden Travnik
Email: jaden.travnik@gmail.com
'''


import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
#print(mpl.matplotlib_fname())
plt.switch_backend('TkAgg')
'''
    Stores and updates the data for one matplotlib line.
    If window_x is not None, the plotted data is restricted to that many data points
'''
class DynamicLine():

    def __init__(self, window_x, line):

        self.window_x = window_x
        self.xdata = []
        self.ydata = []
        self.line = line

    def add_point(self, _x, _y):

        if self.window_x is not None and len(self.xdata) >= self.window_x:

            self.xdata.pop(0)
            self.ydata.pop(0)

        self.xdata.append(_x)
        self.ydata.append(_y)

        #Update data (with the new _and_ the old points)
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)

'''
    A collection of DynamicLines, used to pass on data and redraw in its update function
'''
class DynamicPlot():

    def __init__(self, title = None, xlabel = None, ylabel = None, window_x = None):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.lines = []
        self.ax.set_autoscaley_on(True)
        self.ax.grid()
        self.window_x = window_x

        if title:
            self.ax.set_title(title)

        if xlabel:
            self.ax.set_xlabel(xlabel)

        if ylabel:
            self.ax.set_ylabel(ylabel)

    def add_line(self, label = 'lineName'):
        line, = self.ax.plot([],[], label = label)
        self.lines.append(DynamicLine(self.window_x, line))
        self.ax.legend(loc='upper center')

    ''' update
     Accepts one y data point (eg. timestep) and an data array
     which is ordered based on how the lines were added to the DynamicPlot
    '''
    def update(self, y, data):

        for i in range(len(data)):
            self.lines[i].add_point(y, data[i])

        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


# Example Usage
# d = DynamicPlot(window_x = 30, title = 'Trigonometry', xlabel = 'X', ylabel= 'Y')
# d.add_line('sin(x)')
# d.add_line('cos(x)')
# d.add_line('cos(.5*x)')
#
# for i in np.arange(0,40, .2):
#     d.update(i, [np.sin(i), np.cos(i), np.cos(i/.5)])
#     time.sleep(.01)