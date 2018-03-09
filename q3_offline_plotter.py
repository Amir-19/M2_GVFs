import numpy
from dynamic_plotter import *
import time


def main():

    d1 = DynamicPlot(window_x=100, title='sensorimotor datastream servo 1', xlabel='time_step', ylabel='value')
    d1.add_line('servo 1 ang * 10')
    d1.add_line('Prediction')
    d1.add_line('Gamma*3')
    d1.add_line('Cumulant*6')



    # you need to choose the file you want to do the offline plot from
    s1_ds = numpy.loadtxt('q3_data.txt')
    for i in range(s1_ds.shape[0]):
        d1.update(i, [s1_ds[i][1],s1_ds[i][2],s1_ds[i][3],s1_ds[i][4]])
        # change to make fast and slow plotting
        time.sleep(0.0)
    while True:
        pass
if __name__ == '__main__':
    main()