import numpy as np
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt



def plot_graph(a, b, corr_measure,rmse):

    a = np.asarray(a)
    b = np.asarray(b)

    vec = 10*np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,  0.9,1.0])
    fig,ax = plt.subplots()
    plt.plot(a,b,'o',color='blue',markersize=15)

    plt.plot([0,10],[0,10],zorder=1,color='green', linestyle='dashed',\
            label='y=x',linewidth=1.5)

    plt.xticks(np.arange(0.0,10.5,1),fontsize=28)
    plt.yticks(np.arange(0.0,10.5,1),fontsize=28)

    plt.gca().set_aspect(0.75)

    mm, bb = np.polyfit(a,b,1)
    plt.plot(vec, mm*vec + bb, linestyle='dashdot', color='blue', \
            label='regression line',linewidth=1.5)
    plt.xlabel('Severity - Prediction',fontsize=36)
    plt.ylabel('Severity - Reference', fontsize=36)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.plot([], [], ' ', label="p = {:.3f}".format(corr_measure))
    plt.plot([], [], ' ', label="RMSE = {:.3f}".format(rmse))
    plt.legend(fontsize=26)
    plt.show()

