'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''

import sys
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.datasets import *
from scipy.stats import multivariate_normal

from itertools import combinations
from scipy.spatial import Voronoi, voronoi_plot_2d


'''
X = Data
Y = Cluster Labels (1,2,...), 0 for noise
'''
def plot_data(X,Y=None,canvas=None,xtitle=None,ytitle=None,plt_title=None,colmap=None,show_legend=False):
            
    if(Y is None):
        Y = np.ones(len(X),dtype=int)
    num_colors = max(Y)+1
    if(colmap is None):
        if(num_colors<10):
            map_name = 'tab10'
        elif(num_colors<20):
            map_name = 'tab20'
        else:
            map_name = 'hsv'
        colmap = plt.cm.get_cmap(map_name, num_colors).copy()   
        colmap.set_under('w')

        
    if(canvas is None):
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        ax = canvas
        ax.cla()
    
    if(plt_title is not None):
        ax.set_title(plt_title)  

    scatter = ax.scatter(X[:,0],X[:,1],c=Y,cmap=colmap,edgecolors='black',alpha=0.7,vmin=1,vmax=num_colors)
    if(show_legend):
        L = ax.legend(*scatter.legend_elements(),loc="lower right", title="Clusters")
        
               
    if(xtitle is not None):
        ax.set_xlabel(xtitle,fontweight='bold',fontsize=16)
    
    if(xtitle is not None):
        ax.set_ylabel(ytitle,fontweight='bold',fontsize=16)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

   