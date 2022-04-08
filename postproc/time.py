########################################################################################################################
#                                                                                                                      #
#                                       Plot function to plot Performance Time                                         #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

'''
Plots the time taken for a network to perform inference in different domain sizes


OUTPUTS:
Both in png and pdf:

'Scale_3D.pdf'

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import csv
import pandas as pd

# For now manually set min and max (to be improved)
min_val = 5
max_val =  400

# Network times
mdb_3 = np.array((7.3, 10.9, 7.1, 19.7, 106.8, 207.4, 345.5))
mdb_4 = np.array((9.3, 11.9, 8.1, 20.7, 109.8, 206.4, 340.5))
mdb_5 = np.array((11.3, 13.9, 10.1, 21.7, 107.8, 205.4, 343.5))


# Fourth Image
plt.figure()
sq_size = np.array((16, 32, 48, 64, 128, 160, 192))
sq_r = np.array((16, 32, 64, 128, 192))
cell = sq_size * sq_size * sq_size
cell_reduced = sq_r * sq_r * sq_r

# Plots
plt.plot(cell, mdb_3, color='blue', marker= 'o', \
                linestyle='dashed', linewidth=2, markersize=7)
plt.plot(cell, mdb_4, color='green', marker= 's', \
                linestyle='dashed', linewidth=2, markersize=7)
plt.plot(cell, mdb_5, color='orange', marker= '^', \
                linestyle='dashed', linewidth=2, markersize=7)

#plt.legend( manual_lines, ['MultiScale', 'Scale N', 'Scale N/2', 'Scale N/4'], ncol=1)

savefile_png = 'Scale_3D.png'
savefile_pdf = 'Scale_3D.pdf'


plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'Resolution')
plt.ylabel(r'$t_{inf}$  (ms)')
plt.xticks(cell_reduced, ('$16^3$', '$32^3$', '$64^3$', '$128^3$', '$192^3$'))

plt.ylim(min_val, max_val)

plt.savefig(savefile_png)
plt.savefig(savefile_pdf)