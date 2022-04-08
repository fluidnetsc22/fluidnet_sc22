import numpy as np
import torch
import torch.utils.data
import os
import plotly.graph_objects as go

import argparse
import random
from tqdm import tqdm
import pdb

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def plot_bars(array_p, labels, name_save, y_lim=None):

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, array_p[:, 0], width, color='blue', label = '3 Scales')
    rects2 = ax.bar(x , array_p[:, 1], width, color='red', label = '4 Scales')
    rects3 = ax.bar(x + width, array_p[:, 2], width, color='green', label= '5 Scales')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'$\mathcal{E}$')
    ax.set_xticks(x)
    ax.set_ylim(0, 0.025)
    ax.set_xticklabels(labels)
    #ax.legend()
    if y_lim != None:
        ax.set_ylim(y_lim)

    fig.tight_layout()

    fig.savefig(folder_im + name_save)


# Plot 3D fields

def plot_fields(density, folder_im, it, Ri, network):
    density = np.transpose(density, (0, 2, 1))
    # Define axes
    X, Y, Z = np.mgrid[0:128, 0:128, 0:128]

    # Generate Figure
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=density.flatten(),
        colorscale='Blues',
        isomin=0.0020,
        isomax=0.0030,
        opacity=0.9,
        surface_count=3,
        showscale=False,
        caps=dict(x_show=False, y_show=False),
        ))

    # Modify axes and camera view
    fig.update_layout(scene_xaxis_showticklabels=False,
                    scene_yaxis_showticklabels=False,
                    scene_zaxis_showticklabels=False,
                    scene = dict(
                        xaxis_title="",
                        yaxis_title="",
                        zaxis_title=""),
                    margin=dict(t=0, l=0, r=0.2, b=0),
                    scene_camera_eye=dict(x=1.0, y=1.0, z=0.25))

    # Create folder for saving if does not exist
    fig_folder = os.path.join(folder_im, Ri, network)
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # Save
    fig.write_image(os.path.join(fig_folder, f"Field_{it}.png"))


# Initialize parse
# 3 arguments: Richardson number, cg ref to plot and margin Percent
parser = argparse.ArgumentParser(description='Plot Specified cases (Ri and Geom).')
parser.add_argument('-cg', '--cgmetric',  default='True',
        help='R|CG based metric.', type = str)
parser.add_argument('-P', '--Percen',  default=75,
                help='R|Percentage to identify plume head.', type = int)
parser.add_argument('-Ri', '--richardson',  default='1_0',
        help='R|Richardson number in a string (choose between 0_1, 1_0 and 10_0', type = str)
args = parser.parse_args()

# Addtional parameter to check the Ri number, plot it and margin Percent
margin_perc = args.Percen
cg_analysis = args.cgmetric == 'True' or args.cgmetric == 'true'
Ri = args.richardson

if cg_analysis:
    Networks = ['Jacobi', 'UNet_4_RF_128_p_100', 'UNet_5_RF_128_p_100']

colors = ['black', 'orange', 'blue', 'green', 'blue', 'blue', 'blue', 'yellow', 'yellow', 'yellow', 'yellow']
markers = [None, None, None, None, '^', '*', '*', 'v', 'o', 's', 'p', '*', 'v', 'o', 's',]
dash_line = ['solid', 'dashdot', 'dashed', 'dotted', 'solid', 'dashdot', 'dashed', 'dotted', 'solid', 'dashdot', 'dashed', 'dotted']

if Ri == '0_0':
    every = 20
    Ti = 99
elif Ri == '0_1':
    every = 10
    Ti = 99
elif Ri == '1_0':
    every = 10
    Ti = 61
elif Ri == '10_0':
    every = 5
    Ti = 59

res = 128

# Base folders and Networks
folder =  '/path/to/results/results_3d/Plume_3D/'
folder_im = folder + 'Images/new_head/'

if not os.path.exists(folder_im):
    os.makedirs(folder_im)

# Array to compute integral
pos_array = np.zeros((len(Networks), Ti))
integral_values = np.zeros((len(Networks)-1))
ninety_points_x = np.zeros((len(Networks)-1))
ninety_points_y = np.zeros((len(Networks)-1))
ninety_points_z = np.zeros((len(Networks)-1))
ref_integral = np.zeros((Ti))

# Max density to follow the plume
max_density = 0.01
margin = max_density - ((margin_perc/100)*max_density)
x_range = np.arange(Ti)


for i, network in enumerate(Networks):

    # Basee folder and its subfolders
    folder_load = folder + 'Ri_{}/{}'.format(Ri, network)

    density = np.zeros(Ti)
    Pixels_sum_x = np.zeros(Ti)
    Pixels_sum_y = np.zeros(Ti)
    Pixels_sum_z = np.zeros(Ti)
    x_range = np.arange(Ti)
    ref_found = False

    # Loop through files
    for itt in tqdm(range(Ti)):
        # Load file
        filename = '/Rho_NN_output_{0:05}.npy'.format((itt+1)*every)

        folderfile = folder_load + filename
        rho_loaded = np.load(folderfile)

        # Plot every 5
        if itt % 5 ==0:
            plot_fields(rho_loaded, folder_im, itt, Ri, network)

        density_mask = np.where(rho_loaded > margin, np.ones_like(rho_loaded), np.zeros_like(rho_loaded))


        x = y = z = np.arange(res)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

        density_x = density_mask * xv
        density_y = density_mask * yv
        density_z = density_mask * zv

        max_x = np.max(density_y)
        max_y = np.max(density_x)
        max_z = np.max(density_z)

        Pixels_sum_z[itt] = max_x
        Pixels_sum_y[itt] = max_y
        Pixels_sum_x[itt] = max_z

        #print(f'Maximum X: {Pixels_sum_x[itt]}, Y: {Pixels_sum_y[itt]}, Z: {Pixels_sum_z[itt]}')

        if Pixels_sum_z[itt]/res > 0.5 and i==0 and not ref_found:
            ref_value_y = Pixels_sum_y[itt]/res
            ref_value_x = Pixels_sum_x[itt]/res
            ref_value_z = Pixels_sum_z[itt]/res
            jacobi_tt = itt
            ref_found = True
            print('Max z, y and max x ', max_z, max_y, max_x)
            print('Here ', jacobi_tt)

    if i ==0:
        ref_integral = Pixels_sum_y/res


    # For integral and Point Distance
    if cg_analysis and i!=0:
        ninety_points_x[i-1] = np.abs(Pixels_sum_x[jacobi_tt]/res - ref_value_x)
        ninety_points_y[i-1] = np.abs(Pixels_sum_y[jacobi_tt]/res - ref_value_y)
        ninety_points_z[i-1] = 100*np.abs(Pixels_sum_z[jacobi_tt]/res - ref_value_z)
        pos_array[i-1] =  Pixels_sum_z/res
        print(f'Ninety x point for network {network} = {Pixels_sum_x[jacobi_tt]/res}, resulting in an error of {ninety_points_x[i-1]} ({Pixels_sum_x[jacobi_tt]} vs {ref_value_x*res})')
        print(f'Ninety y point for network {network} = {Pixels_sum_y[jacobi_tt]/res}, resulting in an error of {ninety_points_y[i-1]} ({Pixels_sum_y[jacobi_tt]} vs {ref_value_y*res})')
        print(f'Ninety z point for network {network} = {Pixels_sum_z[jacobi_tt]/res}, resulting in an error of {ninety_points_z[i-1]} ({Pixels_sum_z[jacobi_tt]} vs {ref_value_z*res})')

    plt.plot(x_range[:-1], Pixels_sum_z[:-1]/res, color=colors[i], marker=markers[i], markevery= 10 + i,linestyle=dash_line[i], linewidth=5, markersize=8, label = network)

print('Ninety Points: ', ninety_points_z)


savefile_png = folder_im + 'Head_Ri_{}.png'.format(Ri)
savefile_pdf = folder_im + 'Head_Ri_{}.pdf'.format(Ri)


#plt.legend(fontsize=12)
#plt.yscale("log")
plt.ylim(0,1.1)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.grid()
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\widetilde{t}$',fontsize=16)
plt.ylabel(r'$\widetilde{h}$', fontsize = 16, rotation = 0)

plt.savefig(savefile_png)
plt.savefig(savefile_pdf)

plt.close('all')
