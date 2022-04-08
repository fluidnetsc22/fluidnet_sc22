import plotly.graph_objects as go
import numpy as np
import argparse

arguments = argparse.ArgumentParser(description='PoissonNetwork runs')
arguments.add_argument('-it','--iteration',
        help='Iteration to plot.', type =int)
arguments.add_argument('-ri','--richardson',
        help='Richardson number for loading 0 (0.1), 1 (1), 2 (10) (int).', type =int)
args = arguments.parse_args()

# Define Richardson
Ri_list = ['0_1', '1', '10']

# Define axes
X, Y, Z = np.mgrid[0:128, 0:128, 0:128]

# Load an rearange data
density = np.load(f'/data/ubuntu/results_3d/Unet/Plume_3D/Ri_{Ri_list[args.richardson]}/Unet_test_4GPU_lt_local/Rho_NN_output_00{args.iteration}.npy')
density = np.transpose(density, (0, 2, 1))

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

# Save
fig.write_image(f"Debug_{Ri_list[args.richardson]}_{args.iteration}.png")