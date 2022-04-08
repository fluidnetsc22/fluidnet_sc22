import argparse
import glob
import time

import matplotlib
import torch
import torch.autograd
import yaml

if 'DISPLAY' not in glob.os.environ:
    matplotlib.use('Agg')
import importlib.util
import math
from math import inf
from shutil import copyfile
from timeit import default_timer

import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pyevtk.hl as vtk
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import lib
import lib.fluid as fluid
from lib import MultiScaleNet, fluid

# **************************** Load command line arguments *********************

parser = argparse.ArgumentParser(description='Buoyant plume simulation. \n'
                                 'Read plumeConfig.yaml for more information',
                                 formatter_class=lib.SmartFormatter)
parser.add_argument('--simConf',
                    default='plumeConfig.yaml',
                    help='R|Simulation yaml config file.\n'
                    'Overwrites parameters from trainingConf file.\n'
                    'Default: plumeConfig.yaml')
parser.add_argument('--modelDir',
                    help='R|Neural network model location.\n'
                    'Default: written in simConf file.')
parser.add_argument('--modelFilename',
                    help='R|Model name.\n'
                    'Default: written in simConf file.')
parser.add_argument('--outputFolder',
                    help='R|Folder for sim output.\n'
                    'Default: written in simConf file.')
parser.add_argument('--restartSim', action='store_true', default=False,
                    help='R|Restarts simulation from checkpoint.\n'
                    'Default: written in simConf file.')
parser.add_argument('-sT', '--setThreshold',
                    help='R|Sets the Divergency Threshold.\n'
                    'Default: written in simConf file.', type=float)
parser.add_argument('-delT', '--setdt',
                    help='R|Sets the dt.\n'
                    'Default: written in simConf file.', type=float)

# Cylinder Test
parser.add_argument('--Cylinder',
                    default=True,
                    help='R|Includes a cylinder in the domain.\n'
                    'Default: True')


arguments = parser.parse_args()

# Loading a YAML object returns a dict
with open(arguments.simConf, 'r') as f:
    simConf = yaml.load(f, Loader=yaml.FullLoader)

if not arguments.restartSim:
    restart_sim = simConf['restartSim']
else:
    restart_sim = arguments.restartSim

if not arguments.Cylinder:
    Cylinder = True
else:
    Cylinder = arguments.Cylinder

# Loading already existing data?
resume_test = simConf['resumetest']

# Check that folder to continue simulation exists!
if not resume_test:
    folder = arguments.outputFolder or simConf['outputFolder']
    if (not glob.os.path.exists(folder)):
        glob.os.makedirs(folder)
else:
    folder = simConf['outputFolder']
    assert glob.os.path.exists(folder), 'The foldr {} of the simulation to be continued does not exist!'.format(folder)

restart_config_file = glob.os.path.join('/', folder, 'plumeConfig.yaml')
restart_state_file = glob.os.path.join('/', folder, 'restart.pth')
if restart_sim:
    # Check if configPlume.yaml exists in folder
    assert glob.os.path.isfile(restart_config_file), 'YAML config file does not exists for restarting.'
    with open(restart_config_file) as f:
        simConfig = yaml.load(f, Loader=yaml.FullLoader)

simConf['modelDir'] = arguments.modelDir or simConf['modelDir']
assert (glob.os.path.exists(simConf['modelDir'])), 'Directory ' + str(simConf['modelDir']) + ' does not exists'
simConf['modelFilename'] = arguments.modelFilename or simConf['modelFilename']
simConf['modelDirname'] = simConf['modelDir'] + '/' + simConf['modelFilename']
resume = False  # For training, at inference set always to false


print('Active CUDA Device: GPU', torch.cuda.current_device())
print()
path = simConf['modelDir']
path_list = path.split(glob.os.sep)
saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_simulate.py')
copyfile(saved_model_name, temp_model)

assert glob.os.path.isfile(temp_model), temp_model + ' does not exits!'
# importlib.util.spec_from_file_location(name, location, *, loader=None, submodule_search_locations=None)
# A factory function for creating a ModuleSpec instance based on the path
# to a file. Missing information will be filled in on the spec by making
# use of loader APIs and by the implication that the module will be
# file-based.
spec = importlib.util.spec_from_file_location('model_saved', temp_model)
# importlib.util.module_from_spec(spec)
# Create a new module based on spec and spec.loader.create_module.
# If spec.loader.create_module does not return None, then any pre-existing attributes will not be reset.
# Also, no AttributeError will be raised if triggered while accessing spec or setting an attribute on the module.
# This function is preferred over using types.ModuleType to create a new
# module as spec is used to set as many import-controlled attributes on
# the module as possible.
model_saved = importlib.util.module_from_spec(spec)
# exec_module(module)
# An abstract method that executes the module in its own namespace when a
# module is imported or reloaded. The module should already be initialized
# when exec_module() is called. When this method exists, create_module()
# must be defined.
spec.loader.exec_module(model_saved)

try:
    mconf = {}

    mcpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_mconf.pth')
    assert glob.os.path.isfile(mcpath), mcpath + ' does not exits!'
    mconf.update(torch.load(mcpath))

    print('==> overwriting mconf with user-defined simulation parameters')
    # Overwrite mconf values with user-defined simulation values.
    mconf.update(simConf)

    print('==> loading model')
    mpath = glob.os.path.join(simConf['modelDir'], simConf['modelFilename'] + '_lastEpoch_best.pth')
    assert glob.os.path.isfile(mpath), mpath + ' does not exits!'
    state = torch.load(mpath)

    print('Data loading: done')

    # ********************************** Create the model ***************************
    with torch.no_grad():

        it = 0
        cuda = torch.device('cuda')

        mconf['input_res'] = [mconf['resZ'], mconf['resY'], mconf['resX']]
        net = model_saved.FluidNet(mconf, it, folder, dropout=False)
        if torch.cuda.is_available():
            net = nn.DataParallel(net)
            net = net.cuda()

        net.load_state_dict(state['state_dict'])

        # *********************** Simulation parameters **************************

        resX = simConf['resX']
        resY = simConf['resY']
        resZ = simConf['resZ']

        # Cylinder will always be at the middle of the x and and at 1/4 of the y axis
        # centerZ will be used for the probes
        # We declare the center and radius of the cylinder
        centerX = int(resX / 2)
        centerY = int(resY / 4)
        centerZ = int(resZ / 2)
        radCyl = int(resX / 16)

        if resume_test:
            U = torch.from_numpy(np.load(folder + '/U_last_saved.npy'))
            density = torch.from_numpy(np.load(folder + '/density_last_saved.npy'))
            it = np.int(np.load(folder + '/it_last_saved.npy'))

            print('Continuing Fluid inference at timestep {}'.format(it))

        else:
            U = torch.zeros((1, 3, resZ, resY, resX), dtype=torch.float).cuda()
            density = torch.zeros((1, 1, resZ, resY, resX), dtype=torch.float).cuda()

            print('New Fluid Inference from scratch')

        p = torch.zeros((1, 1, resZ, resY, resX), dtype=torch.float).cuda()
        flags = torch.zeros((1, 1, resZ, resY, resX), dtype=torch.float).cuda()
        Ustar = torch.zeros((1, 3, resZ, resY, resX), dtype=torch.float).cuda()
        div_input = torch.zeros((1, 1, resZ, resY, resX), dtype=torch.float).cuda()

        fluid.emptyDomain(flags)

        # Add cylinder to the domain
        print("Cylinder", Cylinder)
        if Cylinder:

            """
            Creates a cilinder in the flags. It will be located in the point x = 64 and y = 80.
            Radius = 10
            """

            X = torch.arange(0, resX, device=cuda).view(resX).expand((1, resY, resX))
            Y = torch.arange(0, resY, device=cuda).view(resY, 1).expand((1, resY, resX))

            dist_from_center = (X - centerX).pow(2) + (Y - centerY).pow(2)
            mask_cylinder = dist_from_center <= radCyl * radCyl

            flags = flags.masked_fill_(mask_cylinder, 2)

        batch_dict = {}
        batch_dict['p'] = p
        batch_dict['U'] = U
        batch_dict['Ustar'] = Ustar
        batch_dict['flags'] = flags
        batch_dict['density'] = density
        batch_dict['Div_in'] = div_input

        batch_dict['Test_case'] = 'VK'
        # We create a temporary flags for the inflow, in order to avoid affecting the advection
        #flags_i = flags.clone()
        #batch_dict['flags_inflow'] = flags_i

        save_vtk = simConf['saveVTK']
        method = simConf['simMethod']
        #it = 0

        dt = arguments.setdt or simConf['dt']
        Outside_Ja = simConf['outside_Ja']
        Threshold_Div = arguments.setThreshold or simConf['threshold_Div']

        max_iter = simConf['maxIter']
        outIter = simConf['statIter']

        rho1 = simConf['injectionDensity']
        rad = simConf['sourceRadius']
        plume_scale = simConf['injectionVelocity']

        # Only for the VK
        batch_dict['VK'] = plume_scale

        # **************************** Initial conditions ***************************

        if not resume_test:

            fluid.createVKBCs(batch_dict, rho1, plume_scale, rad)

            # Initial field = cte.!
            U = batch_dict['U']
            U[:, 2, :] += 0
            U[:, 1, :] += plume_scale
            U[:, 0, :] += 0
            batch_dict['U'] = U

            if simConf['simMethod'] == 'CG':
                # A matrix creation
                print("--------------------------------------------------------------")
                print("------------------- A matrix creation ------------------------")
                print("--------------------------------------------------------------")

                A_val, I_A, J_A = fluid.CreateCSR_Direct_3D(flags)

                # Save npy files for restarting simulations
                filenameA = folder + '/A_val'
                np.save(filenameA, A_val)
                filenameI = folder + '/I_A'
                np.save(filenameI, I_A)
                filenameJ = folder + '/J_A'
                np.save(filenameJ, J_A)

        else:

            A_val = np.load(folder + '/A_val.npy')
            I_A = np.load(folder + '/I_A.npy')
            J_A = np.load(folder + '/J_A.npy')

        if simConf['simMethod'] == 'CG':
            batch_dict['Val'] = A_val
            batch_dict['IA'] = I_A
            batch_dict['JA'] = J_A

        # If restarting, overwrite all fields with checkpoint.
        if restart_sim:
            # Check if restart file exists in folder
            assert glob.os.path.isfile(restart_state_file), 'Restart file does not exists.'
            restart_dict = torch.load(restart_state_file)
            batch_dict = restart_dict['batch_dict']
            it = restart_dict['it']
            print('Restarting from checkpoint at it = ' + str(it))

        # Create YAML file in output folder
        with open(restart_config_file, 'w') as outfile:
            yaml.dump(simConf, outfile)

        # Plotting Variables
        minY = 0
        maxY = resY
        maxY_win = resY
        minX = 0
        maxX = resX
        maxX_win = resX
        minZ = 0
        maxZ = resZ
        maxZ_win = resZ
        X, Y, Z = np.linspace(0, resX - 1, num=resX),\
            np.linspace(0, resY - 1, num=resY),\
            np.linspace(0, resZ - 1, num=resZ)

        # Time Vec Declaration
        Time_vec = np.zeros(max_iter)
        Time_Pres = np.zeros(max_iter)
        Jacobi_switch = np.zeros(max_iter)
        Max_Div = np.zeros(max_iter)
        Mean_Div = np.zeros(max_iter)
        Max_Div_All = np.zeros(max_iter)
        time_big = np.zeros(max_iter)

        # Probes and Plotting
        Probes_U = np.zeros((3, max_iter))
        range_plt = np.arange(max_iter)

        data = torch.cat((div_input, flags), 1)
        density = batch_dict['density']
        t_plot = np.arange(max_iter) * dt
        batch_dict['Div_analysis'] = True

        # Main loop
        while (it < max_iter):

            if it > 0:
                tensor_vel_prev = fluid.getCentered(batch_dict['U'].clone())
                img_vel_prev = torch.squeeze(tensor_vel_prev).cpu().data.numpy()

            U_modif = batch_dict['U']
            if it < 1:
                U_modif[:, 0, :, centerY + 3 * radCyl:centerY + 6 * radCyl,
                        centerX - 3 * radCyl:centerX + 3 * radCyl] += plume_scale / 2
                U_modif[:, 0, :, centerY + 6 * radCyl:centerY + 9 * radCyl,
                        centerX - 3 * radCyl:centerX + 3 * radCyl] -= plume_scale / 2

            batch_dict['U'] = U_modif

            method = mconf['simMethod']

            start_big = default_timer()
            lib.simulate(
                mconf,
                batch_dict,
                net,
                method,
                Time_vec,
                Time_Pres,
                Jacobi_switch,
                Max_Div,
                Mean_Div,
                Max_Div_All,
                folder,
                it,
                Threshold_Div,
                dt,
                Outside_Ja)
            end_big = default_timer()

            # Store the elapsed time for the whole simulate in one it:
            time_big[it] = (end_big - start_big)
            #print("Simulate module elapsed time: ===================>", (end_big - start_big))

            # Hard coded, Probe value for the Strouhal number, always situated at 1.5 diameters from the cylinder
            Probes_U[0, it] = batch_dict['U'][0, 0, centerZ, centerY + 3 * radCyl, centerX].cpu().data.numpy()
            Probes_U[1, it] = batch_dict['U'][0, 1, centerZ, centerY + 3 * radCyl, centerX].cpu().data.numpy()
            Probes_U[2, it] = batch_dict['U'][0, 2, centerZ, centerY + 3 * radCyl, centerX].cpu().data.numpy()

            # Update iterations
            it += 1
            Ustar = batch_dict['Ustar']

            # Plotting and Saving for loading purposes, every 10 its
            if it % outIter == 0:

                # torch tensors
                tensor_div = fluid.velocityDivergence(batch_dict['U'].clone(),
                                                      batch_dict['flags'].clone())
                pressure = batch_dict['p'].clone()
                density = batch_dict['density'].clone()
                tensor_vel = fluid.getCentered(batch_dict['U'].clone())
                # Convert to numpy
                img_vel = torch.squeeze(tensor_vel).cpu().data.numpy()
                rho = torch.squeeze(density).cpu().data.numpy()
                p = torch.squeeze(pressure).cpu().data.numpy()
                div = torch.squeeze(tensor_div).cpu().data.numpy()

                # Save time and probes
                filename_probe = folder + '/Probes_U'
                np.save(filename_probe, Probes_U)
                filename_big = folder + '/Time_big'
                np.save(filename_big, time_big)
                filename_jac = folder + '/Jacobi_switch'
                np.save(filename_jac, Jacobi_switch)
                filename_max = folder + '/Max_Div'
                np.save(filename_max, Max_Div)
                filename_maxA = folder + '/Max_Div_All'
                np.save(filename_maxA, Max_Div_All)
                filename_time = folder + '/Time_vec'
                np.save(filename_time, Time_vec)
                filename_tp = folder + '/Time_Pres'
                np.save(filename_tp, Time_Pres)

                restart_dict = {'batch_dict': batch_dict, 'it': it}
                torch.save(restart_dict, restart_state_file)

            it_saving = simConf['it_minsave']
            # saving npy fields.
            if it > it_saving and it % outIter == 0:

                filename_p = folder + '/P_output_{0:05}'.format(it)
                np.save(filename_p, p)

                filename_U_s = folder + '/U_output_{0:05}'.format(it)
                np.save(filename_U_s, img_vel)

                filename_U_p = folder + '/U_output_{0:05}'.format(it - 1)
                np.save(filename_U_p, img_vel_prev)

                filename_div = folder + '/Div_output_{0:05}'.format(it)
                np.save(filename_div, div)

            # Plotting and Saving
            if save_vtk and it % outIter == 0 and it > it_saving:
                px, py, pz = 580, 300, 100
                dpi = 25
                figx = px / dpi
                figy = py / dpi
                figz = pz / dpi

                nx = maxX_win
                ny = maxY_win
                nz = maxZ_win
                ncells = nx * ny * nz

                ratio_x = nx / nz
                ratio_y = ny / nz
                lx, ly, lz = ratio_x, ratio_y, 1.0
                dx, dy, dz = lx / nx, ly / ny, lz / nz

                # Coordinates
                x = np.arange(0, lx + 0.1 * dx, dx, dtype='float32')
                y = np.arange(0, ly + 0.1 * dy, dy, dtype='float32')
                z = np.arange(0, lz + 0.1 * dz, dz, dtype='float32')

                # Variables
                div_input = batch_dict['Div_in'][0, 0].clone()
                div = fluid.velocityDivergence(
                    batch_dict['U'].clone(),
                    batch_dict['flags'].clone())[0, 0]
                velstar_div = fluid.velocityDivergence(
                    batch_dict['Ustar'].clone(),
                    batch_dict['flags'].clone())[0, 0]
                vel = fluid.getCentered(batch_dict['U'].clone())
                velstar = fluid.getCentered(batch_dict['Ustar'].clone())
                density = batch_dict['density'].clone()
                pressure = batch_dict['p'].clone()
                b = 1
                w = pressure.size(4)
                h = pressure.size(3)
                d = pressure.size(2)

                rho = density.narrow(4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2)
                rho = rho.clone().expand(b, 3, d - 2, h - 2, w - 2)
                rho_m = rho.clone().expand(b, 3, d - 2, h - 2, w - 2)
                rho_m[:, 0] = density.narrow(4, 0, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2).squeeze(1)
                rho_m[:, 1] = density.narrow(4, 1, w - 2).narrow(3, 0, h - 2).narrow(2, 1, d - 2).squeeze(1)
                rho_m[:, 2] = density.narrow(4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 0, d - 2).squeeze(1)
                gradRho_center = torch.zeros_like(vel)[:, 0:3].contiguous()
                gradRho_faces = rho - rho_m
                gradRho_center[:, 0:3, 1:(d - 1), 1:(h - 1), 1:(w - 1)] = fluid.getCentered(gradRho_faces)[0, 0:3]

                Pijk = pressure.narrow(4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2)
                Pijk = Pijk.clone().expand(b, 3, d - 2, h - 2, w - 2)
                Pijk_m = Pijk.clone().expand(b, 3, d - 2, h - 2, w - 2)
                Pijk_m[:, 0] = pressure.narrow(4, 0, w - 2).narrow(3, 1, h - 2).narrow(2, 1, d - 2).squeeze(1)
                Pijk_m[:, 1] = pressure.narrow(4, 1, w - 2).narrow(3, 0, h - 2).narrow(2, 1, d - 2).squeeze(1)
                Pijk_m[:, 2] = pressure.narrow(4, 1, w - 2).narrow(3, 1, h - 2).narrow(2, 0, d - 2).squeeze(1)
                gradP_center = torch.zeros_like(vel)[:, 0:3].contiguous()
                gradP_faces = Pijk - Pijk_m
                gradP_center[:, 0:3, 1:(d - 1), 1:(h - 1), 1:(w - 1)] = fluid.getCentered(gradP_faces)[:, 0:3]

                pressure = pressure[0, 0]
                density = density[0, 0]

                velX = vel[0, 0].clone()
                velY = vel[0, 1].clone()
                velZ = vel[0, 2].clone()
                velstarX = velstar[0, 0].clone()
                velstarY = velstar[0, 1].clone()
                velstarZ = velstar[0, 2].clone()
                gradRhoX = gradRho_center[0, 0].clone()
                gradRhoY = gradRho_center[0, 1].clone()
                gradRhoZ = gradRho_center[0, 2].clone()
                gradPX = gradP_center[0, 0].clone()
                gradPY = gradP_center[0, 1].clone()
                gradPZ = gradP_center[0, 2].clone()
                flags = batch_dict['flags'][0, 0].clone()

                # Change shape form (D,H,W) to (W,H,D)
                div.transpose_(0, 2).contiguous()
                div_input.transpose_(0, 2).contiguous()
                velstar_div.transpose_(0, 2).contiguous()
                density.transpose_(0, 2).contiguous()
                pressure.transpose_(0, 2).contiguous()
                velX.transpose_(0, 2).contiguous()
                velY.transpose_(0, 2).contiguous()
                velZ.transpose_(0, 2).contiguous()
                velstarX.transpose_(0, 2).contiguous()
                velstarY.transpose_(0, 2).contiguous()
                velstarZ.transpose_(0, 2).contiguous()
                gradRhoX.transpose_(0, 2).contiguous()
                gradRhoY.transpose_(0, 2).contiguous()
                gradRhoZ.transpose_(0, 2).contiguous()
                gradPX.transpose_(0, 2).contiguous()
                gradPY.transpose_(0, 2).contiguous()
                gradPZ.transpose_(0, 2).contiguous()
                flags.transpose_(0, 2).contiguous()

                div_np = div.cpu().data.numpy()
                div_input_np = div_input.cpu().data.numpy()
                velstardiv_np = velstar_div.cpu().numpy()
                density_np = density.cpu().data.numpy()
                pressure_np = pressure.cpu().data.numpy()
                velX_np = velX.cpu().data.numpy()
                velY_np = velY.cpu().data.numpy()
                velZ_np = velZ.cpu().data.numpy()
                velstarX_np = velstarX.cpu().data.numpy()
                velstarY_np = velstarY.cpu().data.numpy()
                velstarZ_np = velstarY.cpu().data.numpy()
                np_mask = flags.eq(2).cpu().data.numpy().astype(float)
                pressure_masked = ma.array(pressure_np, mask=np_mask)
                velx_masked = ma.array(velX_np, mask=np_mask)
                vely_masked = ma.array(velY_np, mask=np_mask)
                velz_masked = ma.array(velZ_np, mask=np_mask)
                velstarx_masked = ma.array(velstarX_np, mask=np_mask)
                velstary_masked = ma.array(velstarY_np, mask=np_mask)
                velstarz_masked = ma.array(velstarZ_np, mask=np_mask)
                ma.set_fill_value(pressure_masked, np.nan)
                ma.set_fill_value(velx_masked, np.nan)
                ma.set_fill_value(vely_masked, np.nan)
                ma.set_fill_value(velz_masked, np.nan)
                pressure_masked = pressure_masked.filled()
                velx_masked = velx_masked.filled()
                vely_masked = vely_masked.filled()
                velz_masked = velz_masked.filled()
                velstarx_masked = velstarx_masked.filled()
                velstary_masked = velstary_masked.filled()
                velstarz_masked = velstarz_masked.filled()

                divergence = np.ascontiguousarray(div_np[minX:maxX, minY:maxY, minZ:maxZ])
                divergence_input = np.ascontiguousarray(div_input_np[minX:maxX, minY:maxY, minZ:maxZ])
                rho = np.ascontiguousarray(density_np[minX:maxX, minY:maxY, minZ:maxZ])
                p = np.ascontiguousarray(pressure_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velx = np.ascontiguousarray(velx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                vely = np.ascontiguousarray(vely_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velz = np.ascontiguousarray(velz_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstarx = np.ascontiguousarray(velstarx_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstary = np.ascontiguousarray(velstary_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstarz = np.ascontiguousarray(velstarz_masked[minX:maxX, minY:maxY, minZ:maxZ])
                velstardiv = np.ascontiguousarray(velstardiv_np[minX:maxX, minY:maxY, minZ:maxZ])
                gradRhox = np.ascontiguousarray(gradRhoX.cpu().data.numpy()[minX:maxX, minY:maxY, minZ:maxZ])
                gradRhoy = np.ascontiguousarray(gradRhoY.cpu().data.numpy()[minX:maxX, minY:maxY, minZ:maxZ])
                gradRhoz = np.ascontiguousarray(gradRhoZ.cpu().data.numpy()[minX:maxX, minY:maxY, minZ:maxZ])
                gradPx = np.ascontiguousarray(gradPX.cpu().data.numpy()[minX:maxX, minY:maxY, minZ:maxZ])
                gradPy = np.ascontiguousarray(gradPY.cpu().data.numpy()[minX:maxX, minY:maxY, minZ:maxZ])
                gradPz = np.ascontiguousarray(gradPZ.cpu().data.numpy()[minX:maxX, minY:maxY, minZ:maxZ])
                filename = folder + '/output_{0:05}'.format(it)
                vtk.gridToVTK(filename, x, y, z, cellData={
                    'density': rho,
                    'divergence': divergence,
                    'pressure': p,
                    'ux': velx,
                    'uy': vely,
                    'uz': velz,
                    'u_star_x': velstarx,
                    'u_star_y': velstary,
                    'u_star_z': velstarz,
                    'u_star_div': velstardiv,
                    'divergence_input': divergence_input,
                    'gradPx': gradPx,
                    'gradPy': gradPy,
                    'gradPz': gradPz,
                    'gradRhox': gradRhox,
                    'gradRhoy': gradRhoy,
                    'gradRhoz': gradRhoz,
                })

finally:
    # Properly deleting model_saved.py, even when ctrl+C
    print()
    print('Deleting' + temp_model)
    glob.os.remove(temp_model)
