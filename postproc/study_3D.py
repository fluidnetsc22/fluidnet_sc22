# General libraries
import numpy as np
import os
import yaml
import pdb

# Plotting Imports
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation

# Scipy computations
import scipy
import scipy.integrate
from scipy.interpolate import griddata
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks


def plot_imshow(field, label, filename):
    # Define
    field_max = np.max(np.abs(field))
    fig = plt.figure()
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    im = plt.imshow(field, label=label, vmin = - field_max, vmax = field_max, cmap= plt.get_cmap('seismic'))
    levels = np.array([-0.1, -0.075, -0.05, -0.025, 0.025, 0.05, 0.075, 0.1])*2.5 * np.max(np.abs(field))
    plt.contour(field, levels, colors='k', origin='lower')
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close()

def plot_line(values, labels, filename):
    # Define
    fig = plt.figure()
    for value, label in zip(values, labels):
        plt.plot(value, label=label)
    plt.legend()
    fig.savefig(filename)
    plt.close()

def plot_cl_vs_cd(cl, cd, filename):
    fig = plt.figure()
    plt.scatter(cl, cd)
    plt.ylabel('Cl')
    plt.xlabel('Cd')
    fig.savefig(filename)
    plt.close()

class Study_3D:
    def __init__(self, config, network):

        # Network name
        self.network = network

        self.base_folder = config['base_folder']
        self.load_folder = config['load_folder']
        self.save_folder = os.path.join(config['save_folder'], self.network)

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Domain sizes
        self.resx = config['resX']
        self.resy = config['resY']
        self.resz = config['resZ']
        self.diam = config['diam']

        self.x_0 = self.resx//2
        self.y_0 = self.resy//4
        self.z_0 = self.resz//2

        self.re = config['Reynolds']
        self.dt = config['dt']
        self.diam = config['diam']

        self.loading_folder = os.path.join(self.load_folder,
                    '{}_{}_{}'.format(self.resy, self.resx, self.resz), 'Re{}'.format(self.re), network)

    	# Iterations
        self.init_it = config['init_it']
        self.n_its = config['n_its']
        self.interval = config['interval']
        self.outiter = config['outiter']

        # Probe and coefficients
        self.probe = np.zeros((3, self.n_its))
        self.cl = np.zeros((self.n_its))
        self.cd = np.zeros((self.n_its))

        # Pixels fro cl and cd
        self.pix = 2

        # Initialize amplitudes for error
        self.amp = 1
        self.ref_amp = 0

    def get_mask(self):
        xv, yv = np.meshgrid(self.x, self.y)
        distance = ((xv - self.x_0)**2 + (yv - self.y_0)**2)**0.5
        cylinder_mask = distance>(self.diam/2)
        values = np.ones_like(xv)*cylinder_mask
        return values


    def compute_vorticity(self, velocity):

        # Initialize vorticity tensor
        vorticity = np.zeros_like(velocity)
        x = np.arange(self.resx)
        y = np.arange(self.resy)
        z = np.arange(self.resz)

        # Vor in x
        dFy_dz = np.gradient(velocity[1], z, axis=0)
        dFz_dy = np.transpose(np.gradient(np.transpose(velocity[2]), y, axis=1))
        vorticity[0] = dFz_dy - dFy_dz

        # Vor in y
        dFz_dx = np.gradient(velocity[2], x, axis=2)
        dFx_dz = np.transpose(np.gradient(np.transpose(velocity[0]), z, axis=2))
        vorticity[1] = dFx_dz - dFz_dx

        # Vor in z
        dFy_dx = np.gradient(velocity[1], x, axis=2)
        dFx_dy = np.transpose(np.gradient(np.transpose(velocity[0]), y, axis=1))
        vorticity[2] =  dFy_dx - dFx_dy

        return vorticity

    def get_st(self):

        # Load Probe and plot for debug
        #U_probe = np.load(os.path.join(self.loading_folder,'Probes_U.npy'))[:, self.init_it:]
        U_probe = self.probe
        plot_line([U_probe[0], U_probe[1], U_probe[2]], ['Ux', 'Uy', 'Uz'], os.path.join(self.save_folder, 'Probe_debug.png'))

        # Initialize loop and create zeros list
        sign = np.sign(U_probe[0, 0])
        zeros = []
        initial = True

        for i in range(self.n_its):
            if (np.sign(U_probe[0, i]) != sign):
                if initial:
                    self.first_zero = i + self.init_it
                    initial = False
                zeros.append(i)
                #print( "Zero in it ", i + self.init_it)
                sign *= -1
                last = i + self.init_it

        # Self the Main Freq Values
        self.n_periods = (len(zeros)-1)/2
        self.avg_period = self.interval *(last-self.first_zero)/self.n_periods
        self.st = self.diam/(self.dt * self.avg_period)

        #print('N of zeros = {}'.format(len(zeros)))
        #print('N periods = {}'.format(self.n_periods))
        #print("Avg period = {}".format(self.avg_period))
        #print('Strouhal : {}'.format(self.st))

    def compute_cp_and_cl(self, pressure, it):
        cd, cl, cp = [], [], []

        x_coord = []
        y_coord = []
        theta = []
        order = []
        theta_rad = []


        for i in range(10000):
            rad = np.pi*i/(5000)
            x_c = int((self.diam+self.pix)*np.sin(rad)/2)
            y_c = int((self.diam+self.pix)*np.cos(rad)/2)

            y_coord.append(y_c+int(4*self.diam))
            x_coord.append(x_c+int(3*self.diam))
            theta_rad.append(rad)
            theta.append(rad*180/np.pi)
            order.append(i)

        x_indices_right = x_coord[:len(x_coord)//2]
        y_indices_right = y_coord[:len(y_coord)//2]
        x_indices_left = x_coord[len(x_coord)//2:]
        y_indices_left = y_coord[len(y_coord)//2:]

        new_order = [value - len(y_coord)//4 for value in order]

        x_coord_tb = [x_coord[i] for i in new_order]
        y_coord_tb = [y_coord[i] for i in new_order]

        x_indices_top = x_coord_tb[:len(x_coord_tb)//2]
        y_indices_top = y_coord_tb[:len(y_coord_tb)//2]
        x_indices_bot = x_coord_tb[len(x_coord_tb)//2:]
        y_indices_bot = y_coord_tb[len(y_coord_tb)//2:]

        pressure_2D = pressure[self.resz//2]

        pressure_all = np.zeros_like(pressure_2D)
        pressure_left = np.zeros_like(pressure_2D)
        pressure_right = np.zeros_like(pressure_2D)
        pressure_top = np.zeros_like(pressure_2D)
        pressure_bot = np.zeros_like(pressure_2D)

        pressure_all[x_coord, y_coord] = 1.0
        pressure_left[x_indices_left, y_indices_left] = 1.0
        pressure_right[x_indices_right, y_indices_right] = 1.0
        pressure_top[x_indices_top, y_indices_top] = 1.0
        pressure_bot[x_indices_bot, y_indices_bot] = 1.0


        # Debug only in the first it
        if it ==0:
            # For now
            net_name = self.network

            fig, ax = plt.subplots(figsize=(20,15))
            ax.imshow(pressure_2D[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'))
            ax.imshow(pressure_all[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'), alpha= 0.5)
            ax.grid()
            fig.savefig(os.path.join(self.save_folder, 'debug_pressure_{}.png'.format(net_name)))
            plt.close()

            fig, ax = plt.subplots(figsize=(20,15))
            ax.imshow(pressure_2D[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'))
            ax.imshow(pressure_top[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'), alpha=0.5)
            ax.grid()
            fig.savefig(os.path.join(self.save_folder, 'debug_pressure_top_{}.png'.format(net_name)))
            plt.close()

            fig, ax = plt.subplots(figsize=(20,15))
            ax.imshow(pressure_2D[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'))
            ax.imshow(pressure_bot[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'), alpha=0.5)
            ax.grid()
            fig.savefig(os.path.join(self.save_folder, 'debug_pressure_bot_{}.png'.format(net_name)))
            plt.close()

            fig, ax = plt.subplots(figsize=(20,15))
            ax.imshow(pressure_2D[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'))
            ax.imshow(pressure_left[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'), alpha=0.5)
            ax.grid()
            fig.savefig(os.path.join(self.save_folder, 'debug_pressure_left_{}.png'.format(net_name)))
            plt.close()

            fig, ax = plt.subplots(figsize=(20,15))
            ax.imshow(pressure_2D[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'))
            ax.imshow(pressure_right[int(2*self.diam):int(4*self.diam), int(3*self.diam):int(5*self.diam)], cmap= plt.get_cmap('seismic'), alpha=0.5)
            ax.grid()
            fig.savefig(os.path.join(self.save_folder, 'debug_pressure_right_{}.png'.format(net_name)))
            plt.close()


        # Correct Pressure with dt
        pressure /= self.dt

        # Shift pressure value on the network
        # As only grad p is used, the absolute values does not necessarily need to be correctly computed, so
        # it is shifted imposing cp=1 in the front stagnation point
        if 2*pressure[int(2*self.diam), x_coord_tb, y_coord_tb][0]<0.5 or 2*pressure[int(2*self.diam), x_coord_tb, y_coord_tb][0]>1.5:
            pressure -= (pressure[int(2*self.diam), x_coord_tb, y_coord_tb][0]-0.5)

        cp = np.mean((2*pressure[int(self.diam):int(3*self.diam), x_coord_tb, y_coord_tb]), axis=0)
        cd = np.mean((2*np.sum(pressure[int(self.diam):int(3*self.diam), x_coord_tb, y_coord_tb]*np.cos(theta_rad), axis=1)*np.pi/(10000)))
        cl = np.mean((2*np.sum(pressure[int(self.diam):int(3*self.diam), x_coord_tb, y_coord_tb]*np.sin(theta_rad), axis=1)*np.pi/(10000)))

        return cp, cl, cd


    def load_data(self):

        # Loop over saved timesteps
        for it in range(self.n_its):
            # Load Velocity and pressure fields, and save velocity values to ensure probe is OK!
            u_loaded = np.load(os.path.join(self.loading_folder, 'U_output_{}.npy'.format(self.init_it + it*self.interval)))
            p_loaded = np.load(os.path.join(self.loading_folder, 'P_output_{}.npy'.format(self.init_it + it*self.interval)))
            u_zone = u_loaded[:, self.resz//2-self.diam:self.resz//2+self.diam,
                                         self.resy//4 +int(1.0*self.diam):self.resy//4 +int(2.0*self.diam),
                                         self.resx//2-int(0.5*self.diam):self.resx//2+int(0.5*self.diam)]
            self.probe[:, it] = np.mean(u_zone, axis=(1,2,3))

            cp, self.cl[it], self.cd[it] = self.compute_cp_and_cl(p_loaded, it)
            #print(f'Coefficient values: Cl = {self.cl[it]} and Cd = {self.cd[it]}')

            if it%self.outiter == 0:
                vorticity = self.compute_vorticity(u_loaded)

                # To debug velocity and p fields, plot the Z//2 plane!
                plot_imshow(u_loaded[0, self.resz//2], 'Ux', os.path.join(self.save_folder, 'Ux_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(u_loaded[1, self.resz//2], 'Uy', os.path.join(self.save_folder, 'Uy_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(u_loaded[2, self.resz//2], 'Uz', os.path.join(self.save_folder, 'Uz_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(p_loaded[self.resz//2], 'P', os.path.join(self.save_folder, 'P_debug_{}_it_{}.png'.format(network, it*self.interval)))

                plot_imshow(vorticity[0, self.resz//2, 10:-10, 10:-10], 'Vor X', os.path.join(self.save_folder, 'Vorticityzplane_X_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(vorticity[1, self.resz//2, 10:-10, 10:-10], 'Vor Y', os.path.join(self.save_folder, 'Vorticityzplane_Y_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(vorticity[2, self.resz//2, int(1.5*self.diam):int(8.5*self.diam), int(1.5*self.diam):int(6.5*self.diam)], 'Vor Z', os.path.join(self.save_folder, 'Vorticityzplane_Z_debug_{}_it_{}.pdf'.format(network, it*self.interval)))

                plot_imshow(vorticity[0, 10:-10, int(1.5*self.diam):int(8.5*self.diam), self.resx//2], 'Vor X', os.path.join(self.save_folder, 'Vorticityxplane_X_debug_{}_it_{}.pdf'.format(network, it*self.interval)))
                plot_imshow(vorticity[1, 10:-10, int(1.5*self.diam):int(8.5*self.diam), self.resx//2], 'Vor Y', os.path.join(self.save_folder, 'Vorticityxplane_Y_debug_{}_it_{}.pdf'.format(network, it*self.interval)))
                plot_imshow(vorticity[2, 10:-10, int(1.5*self.diam):int(8.5*self.diam), self.resx//2], 'Vor Z', os.path.join(self.save_folder, 'Vorticityxplane_Z_debug_{}_it_{}.pdf'.format(network, it*self.interval)))

                plot_imshow(vorticity[0, 10:-10, int(1.5*self.diam):int(8.5*self.diam), self.resx//2 + self.resx//8], 'Vor X', os.path.join(self.save_folder, 'Vorticityxplane_l_X_debug_{}_it_{}.pdf'.format(network, it*self.interval)))
                plot_imshow(vorticity[1, 10:-10, int(1.5*self.diam):int(8.5*self.diam), self.resx//2 + self.resx//8], 'Vor Y', os.path.join(self.save_folder, 'Vorticityxplane_l_Y_debug_{}_it_{}.pdf'.format(network, it*self.interval)))
                plot_imshow(vorticity[2, 10:-10, int(1.5*self.diam):int(8.5*self.diam), self.resx//2 + self.resx//8], 'Vor Z', os.path.join(self.save_folder, 'Vorticityxplane_l_Z_debug_{}_it_{}.pdf'.format(network, it*self.interval)))

                plot_imshow(vorticity[0, 10:-10, self.resy//2, 10:-10], 'Vor X', os.path.join(self.save_folder, 'Vorticityyplane_X_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(vorticity[1, 10:-10, self.resy//2, 10:-10], 'Vor Y', os.path.join(self.save_folder, 'Vorticityyplane_Y_debug_{}_it_{}.png'.format(network, it*self.interval)))
                plot_imshow(vorticity[2, 10:-10, self.resy//2, 10:-10], 'Vor Z', os.path.join(self.save_folder, 'Vorticityyplane_Z_debug_{}_it_{}.png'.format(network, it*self.interval)))

                plot_line([cp], ['Probe'], os.path.join(self.save_folder, 'Cp_evolution_{}_it_{}.png'.format(network, it*self.interval)))


        # Get Strouhal
        self.get_st()

        # Debug to compare the probe saved by the simulation and the one loaded from velocity fields
        plot_line([self.probe[0], self.probe[1], self.probe[2]], ['Ux', 'Uy', 'Uz'], os.path.join(self.save_folder, 'Probe_loaded_debug_Re_{}.png'.format(self.re)))
        plot_cl_vs_cd(self.cl, self.cd, os.path.join(self.save_folder, 'CDCL_{}_Re_{}.png'.format(network, self.re)))

        # Store amplitudes and reference
        self.amp = np.max(self.cl)-np.min(self.cl)
        if 'CG' in network:
            print('Reference found')
            self.ref_amp = self.amp

        # Final results
        print(f'Strouhal number: {self.st:.3f}')
        print(f'Averages  : Lift = {(np.mean(self.cl)):.3f} and Drag = {(np.mean(self.cd)):.3f} ')
        print(f'Amplitudes: Lift = {((np.max(self.cl)-np.min(self.cl))):.3f} and Drag = {((np.max(self.cd)-np.min(self.cd))):.3f} ')
        print(f'Error for lift amplitude = {(100*abs(self.ref_amp - self.amp)/self.ref_amp):.3f} %')

if __name__ == '__main__':

    with open('config_3D.yml', 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)


    reynolds_studies = {}

    # Loop over networks
    for i, network in enumerate(config['networks']):
            print('=============================================================')
            print(f'Starting Network: {network}')
            print('=============================================================')

            study = Study_3D(config, network)

            if 'CG' not in network:
                study.ref_amp = ref_amp

            study.load_data()

            # Store reference value
            if 'CG' in network:
                ref_amp = study.ref_amp

    # Loop over Reynolds and Resolutions
    # for i, Re in enumerate(config['Reynolds']):
    #         study = Study_3D(config, Re)
    #         study.load_data()
