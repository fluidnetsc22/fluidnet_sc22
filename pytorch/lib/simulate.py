import pdb
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch

import lib.fluid as fluid


def setConstVals(batch_dict, p, U, flags, density):
    # apply external BCs.
    # batch_dict at output = {p, UDiv, flags, density, UBC,
    #                         UBCInvMask, densityBC, densityBCInvMask}

    if ('UBCInvMask' in batch_dict) and ('UBC' in batch_dict):

        if batch_dict['Test_case'] == 'Step':

            # Zero out the U values on the BCs.
            # U.mul_(batch_dict['UBCInvMask'])

            Mask = batch_dict['densityBCInvMask']

            U_sizes = batch_dict['U']
            bsz = U_sizes.size(0)
            d = U_sizes.size(2)
            h = U_sizes.size(3)
            w = U_sizes.size(4)

            i = torch.arange(start=0, end=w, dtype=torch.float) \
                .view(1, w).expand(bsz, d, h, w)
            h_s = np.int(w / 2.0)
            u_scale = batch_dict['Step']

            input_U = -6.0 * u_scale * (i) * (i - h_s) / (h_s * h_s)
            input_U[:, :, :, h_s:-1] = 0.0
            output_U = -3.0 * u_scale * (i) * (i - 2.0 * h_s) / (4. * (h_s * h_s))

            U[:, 1, :, 1:5, 1:-1] = input_U[:, :, 1:5, 1:-1]
            U[:, 1, :, -6:, 1:-1] = output_U[:, :, -6:, 1:-1]

            batch_dict['U'] = U.clone()

        else:

            # Zero out the U values on the BCs.
            # U.mul_(batch_dict['UBCInvMask'])
            Mask = batch_dict['densityBCInvMask']
            U[:, :, :, 1:2, :] = 0
            U.mul_(batch_dict['densityBCInvMask'])

            # Add back the values we want to specify.
            U.add_(batch_dict['UBC'])
            batch_dict['U'] = U.clone()

    if ('densityBCInvMask' in batch_dict) and ('densityBC' in batch_dict):

        Mask = batch_dict['densityBCInvMask']
        density.mul_(batch_dict['densityBCInvMask'])
        density.add_(batch_dict['densityBC'])
        batch_dict['density'] = density.clone()

# def simulate(mconf, batch_dict, net, sim_method, output_div=False):


def simulate(
        mconf,
        batch_dict,
        net,
        sim_method,
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
        Outside_Ja,
        output_div=False):
    """Top level simulation loop.

    Arguments:
        mconf (dict): Model configuration dictionnary.
        batch_dict (dict): Dictionnary of torch Tensors.
            Keys must be 'U', 'flags', 'p', 'density', plus several other depending on the test case.
            Usually, Max Div, VK as well as the parameters for the CG (IA, val, JA)
            Simulations are done INPLACE.
        net (nn.Module): convNet model.
        sim_method (string): Options are 'convnet', 'CG','PCG' and 'jacobi'
        Time_vec (np.array(max_it)): Time taken for the Poisson equation resolution at each it.
        Time_Pres (np.array(max_it)): Time taken for the Pressure correction step at each it.
        Jacobi_switch (np.array(max_it)): In case of hybrid resolution, number of Jacobi it needed.
        Max_Div (np.array(max_it)): Max divergence level after the first Poisson equation resolution.
        Mean_Div (np.array(max_it)): Mean divergence level after the first Poisson equation resolution.
        Max_Div_All (np.array(max_it)): Max divergence level after the whole pressure resolution process,
        specially useful for the hybrid case as shows the final divergence, whereas the Max div corresponds
        to the divergence before the extra Jacobi its.
        folder (string): Output folder, maybe not the most efficient way to add it, but useful for saving purposes.
        it (int): actual iteration n.
        Threshold_Div (float): In case of Hybrid method, stopping divergence level.
        dt (float): Time dt (input value).
        Outside_Ja (bool): If Hybrid method is activated or not.
        output_div (bool, optional): returns just before solving for pressure.
            i.e. leave the state as UDiv and pDiv (before substracting divergence)


    """

    cuda = torch.device('cuda')
    assert sim_method == 'convnet' or sim_method == 'jacobi' or sim_method == 'PCG'\
        or sim_method == 'CG', 'Simulation method \
                not supported. Choose either convnet, PCG, CG or jacobi.'

    #dt = arguments.setdt or float(mconf['dt'])
    maccormackStrength = mconf['maccormackStrength']
    sampleOutsideFluid = mconf['sampleOutsideFluid']

    buoyancyScale = mconf['buoyancyScale']
    gravityScale = mconf['gravityScale']

    viscosity = mconf['viscosity']
    assert viscosity >= 0, 'Viscosity must be positive'

    # Get p, U, flags and density from batch.
    p = batch_dict['p']
    U = batch_dict['U']
    flags = batch_dict['flags']

    Div_analysis = True
    stick = False

    if 'flags_stick' in batch_dict:
        stick = True
        print(" Stick = True ")
        flags_stick = batch_dict['flags_stick']

    # If viscous model, add viscosity
    if (viscosity > 0):
        orig = U.clone()
        fluid.addViscosity(dt, orig, flags, viscosity)

    if 'density' in batch_dict:
        density = batch_dict['density']

        # First advect all scalar fields.
        density = fluid.advectScalar(dt, density, U, flags,
                                     method="eulerFluidNet", \
                                     # method="maccormackFluidNet", \
                                     boundary_width=1, sample_outside_fluid=sampleOutsideFluid, \
                                     #                maccormack_strength=maccormackStrength)
                                     )

        if mconf['correctScalar']:
            div = fluid.velocityDivergence(U, flags)
            fluid.correctScalar(dt, density, div, flags)
    else:
        density = torch.zeros_like(flags)

    flags_only = flags.clone()

    if viscosity == 0:
        # Self-advect velocity if inviscid
        U = fluid.advectVelocity(dt=dt, orig=U, U=U, flags=flags,
                                 method="eulerFluidNet", \
                                 # method="maccormackFluidNet", \
                                 boundary_width=1, maccormack_strength=maccormackStrength)

    else:
        # Advect viscous velocity field orig by the non-divergent
        # velocity field U.
        U = fluid.advectVelocity(dt=dt, orig=orig, U=U, flags=flags,
                                 method="eulerFluidNet", \
                                 # method="maccormackFluidNet", \
                                 boundary_width=1, maccormack_strength=maccormackStrength)

    if 'density' in batch_dict:
        if buoyancyScale > 0:
            # Add external forces: buoyancy.
            gravity = torch.FloatTensor(3).fill_(0).cuda()
            gravity[0] = mconf['gravityVec']['x']
            gravity[1] = mconf['gravityVec']['y']
            gravity[2] = mconf['gravityVec']['z']
            gravity.mul_(-buoyancyScale)
            rho_star = mconf['operatingDensity']

            # Buoyancy adding ... Different functions were used (with not great success):
            U = fluid.addBuoyancy(U, flags, density, gravity, rho_star, dt)

        if gravityScale > 0:
            gravity = torch.FloatTensor(3).fill_(0).cuda()
            gravity[0] = mconf['gravityVec']['x']
            gravity[1] = mconf['gravityVec']['y']
            gravity[2] = mconf['gravityVec']['z']
            # Add external forces: gravity.
            gravity.mul_(-gravityScale)
            U = fluid.addGravity(U, flags, gravity, dt)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()

        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
                U[:, 0, :, :, 1] = U_temp[:, 0, :, :, U.size(4) - 1]
                density[:, 0, :, :, 1] = density_temp[:, 0, :, :, U.size(4) - 1]
            if mconf['periodic-y']:

                U[:, 1, :, :, -1] = U_temp[:, 1, :, :, 1]
                U[:, 0, :, :, -1] = -U_temp[:, 0, :, :, 1]

    if sim_method == 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:

                U[:, 1, :, :, 1] = U_temp[:, 1, :, :, U.size(4) - 1]
                density[:, 0, :, :, 1] = density_temp[:, 0, :, :, U.size(4) - 1]

            if mconf['periodic-y']:

                print(" Periodic y 0 !")

                U[:, 1, :, :, -1] = U_temp[:, 1, :, :, 1]
                U[:, 0, :, :, -1] = -U_temp[:, 0, :, :, 1]

    elif stick:
        fluid.setWallBcsStick(U, flags, flags_stick)

    # if sim_method == 'convnet':
    U = fluid.setWallBcs(U, flags)

    # Special VK
    if 'VK' in batch_dict.keys():
        #print("VK BC Before")
        BC_V = batch_dict['VK']
        U = fluid.setWallVKBcs(U, flags, BC_V)

    if batch_dict['Test_case'] == 'Step':
        print("Step")
        BC_V = batch_dict['Step']
        U = fluid.setWallStepBcs(U, flags, BC_V)

    setConstVals(batch_dict, p, U, flags, density)

    batch_dict['U'] = U
    div = fluid.velocityDivergence(U, flags)
    Advected_Div = (abs(div).max()).item()

    # Save velocity field after the advection step!
    # if Div_analysis:
    Ustar = U.clone()
    batch_dict['Ustar'] = Ustar
    # Timing for the whole P solving
    start_Pres = default_timer()

    if (sim_method == 'convnet'):

        # Uses the model to perform the pressure projection and velocity calculation.
        # Set wall BCs is performed inside the model, before and after the projection.
        # No need to call it again.

        #UDiv = fluid.setWallBcs(UDiv, flags)
        #U = fluid.setWallBcs(U, flags)

        if (batch_dict['Test_case'] == 'RT' and it > 0):

            # Save the divergence that is inputted to the Network
            div = fluid.velocityDivergence(U, flags)
            Advected_Div = (abs(div).max()).item()

            initial_p = batch_dict['Init_p']
            fluid.velocityUpdate(pressure=initial_p, U=U, flags=flags)

            # Save the divergence that is inputted to the Network
            div = fluid.velocityDivergence(U, flags)
            Advected_Div = (abs(div).max()).item()

        div = fluid.velocityDivergence(U, flags)

        if Div_analysis:
            div_input = div.clone()
            batch_dict['Div_in'] = div_input

        # It might be strait forward ... BUT remember that the model is saved in:
        # ../MODELFOLDER/MODELNAME_saved.py

        net.eval()
        data = torch.cat((p, U, flags, density), 1)
        p, U, time = net(data, it, folder)

        # Transform time
        time = time.data[0]
        # Set BC
        setConstVals(batch_dict, p, U, flags, density)

        # Special VK
        if batch_dict['Test_case'] == 'VK':
            #print("VK BC")
            BC_V = batch_dict['VK']
            U = fluid.setWallVKBcs(U, flags, BC_V)

        if batch_dict['Test_case'] == 'Step':
            print("Step")
            BC_V = batch_dict['Step']
            U = fluid.setWallStepBcs(U, flags, BC_V)

        #U = batch_dict['U']

        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:

                U[:, 1, :, :, 1] = U_temp[:, 1, :, :, U.size(4) - 1]
                density[:, 0, :, :, 1] = density_temp[:, 0, :, :, U.size(4) - 1]

            if mconf['periodic-y']:

                print(" Periodic y 1 !")

                U[:, 1, :, :, -1] = U_temp[:, 1, :, :, 1]
                U[:, 0, :, :, -1] = -U_temp[:, 0, :, :, 1]

    elif (sim_method == 'jacobi'):

        div = fluid.velocityDivergence(U, flags)

        bsz = div.size(0)
        ch = div.size(1)
        d = div.size(2)
        h = div.size(3)
        w = div.size(4)

        if Div_analysis:
            div_input = div.clone()
            batch_dict['Div_in'] = div_input

        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        load_file = folder + '/Jacobi_switch_loading.npy'
        #Maxi_Try = np.load(load_file)
        maxIter = mconf['jacobiIter']

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        p, residual = fluid.solveLinearSystemJacobi(
            flags=flags, div=div, is_3d=is3D, p_tol=pTol,
            max_iter=maxIter)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        #print("elapsed Jacobi time: ", elapsed_time_ms)

        time = elapsed_time_ms

        resX = resY = resZ = 10

        X = torch.arange(0, resX).view(resX).expand((1, 1, resZ, resY, resX)).cuda().float()
        Y = torch.arange(0, resY).view(resY, 1).expand((1, 1, resZ, resY, resX)).cuda().float()
        Z = torch.arange(0, resZ).view(resZ, 1, 1).expand((1, 1, resZ, resY, resX)).cuda().float()

        pressure_ex = torch.zeros_like(p)

        #fluid.velocityUpdate_Density(pressure=p, U=U, flags=flags, density=density)
        fluid.velocityUpdate(pressure=p, U=U, flags=flags)
        setConstVals(batch_dict, p, U, flags, density)

        # Special VK
        if batch_dict['Test_case'] == 'VK':
            #print("VK BC")
            BC_V = batch_dict['VK']
            U = fluid.setWallVKBcs(U, flags, BC_V)

        if batch_dict['Test_case'] == 'Step':
            print("Step")
            BC_V = batch_dict['Step']
            U = fluid.setWallStepBcs(U, flags, BC_V)

        #U = fluid.setWallVKBcs(U, flags)
        U = batch_dict['U']

    elif (sim_method == 'CG'):

        div = fluid.velocityDivergence(U, flags)

        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        maxIter_CG = mconf['jacobiIter']
        pTol_CG = pTol

        A_val = batch_dict['Val']
        I_A = batch_dict['IA']
        J_A = batch_dict['JA']

        # Input
        if Div_analysis:
            div_input = div.clone()
            batch_dict['Div_in'] = div_input

        # Modify rhs for inflow
        #fluid.set_inflow_bc(div, U, flags, batch_dict)

        # Timing Test
        start = default_timer()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        residual = fluid.solveLinearSystemCG(flags, p, div,
                                             A_val, I_A, J_A, is_3d=is3D, p_tol=pTol_CG,
                                             max_iter=maxIter_CG)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)

        time = elapsed_time_ms

        if (batch_dict['Test_case'] == 'RT' and it < 1):
            initial_p = p.clone()
            batch_dict['Init_p'] = initial_p

        fluid.velocityUpdate(pressure=p, U=U, flags=flags)
        #U = fluid.setWallBcs(U, flags)
        setConstVals(batch_dict, p, U, flags, density)

        # Special VK
        if 'VK' in batch_dict.keys():
            #print("VK BC After")
            BC_V = batch_dict['VK']
            U = fluid.setWallVKBcs(U, flags, BC_V)

        if batch_dict['Test_case'] == 'Step':
            print("Step")
            BC_V = batch_dict['Step']
            U = fluid.setWallStepBcs(U, flags, BC_V)

        div_out = fluid.velocityDivergence(U, flags)
        U = batch_dict['U']

    elif (sim_method == 'PCG'):
        is3D = (U.size(2) > 1)
        pTol = mconf['pTol']
        maxIter = mconf['jacobiIter']
        maxIter_PCG = 1
        pTol_PCG = 0.5e-6

        # Input
        if Div_analysis:
            div_input = batch_dict['Div_in']
            div_input = div.clone()

        # Timing Test
        start = default_timer()

        # Inflow
        inflow = torch.zeros_like(flags)
        inflow_border = torch.zeros_like(flags)

        inflow_bool = False

        if inflow_bool:

            inflow = ((batch_dict['UBC'][:, 1, :, :, :]).unsqueeze(1)) > 0.0001
            inflow_border[0, 0, 0, 1, :] = inflow[0, 0, 0, 1, :]

        # Debug
        print(" ========================================================================")
        print("IT  ", it)
        print(" ========================================================================")

        p, residual = fluid.solveLinearSystemPCG(
            flags=flags, div=div, inflow=inflow_border, is_3d=is3D, p_tol=pTol_PCG,
            max_iter=maxIter_PCG)

        end = default_timer()
        time = (end - start)
        print("time", time)

        fluid.velocityUpdate(pressure=p, U=U, flags=flags)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        U = fluid.setWallBcs(U, flags)
        setConstVals(batch_dict, p, U, flags, density)

        U = batch_dict['U']
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:
                U[:, 1, :, :, 1] = U_temp[:, 1, :, :, U.size(4) - 1]
                density[:, 0, :, :, 1] = density_temp[:, 0, :, :, U.size(4) - 1]
            if mconf['periodic-y']:
                U[:, 1, :, :, -1] = U_temp[:, 1, :, :, 1]
                U[:, 0, :, :, -1] = -U_temp[:, 0, :, :, 1]

    div_after = fluid.velocityDivergence(U, flags)

    # Time Vec Saving
    Time_vec[it] = time
    filename = folder + '/Time'
    np.save(filename, Time_vec)

    Threshold = Threshold_Div

    div_after = fluid.velocityDivergence(U, flags)
    Max_Div[it] = (abs(div_after).max()).item()
    Mean_Div[it] = (abs(div_after).mean()).item()

    if (Outside_Ja):

        print(" Treshold surpassed ==========================================================> ")

        Jacobi_switch[it] = 1
        Counter = 0

        loaded_mean_div = np.load('/manually/set/path/to/Mean_Div.npy')
        while (abs(div_after).mean()).item() > loaded_mean_div[it]:

            print(" Treshold surpassed ==========================================================> ")
            Jacobi_switch[it] += 1

            div = fluid.velocityDivergence(U, flags)
            is3D = (U.size(2) > 1)
            pTol = mconf['pTol']
            maxIter = mconf['jacobiIter']

            Jacobi_switch[it] += 1

            div = fluid.velocityDivergence(U, flags)

            is3D = (U.size(2) > 1)
            pTol = mconf['pTol']
            maxIter = 1

            p, residual = fluid.solveLinearSystemJacobi(
                flags=flags, div=div, is_3d=is3D, p_tol=pTol,
                max_iter=maxIter)

            fluid.velocityUpdate(pressure=p, U=U, flags=flags)

            U = fluid.setWallBcs(U, flags)
            setConstVals(batch_dict, p, U, flags, density)
            U = batch_dict['U']

            Counter += 1
            div_after = fluid.velocityDivergence(U, flags)

        print("Ending Divergence level: ", (abs(div_after).max()).item(),
              " . Number of Jacobi its needed:  {} ".format(Jacobi_switch[it]))

    end_Pres = default_timer()
    time_Pressure = (end_Pres - start_Pres)

    Time_Pres[it] = time_Pressure
    filename_pres = folder + '/Time_Pres'
    np.save(filename_pres, Time_Pres)

    div_final = fluid.velocityDivergence(U, flags)
    Max_Div_All[it] = (abs(div_final).max()).item()

    filename_div = folder + '/Max_Div_All'
    np.save(filename_div, Max_Div_All)

    if sim_method != 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:

                U[:, 1, :, :, 1] = U_temp[:, 1, :, :, U.size(4) - 1]
                density[:, 0, :, :, 1] = density_temp[:, 0, :, :, U.size(4) - 1]

            if mconf['periodic-y']:

                print(" Periodic y !")

                U[:, 1, :, :, -1] = U_temp[:, 1, :, :, 1]
                U[:, 0, :, :, -1] = -U_temp[:, 0, :, :, 1]

    if sim_method == 'convnet':
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            U_temp = U.clone()
            # Density peridoicty
            density_temp = density.clone()
        if 'periodic-x' in mconf and 'periodic-y' in mconf:
            if mconf['periodic-x']:

                U[:, 1, :, :, 1] = U_temp[:, 1, :, :, U.size(4) - 1]
                density[:, 0, :, :, 1] = density_temp[:, 0, :, :, U.size(4) - 1]

            if mconf['periodic-y']:

                print(" Periodic y 2 !")
                U[:, 1, :, :, -1] = U_temp[:, 1, :, :, 1]
                U[:, 0, :, :, -1] = -U_temp[:, 0, :, :, 1]

    if Div_analysis:
        batch_dict['Ustar'] = Ustar
        batch_dict['Div_in'] = div_input

    batch_dict['U'] = U
    batch_dict['density'] = density
    batch_dict['p'] = p
