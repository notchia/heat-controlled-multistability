""" Model for magnetic multistable metamaterial quasistatic force and potential energy
    
    Represents the flexible hinge as linear torsional spring and the magnets as
    dipoles. For both, fit parameters and compute resulting force and energy.    

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.colors as col
from time import process_time

import metamaterialmodel as energy
import pltformat  # module: no funcions, just standardized plot formatting


"""" limFlag options: 'exp', 'piecewise' """
class ParameterSpace:
    def __init__(self, h_range, r_range, T_range, thetaL_range):
        self.h_range = h_range
        self.r_range = r_range
        self.T_range = T_range
        self.thetaL_range = thetaL_range


    """ Plot change in angle with temperature """   
    def plot_angle_temp(self, T_range, angles):
        plt.figure()
        plt.plot(T_range, np.degrees(angles),'k')
        plt.ylabel("Angle (degrees)")
        plt.xlabel("Temperature (Celcius)")
        plt.title("h = {0}, r = {1}".format(self.h_total, self.ratio))
        plt.tight_layout()
        
        return
    
    """ Map stability along three parameter axes: thickness, LCE:PDMS ratio, initial angle """
    def analyze_parameter_space(self):
        q = np.radians(np.arange(-45.0,45.5,0.5))
        
        n_T = len(self.T_range)
        n_ratio = len(self.r_range)
        n_h = len(self.h_range)
        n_L = len(self.thetaL_range)
        
        angleT_array = np.zeros((n_T, n_h, n_ratio, n_L))
        angle0_array = np.zeros((n_T, n_h, n_ratio, n_L))
        numMin_array = np.zeros((n_T, n_h, n_ratio, n_L))
        phase_array = np.zeros((n_T, n_h, n_ratio, n_L))
       
        for i_h in range(n_h):
            sys.stdout.write("\rMapping parameter space: analyzing {0}/{1} at {2:0.1f} minutes".format(
                    i_h + 1, n_h, (process_time()-t_start)/60.0))
            sys.stdout.flush()
            for i_r in range(n_ratio):
                for i_L in range(n_L):
                    thetaL = self.thetaL_range[i_L]
                    sample = energy.MetamaterialModel(h_total=self.h_range[i_h],
                                                      ratio=self.r_range[i_r], thetaL=thetaL)
                    for i_t in range(n_T):
                        sample.update_T(T_range[i_t])
                        U_total, U_m, U_k, U_lim = sample.calculate_total_energy(q)
                        thetaT = sample.thetaT
                        
                        numMin, diff, barrier, phase = sample.analyze_energy_local_extrema(U_total, q, thetaL)
    
                        phase_array[i_t, i_h, i_r, i_L] = phase
                        numMin_array[i_t, i_h, i_r, i_L] = numMin
                        angleT_array[i_t, i_h, i_r, i_L] = np.degrees(thetaT)
                        angle0_array[i_t, i_h, i_r, i_L] = np.degrees(thetaT + thetaL)
    
        return numMin_array, phase_array, angleT_array, angle0_array



#-----------------------------------------------------------------------------
            

            
""" Plot 3D phase diagram """  
def find_phase_boundaries(T_range, h_range, ratio_range, thetaL_range,
                          minima, phases, angleT_vals, angle0_vals):   
    for ir in range(len(self.r_range)):
        diffs = np.diff(phases[:,:,ir,:])
        boundaries = np.argwhere(diffs != 0)
        N = max(boundaries.shape)
        boundary_vals = np.zeros(N)
        for pt in range(N):
            loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
            val = phases[loc[0],loc[1],ir,loc[2]]
            max_diff = 0
            check = [-1,0,1]
            vals = np.zeros(9,dtype=int)
            if not ((-1 in loc) or loc[0] == phases.shape[0]-1 or loc[1] == phases.shape[1]-1 or loc[2] == phases.shape[3]-1):
                count = 0
                for i in check:
                    for j in check:
                        val_temp = phases[loc[0]+i,loc[1]+j,ir,loc[2]]
                        val_diff = np.abs(val - val_temp)
                        if val_diff > max_diff:
                            max_diff = val_diff
                        vals[count] = val_temp
                        count += 1
            if max_diff == 0:
                boundary_vals[pt] = np.nan#-1
            elif (0 in vals) and (1 in vals):
                boundary_vals[pt] = 0
            elif (0 in vals) and (1 not in vals):
                boundary_vals[pt] = 1
            elif ((2 in vals) or (3 in vals)) and (5 not in vals):
                boundary_vals[pt] = 3
            elif ((4 in vals) or (6 in vals)) and (5 not in vals):
                boundary_vals[pt] = 2
            elif (max_diff == 2 or max_diff == 3 or max_diff == 5):
                boundary_vals[pt] = 6
            elif 1 in vals:
                boundary_vals[pt] = 3
            else:
                boundary_vals[pt] = 5

    return boundaries, boundary_vals
    

''' Plot 3D phase diagram for given ratio index'''
def plot_phase_boundaries(boundaries, boundary_vals,
                          T_range, h_range, ir, thetaL_range,
                          minima, phases, angleT_vals, angle0_vals,
                          T_plane=0):
    if T_plane:
        iT = np.argwhere(T_range > T_plane)[0][0]
    colors = ['xkcd:light red', 'xkcd:red', 'xkcd:apple green', 'xkcd:dark lime green', 'xkcd:dark lime green', 'xkcd:electric blue','xkcd:electric blue']
    colors3 = ['xkcd:light red', 'xkcd:apple green', 'xkcd:electric blue']
    customcm = col.ListedColormap(colors3[::-1])
      
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-105)
    x = 1000*self.h_range
    y = np.degrees(self.thetaL_range)
    z = T_range
    X, Y = np.meshgrid(x, y)

    x_below = np.full_like(boundary_vals, np.nan)
    x_above = np.full_like(boundary_vals, np.nan)
    y_below = np.full_like(boundary_vals, np.nan)
    y_above = np.full_like(boundary_vals, np.nan)
    z_below = np.full_like(boundary_vals, np.nan)
    z_above = np.full_like(boundary_vals, np.nan)
    nPoints = max(boundaries.shape)
    for pt in range(nPoints):
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        if loc[0] <= iT:
            x_below[pt] = x[loc[1]] 
            y_below[pt] = y[loc[2]] 
            z_below[pt] = z[loc[0]] 
        else:
            x_above[pt] = x[loc[1]] 
            y_above[pt] = y[loc[2]] 
            z_above[pt] = z[loc[0]]             

    iL = np.argwhere(self.thetaL_range > 0)[0][0]

    x_rb = np.full_like(boundary_vals, np.nan)
    x_rt = np.full_like(boundary_vals, np.nan)
    y_rb = np.full_like(boundary_vals, np.nan)
    y_rt = np.full_like(boundary_vals, np.nan)
    z_rb = np.full_like(boundary_vals, np.nan)
    z_rt = np.full_like(boundary_vals, np.nan)
    for pt in range(nPoints):
        loc = [boundaries[pt,0], boundaries[pt,1], boundaries[pt,2]]
        if loc[2] >= iL:
            if loc[0] <= iT:
                x_rb[pt] = x[loc[1]] 
                y_rb[pt] = y[loc[2]] 
                z_rb[pt] = z[loc[0]] 
            else:
                x_rt[pt] = x[loc[1]] 
                y_rt[pt] = y[loc[2]] 
                z_rt[pt] = z[loc[0]]  
                
    diagram = ax.scatter(x_below, y_below, z_below, c=boundary_vals, 
                         marker='.', cmap=col.ListedColormap(colors), zorder=1)        
    if T_plane:
        ax.plot_surface(X, Y, np.full_like(Y, T_range[iT]), rstride=1, cstride=1, facecolors=customcm(np.transpose((minima[iT,:,ir,:])-1)/3), alpha=0.075, zorder=2)
#        diagram = ax.scatter(x_above, y_above, z_above, marker='.', c=boundary_vals, cmap=col.ListedColormap(colors), zorder=3)
    diagram = ax.scatter(x_rt, y_rt, z_rt, marker='.', c=boundary_vals, cmap=col.ListedColormap(colors), zorder=3)        

    ax.set_xlabel(r'$h$ (mm)', fontsize=14)
    ax.set_ylabel(r'$\theta_L$ (deg)', fontsize=14, rotation=150)
    ax.set_zlabel(r'$T$ ($^{\circ}$C)', fontsize=14, rotation=60)
    bar = fig.colorbar(diagram, aspect=10)
        
#    # Mayavi version
#    x_b = boundaries[:,1]#x[boundaries[:,1]]
#    y_b = boundaries[:,2]#y[boundaries[:,2]]
#    z_b = boundaries[:,0] #z[boundaries[:,0]]
#    pts = mlab.points3d(x_b, y_b, z_b, boundary_vals)
#    mesh = mlab.pipeline.delaunay2d(pts)
#    pts.remove()
#    surf = mlab.pipeline.surface(mesh)
#    mlab.show()


''' Plot isotherm phase diagram for each composite: h vs total equilibrium angle'''
def plot_isotherms(T_range, h_range, ratio_range, thetaL_range, minima, phases, angleT_vals, angle0_vals, T_plane=False):
    
    for ir in range(len(self.r_range)):
        if T_plane:
            iT = np.argwhere(T_range > T_plane)[0][0]
            plot_single_isotherm(iT, ir, T_range, self.h_range, self.r_range, self.thetaL_range, minima, phases, angleT_vals, angle0_vals)
        else:
            for iT in range(len(T_range)):
                plot_single_isotherm(iT, ir, T_range, self.h_range, ratio_range, self.thetaL_range, minima, phases, angleT_vals, angle0_vals)
    return


def plot_single_isotherm(iT, ir, T_range, h_range, ratio_range, thetaL_range, minima, phases, angleT_vals, angle0_vals):
    colors = ['xkcd:light red', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:apple green', 'xkcd:electric blue', 'xkcd:blue purple', 'xkcd:electric blue']
    fig = plt.figure()
    ax = fig.gca()
    X = 1000*self.h_range
    Y =angle0_vals[iT,0,ir,:]
    Z = np.transpose(phases[iT,:,ir,:])
    diagram = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max() ], origin='lower', aspect='auto', cmap=col.ListedColormap(colors))           
    
    # Formatting
    plt.title(r'$T$ = {0}$^\circ$C, $r$ = {1:0.2f}'.format(int(T_range[iT]), self.r_range[ir]), size=18)
    plt.xlabel(r'hinge thickness $h$ (mm)')
    plt.ylabel(r'total angle $\theta_0$ (degrees)')
       
    plt.ylim([np.amin(angle0_vals), np.amax(angle0_vals)])
    plt.savefig('angle_scaled_0_{0}_r_{1:0.2f}.png'.format(int(T_range[iT]), self.r_range[ir]))
    plt.close()

    return


""" 2D color plot of angle on h-T axes """  
def plot_angleSurface(ratio_range, h_range, T_range, thetaL_range, angleT_vals):
    self.r_range, T_range = np.meshgrid(self.r_range, T_range)
    colorlevels = np.arange(np.amin(angleT_vals), np.amax(angleT_vals), 0.5)
    
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.contourf(self.r_range, T_range, angleT_vals[:,0,:,0], levels=colorlevels)    
    plt.xlabel('thickness ratio, LCE:total')
    plt.ylabel('Temperature (deg C)')
    plt.title('h = {0:0.2f} mm'.format(1000*self.h_range[0]))
    bar = fig.colorbar(surf, aspect=5)
    bar.set_label('Angle (degrees)')
    plt.savefig('r_vs_thetaT.png')
    plt.close()

    return

''' Given parameters, plot energy and return model, figure flag'''
def analyze_parameter_energy(h_total, ratio, T, thetaL):
    q = np.radians(np.arange(-45.0,45.5,0.5))
    sample = MetamaterialModel(h_total=h_total, ratio=ratio, T=T, thetaL=thetaL)
    U_total, U_m, U_k, U_lim = sample.calculate_total_energy(q)
    sample.plot_energy(q, U_total, U_m, U_k, U_lim)
    print("Equivalent torsional spring constant: {0}".format(sample.k))
    return sample
   
    
''' Export general 2D data array to csv file. fmt can be list '''
def export_to_csv(filename, data, headerList=[], fmt='%0.18e'):
    nCol = data.shape[1]
    if headerList == []:
        headerList = ['col{0}'.format(i) for i in range(nCol)]
    header = ", ".join(map(str, headerList))
    np.savetxt(filename, data, delimiter=', ', fmt=fmt, header=header)
    return filename

    
''' Export parameters used to map parameter space '''
def export_parameters(filename, thetaL, T, h, r):
    with open(filename, 'w') as f:
        f.write(", ".join(map(str, thetaL)) + "\n")
        f.write(", ".join(map(str, T)) + "\n")
        f.write(", ".join(map(str, h)) + "\n")
        f.write(", ".join(map(str, r)) + "\n")


''' Import parameters used for previously-calculated parameter-space analysis '''
def import_parameters(filename):
    with open(filename, 'w') as f:
        p = f.readlines()
        thetaL = [ n.split(',') for n in p[0] ]
        T = [ n.split(', ') for n in p[1] ]
        h = [ n.split(',') for n in p[2] ]
        r = [ n.split(',') for n in p[3] ]

    print("Imported the following parameters:")
    print("thetaL \t ({0}:{1}), length {2}".format(thetaL[0],thetaL[-1],len(thetaL)))
    print("T \t ({0}:{1}), length {2}".format(T[0],T[-1],len(T)))
    print("h \t ({0}:{1}), length {2}".format(h[0],h[-1],len(h)))
    print("r \t ({0}:{1}), length {2}".format(r[0],r[-1],len(r)))

    return thetaL, T, h, r


''' Create vtk file of unstructured grid type from csv array '''
def csv2vtk(filename):
    data = np.genfromtxt(filename,delimiter=", ",skip_header=1)
    x, y, value = data[:,0], data[:,1], data[:,2]
    nPoints = len(x)
    basename = os.path.splitext(filename)[0]
    with open(basename + ".vtk", "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(basename + "\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS {0} float\n".format(nPoints))
        for i in range(nPoints):
            f.write(x[i] + " " + y[i] + " 0\n")
        f.write("CELLS {0} {1}\n".format(nPoints, 2*nPoints))
        for i in range(nPoints):
            f.write("1 %d\n" % i)
        f.write("CELL_TYPES {0}\n".format(nPoints))
        for i in range(nPoints):
            f.write("1\n")
        f.write("POINT_DATA {0}\n".format(nPoints))
        f.write("SCALARS point_scalars float\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(0, nPoints):
            f.write(value[i] + "\n")
    return()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    
    plt.close('all')
    
    #datestr = datetime.today().strftime('%Y%m%d')
    datestr = '20200422'
    # Filenames
    parameter_file = os.path.join(cwd, '{0}_parameters.csv'.format(datestr))
    minima_file = os.path.join(cwd, '{0}_minima.npy'.format(datestr))
    phases_file = os.path.join(cwd, '{0}_phases.npy'.format(datestr))
    thetaT_file = os.path.join(cwd, '{0}_thetaT.npy'.format(datestr))
    theta0_file = os.path.join(cwd, '{0}_theta0.npy'.format(datestr))
    boundaries_file = os.path.join(cwd, '{0}_boundaries.csv'.format(datestr))
    boundaryVals_file = os.path.join(cwd, '{0}_boundaryVals.csv'.format(datestr))

    
    self.thetaL_range = np.radians(np.arange(-30.0, 30.1, 0.1))
    T_range = np.arange(25.0, 130.0, 1)    
    h_range = np.arange(0.0005, 0.00301, 0.000025)
    r_range = np.array([0.50])
    
    export_parameters(parameter_file, self.thetaL_range, T_range, h_range, r_range)
    
    T_plane = 75.0

    # Range for low-resolution tests
#    thetaL_range = np.radians(np.arange(-30.0, 30.1, 1))
#    T_range = np.arange(25.0, 130.0, 25)
#    h_range = np.arange(0.0005, 0.00301, 0.0001)
#    r_range = np.arange(0.0, 1.0, 0.25)
    
    # Range for high-resolution final plots for plot_isotherms
    #thetaL_range = np.radians(np.arange(-30.0, 30.1, 0.05))
    #T_range = np.arange(25.0, 130.0, 25)
    #h_range = np.arange(0.0005, 0.00301, 0.000005)
    #r_range = np.arange(0.0, 1.0, 0.25)

    
    # Analyze parameter space, or load previously generated results
    try:
        print("Try loading previous parameter-space analysis...")
        #thetaL_range, T_range, h_range, r_range = import_parameters(parameter_file)
        minima = np.load(minima_file)
        phases = np.load(phases_file)
        thetaT = np.load(thetaT_file)
        theta0 = np.load(theta0_file)
        print("\tLoaded parameters, minima, phases, theta_T, and theta_0")
    except IOError:
        print("\tRunning new parameter-space analysis...")
        t_start = process_time()
        minima, phases, thetaT, theta0 = analyze_parameter_space(T_range, h_range, r_range, thetaL_range)
        print("\tSaving parameters, minima, phases, theta_T, and theta_0")
        np.save(minima_file, minima, allow_pickle=False)
        np.save(phases_file, phases, allow_pickle=False)
        np.save(thetaT_file, thetaT, allow_pickle=False)
        np.save(theta0_file, theta0, allow_pickle=False)
        export_parameters(thetaL_range, T_range, h_range, r_range)
        t_stop = process_time()
        print('\n Total runtime: {0:0.2f} hours'.format((t_stop - t_start)/3600.0))

    # Find boundaries, or load previously generated results   
    try:
        print("Try loading previous boundary-finding analysis...")
        boundaries = np.genfromtxt(boundaries_file,delimiter=", ",dtype='int64',skip_header=1)
        boundaryVals = np.genfromtxt(boundaryVals_file,delimiter=", ",dtype='float64',skip_header=1)
        print("\tLoaded boundaries and boundaryVals")
    except IOError:
        print("\tFinding phase boundaries...")
        boundaries, boundaryVals = find_phase_boundaries(T_range, h_range, r_range, thetaL_range,
                                                         minima, phases, thetaT, theta0)
        print("\tSaving diffs, boundaries, and boundaryVals...")        
        export_to_csv(boundaries_file, boundaries, fmt='%d', headerList=["T index", "h index", "thetaL index"])
        np.savetxt(boundaryVals_file, boundaryVals, delimiter=', ', fmt='%0.1f', header='Boundary values')

    plot_phase_boundaries(boundaries, boundaryVals,
                          T_range, h_range, 0, thetaL_range,
                          minima, phases, thetaT, theta0,
                          T_plane=T_plane)

    #plot_isotherms(T_range, h_range, r_range, thetaL_range, minima, phases, thetaT, theta0, T_plane)
    
    
    