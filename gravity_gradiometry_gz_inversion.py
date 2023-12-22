
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile
import torch
import warnings
warnings.filterwarnings("ignore")

from discretize import TreeMesh,TensorMesh
from discretize.utils import active_from_xyz,refine_tree_xyz
from SimPEG.utils import plot2Ddata, model_builder,io_utils
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
mkvc
)

dir_path = r"C:\Users\parth\OneDrive_UBC\Masters\2023-2024\EOSC_555B\Project\gravity\generated_80m"

# files to work with
topo_filename = os.path.join(dir_path, "grav_topo_xyz.txt")
data_filename = os.path.join(dir_path, "grav_data_block.obs")

# Load topography
xyz_topo = np.loadtxt(str(topo_filename))

# Load field data
dobs = np.loadtxt(str(data_filename))

# Define receiver locations and observed data
receiver_locations = dobs[:, 0:3]
receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
receiver_list = [receiver_list]
dobs = dobs[:, -1]
# Define the source field
source_field = gravity.sources.SourceField(receiver_list=receiver_list)

# Define the survey
survey = gravity.survey.Survey(source_field)
#uncertainties
uncertainties = 0.05 * np.ones(np.shape(dobs))
maximum_anomaly = np.max(np.abs(dobs))

uncertainties = 0.01 * maximum_anomaly * np.ones(np.shape(dobs))

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)
def mesh():
    # Defining an OcTree Mesh
    # -----------------------
    #
    # Here, we create the OcTree mesh that will be used in the forward simulation.
    #

    dx = 25  # minimum cell width (base mesh cell width) in x
    dy = 25  # minimum cell width (base mesh cell width) in y
    dz = 25  # minimum cell width (base mesh cell width) in z

    x_length = 2000.0  # domain width in x
    y_length = 2000.0  # domain width in y
    z_length = 1000.0  # domain width in z

    # Compute number of base mesh cells required in x and y
    nbcx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
    nbcy = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))
    nbcz = 2 ** int(np.round(np.log(z_length / dz) / np.log(2.0)))

    # Define the base mesh
    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    hz = [(dz, nbcz)]
    mesh = TreeMesh([hx, hy, hz], x0="CCN")

    # Refine based on surface topography
    mesh = refine_tree_xyz(
        mesh, xyz_topo, octree_levels=[2, 2], method="surface", finalize=False
    )
    xp, yp, zp = np.meshgrid([-400.0, 400.0], [-400.0, 400.0], [-500.0, 0.0])
    xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

    mesh = refine_tree_xyz(mesh, xyz, octree_levels=[4, 4], method="box", finalize=False)

    mesh.finalize()
    return mesh,dx

mesh,mesh_spacing = mesh()
# Define density contrast values for each unit in g/cc
background_density = 0.0
block_density = 0.3
#sphere_density = 0.3

# Define mapping from model to active cells
ind_active = active_from_xyz(mesh, xyz_topo)
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)
# Define model. Models in SimPEG are vector arrays.
model = background_density * np.ones(nC)

# You could find the indicies of specific cells within the model and change their
# value to add structures.
ind_block = (
    (mesh.gridCC[ind_active, 0] > -200.0)
    & (mesh.gridCC[ind_active, 0] < 200.0)
    & (mesh.gridCC[ind_active, 1] > -100.0)
    & (mesh.gridCC[ind_active, 1] < 100.0)
    & (mesh.gridCC[ind_active, 2] > -300.0)
    & (mesh.gridCC[ind_active, 2] < -150.0)
)
model[ind_block] = block_density

# You can also use SimPEG utilities to add structures to the model more concisely
#ind_sphere = model_builder.getIndicesSphere(np.r_[200.0, 0.0, -200.0], 100.0, mesh.gridCC)
#ind_sphere = ind_sphere[ind_active]
#model[ind_sphere] = [sphere_density]

###Pytroch Transform#################
def torch_transform(m, params, xyz):
    xyz = xyz
    n_shape = params[1]
    mesh_spacing = params[0]
    p_0 = 0
    H_sum,p_sum = 0,0
    num_params = 10
    X = torch.tensor(xyz[:, 0])
    Y = torch.tensor(xyz[:, 1])
    Z = torch.tensor(xyz[:, 2])

    xyz = torch.vstack((X, Y, Z))


    for i in range(n_shape):
        hx, hy, hz, phix, phiy, phiz, x_0, y_0, z_0, p_1 = m[i * num_params:(i + 1) * num_params]

        xyz_0 = torch.vstack((x_0, y_0, z_0))

        Rx = torch.zeros((3, 3), dtype=torch.float64)

        Rx[0, 0] = 1
        Rx[1, 1] = torch.cos(phix)
        Rx[1, 2] = -torch.sin(phix)
        Rx[2, 2] = torch.cos(phix)
        Rx[2, 1] = torch.sin(phix)

        Ry = torch.zeros_like(Rx)
        Ry[1, 1] = 1
        Ry[0, 0] = torch.cos(phiy)
        Ry[2, 0] = -torch.sin(phiy)
        Ry[2, 2] = torch.cos(phiy)
        Ry[0, 2] = torch.sin(phiy)

        Rz = torch.zeros_like(Rx)
        Rz[2, 2] = 1
        Rz[0, 0] = torch.cos(phiz)
        Rz[0, 1] = -torch.sin(phiz)
        Rz[1, 1] = torch.cos(phiz)
        Rz[1, 0] = torch.sin(phiz)

        M = Rx @ Ry @ Rz

        xyz_m_xyz_0 = xyz - xyz_0
        tau = xyz_m_xyz_0.T @ M

        lx = mesh_spacing
        ly = mesh_spacing
        lz = mesh_spacing

        sigma_x = 1 / ((1 + torch.exp(-(tau[:,0] + hx) / lx))) - 1 / ((1 + torch.exp(-(tau[:,0] - hx) / lx)))
        sigma_y = 1 / ((1 + torch.exp(-(tau[:,1] + hy) / ly))) - 1 / ((1 + torch.exp(-(tau[:,1] - hy) / ly)))
        sigma_z = 1 / ((1 + torch.exp(-(tau[:,2] + hz) / lz))) - 1 / ((1 + torch.exp(-(tau[:,2] - hz) / lz)))

        H = sigma_x * sigma_y * sigma_z

        p = (H) * (p_1 - p_0)

       # if i ==0:
       #     num = p * torch.exp(p)
       #     den = torch.exp(p)
       # else:
       #     num = num+(p * torch.exp(p))
       #     den = den+ (torch.exp(p))
    #p = num / den
        p_sum = p_sum + p
        H_sum = H_sum + H


    w = torch.minimum(torch.tensor(1.0, dtype=H_sum.dtype), 1 / H_sum)
    p = p_0 +  w*p_sum
    #plt.plot(p,'.')
    #plt.show()
    return p
#             hx,hy, hz,phix,phiy,phiz,x_0,y_0,z_0,p_1
m = np.array([200,100,100,0.0,0.0,0.0,0.0,00.0,-200.0,0.2])

num_params = len(m)
params = [mesh_spacing,1] #hx,n_ellipse

torch_map = maps.PytorchMapping(mesh=mesh,nP=num_params,indicesActive=ind_active,forward_transform=torch_transform,params=params)
active_map = maps.InjectActiveCells(mesh,ind_active, 0)
identity_map = maps.IdentityMap(nP=nC)
model_map = torch_map


def plot_true():
    # Plot Density Contrast Model
    fig = plt.figure(figsize=(9, 4))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

    ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
    mesh.plot_slice(
        plotting_map * model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.h[1].size / 2),
        grid=True,
        clim=(np.min(model), np.max(model)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")

    ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
    )
    cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)

    plt.show()

def plot_initial():
    # Plot initial model
    fig = plt.figure(figsize=(9, 4))

    model_plot = maps.InjectActiveCells(mesh, ind_active, 0) * model_map * m
    ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])
    mesh.plot_slice(
        model_plot,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2),
        grid=True,
        clim=(np.min(model_plot), np.max(model_plot)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(model), vmax=np.max(model))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
    )
    cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)

    plt.show()
#plot initial and true model
plot_initial()
plot_true()

#Set simulation
simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey, mesh=mesh, rhoMap=model_map,
    ind_active=ind_active
)


##Misfit and Regularization and optimization
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
reg = regularization.Tikhonov(TensorMesh([np.ones(num_params)]))

lower_bounds = np.r_[0,0,0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-0.8]
upper_bounds = np.r_[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0,0.8]


opt = optimization.ProjectedGNCG(
    maxIter=20,lower=lower_bounds,upper=upper_bounds, maxIterLS=20, maxIterCG=100, tolCG=.0001,
    tolF=1e-2, tolX = 1e-2, tolG=1e-2,
)
##Directives
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
#target_misfit = directives.TargetMisfit(chifact=1)
sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
directives_list = [sensitivity_weights, update_jacobi]

#Inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0)
inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)
recovered_model = inv.run(m)

#plot results
def plot_recovered():
    # Plot Recovered Model
    fig = plt.figure(figsize=(9, 4))
    # plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    plotting_map = active_map * model_map * recovered_model

    ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])
    mesh.plot_slice(
        plotting_map,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2),
        grid=True,
        clim=(np.min(plotting_map), np.max(plotting_map)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")

    ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    norm = mpl.colors.Normalize(vmin=np.min(plotting_map), vmax=np.max(plotting_map))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
    )
    cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)

    plt.show()

def plot_recovered_grids():
    dpred = inv_prob.dpred

    # Observed data | Predicted data | Normalized data misfit
    data_array = np.c_[dobs, dpred, (dobs - dpred) / uncertainties]

    fig = plt.figure(figsize=(17, 4))
    plot_title = ["Observed", "Predicted", "Normalized Misfit"]
    plot_units = ["mgal", "mgal", ""]

    ax1 = 3 * [None]
    ax2 = 3 * [None]
    norm = 3 * [None]
    cbar = 3 * [None]
    cplot = 3 * [None]
    v_lim = [np.max(np.abs(dobs)), np.max(np.abs(dobs)), np.max(np.abs(data_array[:, 2]))]

    for ii in range(0, 3):
        ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.23, 0.84])
        cplot[ii] = plot2Ddata(
            receiver_list[0].locations,
            data_array[:, ii],
            ax=ax1[ii],
            ncontour=30,
            clim=(-v_lim[ii], v_lim[ii]),
            contourOpts={"cmap": "bwr"},
        )
        ax1[ii].set_title(plot_title[ii])
        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = fig.add_axes([0.33 * ii + 0.25, 0.11, 0.01, 0.85])
        norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
        cbar[ii] = mpl.colorbar.ColorbarBase(
            ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
        )
        cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)

plot_recovered()
plot_recovered_grids()
print("Run complete")