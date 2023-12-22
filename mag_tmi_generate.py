"""
Forward Simulation of Gradiometry Data on a Tree Mesh
=====================================================

Here we use the module *SimPEG.potential_fields.gravity* to predict gravity
gradiometry data for a synthetic density contrast model. The simulation is
carried out on a tree mesh. For this tutorial, we focus on the following:

    - How to define the survey when we want multiple field components
    - How to predict gravity gradiometry data for a density contrast model
    - How to construct tree meshes based on topography and survey geometry
    - The units of the density contrast model and resulting data


"""

#########################################################################
# Import Modules
# --------------
#
#import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from SimPEG.utils import plot2Ddata, model_builder
from SimPEG.potential_fields import gravity,magnetics
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
)


from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz
from SimPEG.utils import plot2Ddata, model_builder
from SimPEG import maps
from SimPEG.potential_fields import gravity

# sphinx_gallery_thumbnail_number = 2

#############################################
# Defining Topography
# -------------------
#
# Surface topography is defined as an (N, 3) numpy array. We create it here but
# the topography could also be loaded from a file.
#

[x_topo, y_topo] = np.meshgrid(np.linspace(-2000, 2000, 81), np.linspace(-2000, 2000, 81))
z_topo = -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2)
x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
xyz_topo = np.c_[x_topo, y_topo, z_topo]


#############################################
# Defining the Survey
# -------------------
#
# Here, we define survey that will be used for the forward simulation. Gravity
# surveys are simple to create. The user only needs an (N, 3) array to define
# the xyz locations of the observation locations, and a list of field components
# which are to be measured.
#

# Define the observation locations as an (N, 3) numpy array or load them
x = np.linspace(-400.0, 400.00)
y = np.linspace(-400.0, 400.0, 80)
x, y = np.meshgrid(x, y)
x, y = mkvc(x.T), mkvc(y.T)
fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
z = fun_interp(np.c_[x, y]) + 5
receiver_locations = np.c_[x, y, z]

# Define the component(s) of the field we want to simulate as strings within
# a list. Here we measure the x, y and z components of the gravity anomaly at
# each observation location.
components = ["tmi"]

# Use the observation locations and components to define the receivers. To
# simulate data, the receivers must be defined as a list.
receiver_list = magnetics.receivers.Point(receiver_locations, components=components)

receiver_list = [receiver_list]
# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
inclination = 90
declination = 0
strength = 50000
inducing_field = (strength, inclination, declination)

# Defining the source field.
source_field = magnetics.sources.SourceField(receiver_list=receiver_list, parameters=inducing_field)
# Defining the survey
survey = magnetics.survey.Survey(source_field)

##########################################################
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

# Refine box based on region of interest
xp, yp, zp = np.meshgrid([-400.0, 400.0], [-400.0, 400.0], [-500.0, 0.0])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

mesh = refine_tree_xyz(mesh, xyz, octree_levels=[4, 4], method="box", finalize=False)

mesh.finalize()
print('Mesh generated')
#######################################################
# Density Contrast Model and Mapping on OcTree Mesh
# -------------------------------------------------
#
# Here, we create the density contrast model that will be used to simulate gravity
# gradiometry data and the mapping from the model to the mesh. The model
# consists of a less dense block and a more dense sphere.
#

# Define susc  values for each unit
background_density = 1e-4
block_density = 0.1
sphere_density = 2.0

# Find the indecies for the active mesh cells (e.g. cells below surface)
ind_active = active_from_xyz(mesh, xyz_topo)

# Define mapping from model to active cells. The model consists of a value for
# each cell below the Earth's surface.
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model will be value of active cells

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
#Ry = np.zeros((3, 3))
#phiy=-30
#Ry[1, 1] = 1
#Ry[0, 0] = np.cos(phiy)
#Ry[2, 0] = -np.sin(phiy)
#Ry[2, 2] = np.cos(phiy)
#Ry[0, 2] = np.sin(phiy)

#model = np.*Ry,model


# You can also use SimPEG utilities to add structures to the model more concisely
#ind_sphere = model_builder.getIndicesSphere(np.r_[200.0, 0.0, -200.0], 100.0, mesh.gridCC)
#ind_sphere = ind_sphere[ind_active]
#model[ind_sphere] = sphere_density

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

#plt.show()

##############################################################
# Simulation: Gravity Gradiometry Data on an OcTree Mesh
# ------------------------------------------------------
#
# Here we demonstrate how to predict gravity anomaly data using the integral
# formulation.
#

# Define the forward simulation. By setting the 'store_sensitivities' keyword
# argument to "forward_only", we simulate the data without storing the sensitivities
simulation = magnetics.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    model_type="scalar",
    chiMap=model_map,
    ind_active=ind_active,
    store_sensitivities="forward_only",
)
#simulation = magnetics.simulation.Simulation3DIntegral(
#    survey=survey,
#    mesh=mesh,
#    rhoMap=model_map,
#    ind_active=ind_active,
#    store_sensitivities="forward_only"
#)

# Convert the inclination declination to vector in Cartesian


# Compute predicted data for some model
dpred = simulation.dpred(model)
print('Simulation generated')
std = 0.05 * np.abs(dpred)

dir_path = r"C:\Users\parth\OneDrive_UBC\Masters\2023-2024\EOSC_555B\Project\magnetics\generated_80m"

data_obj = data.Data(survey, dobs=dpred, standard_deviation=std)
fname = dir_path + "\mag_data.obs"
utils.io_utils.write_grav3d_ubc(fname, data_obj)
fname = dir_path + "\mag_topo_xyz.txt"
np.savetxt(fname,xyz_topo, fmt="%.4e")
#fname=dir_path + "\grav_mesh.msh"
#TreeMesh.write_UBC(fname,mesh)
print('Gravity data generated')