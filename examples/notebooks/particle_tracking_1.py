# # Particle tracking example 1: backwards tracking in a steady flow field
#
# This example demonstrates backward particle tracking in a steady-state
# flow field, reproducing [Example 1](https://modflow6-examples.readthedocs.io/en/develop/_notebooks/ex-prt-mp7-p01.html)
# from the MODPATH 7 user guide. The scenario involves determining the
# capture area for a pumping well.
#
# A MODFLOW 6 GWF model is run to produce a flow solution. Then both a
# MODPATH 7 model and an equivalent MODFLOW 6 PRT model are run side by
# side and their results compared. This is an opportunity to compare the
# features of MODPATH 7 and MODFLOW 6 PRT, and to demonstrate how to
# migrate an existing MODPATH 7 simulation to PRT.

# ## Imports

import warnings
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely import MultiPoint, convex_hull

# +
%matplotlib inline

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
# -

# ## Setup

example_name = "ex1"
base_ws = Path("temp") / example_name
base_ws.mkdir(exist_ok=True, parents=True)

# ## Flow model
#
# Define a flow model which will be used by both PRT and MP7.

gwf_name = f"{example_name}-gwf"
gwf_ws = base_ws / "gwf"
gwf_ws.mkdir(exist_ok=True, parents=True)

# Define model units.

length_units = "feet"
time_units = "days"

# Define model parameters.

nper = 1
nlay = 3
nrow = 21
ncol = 20
Lx = 10000.0
Ly = 10500.0
delr = Lx / ncol
delc = Ly / nrow
top = 400.0
botm = [220.0, 200.0, 0.0]
porosity = 0.1
rch = 0.005
kh = [50.0, 0.01, 200.0]
kv = [10.0, 0.01, 20.0]
wel_q = -150000.0
riv_h = 320.0
riv_z = 317.0
riv_c = 1.0e5

# Define time discretization.

nstp = 1
perlen = 1000.0
tsmult = 1.0
tdis_rc = [(perlen, nstp, tsmult)]

# Construct the GWF simulation.

gwf_sim = flopy.mf6.MFSimulation(
    sim_name=gwf_name, exe_name="mf6", version="mf6", sim_ws=gwf_ws
)

# Create the time discretization package.

tdis = flopy.mf6.ModflowTdis(
    gwf_sim,
    pname="tdis",
    time_units="DAYS",
    perioddata=tdis_rc,
    nper=len(tdis_rc),
)

# Create the flow model.

gwf = flopy.mf6.ModflowGwf(
    gwf_sim, modelname=gwf_name, model_nam_file=f"{gwf_name}.nam"
)
gwf.name_file.save_flows = True

# Create the discretization package.

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    length_units=length_units,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)

# Create the initial conditions package.

ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=top)

# Create the node property flow package.

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=[1, 0, 0],
    k=kh,
    k33=kv,
    save_saturation=True,
    save_specific_discharge=True,
)

# Define boundary conditions: a well, a river, and recharge.

# +
# Well
wel_loc = (2, 10, 9)
wd = [(wel_loc, wel_q)]

# River
riv_iface = 6
riv_iflowface = -1
rd = []
for i in range(nrow):
    rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z, riv_iface, riv_iflowface])

# Recharge
rch_iface = 6
rch_iflowface = -1
# -

# Compute node numbers for the well and river cells, used for zone assignment.

# +
nodes = {}
k, i, j = wel_loc
nodes["well"] = [ncol * (nrow * k + i) + j]
nodes["river"] = []
for rivspec in rd:
    k, i, j = rivspec[0]
    nodes["river"].append(ncol * (nrow * k + i) + j)
# -

# Create boundary condition packages.

rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
    gwf,
    recharge=rch,
    auxiliary=["iface", "iflowface"],
    aux=[rch_iface, rch_iflowface],
)

wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(
    gwf, maxbound=1, stress_period_data={0: wd}
)

riv = flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(
    gwf, auxiliary=["iface", "iflowface"], stress_period_data={0: rd}
)

# Create the output control package.

# +
headfile_name = f"{gwf_name}.hds"
budgetfile_name = f"{gwf_name}.cbb"

oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
    gwf,
    pname="oc",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    head_filerecord=[headfile_name],
    budget_filerecord=[budgetfile_name],
)
# -

# Create the solver package.

ims = flopy.mf6.ModflowIms(
    gwf_sim,
    pname="ims",
    complexity="SIMPLE",
    outer_dvclose=1e-6,
    inner_dvclose=1e-6,
    rcloserecord=1e-6,
)

# Write and run the flow model.

gwf_sim.write_simulation(silent=False)
gwf_sim.run_simulation(silent=False)

# Load heads.

hds = gwf.output.head().get_data()

# Plot heads in layer 3. Define a helper function we will reuse later.

def plot_heads(ax, gwf, heads, colorbar=True):
    mm = flopy.plot.PlotMapView(gwf, ax=ax, layer=2)
    mm.plot_grid(alpha=0.25)
    mm.plot_bc("WEL", plotAll=True, color="red")
    mm.plot_bc("RIV", plotAll=True, color="blue")
    pc = mm.plot_array(heads, edgecolor="black", alpha=0.25)
    if colorbar:
        cb = plt.colorbar(pc, shrink=0.25, pad=0.1)
        cb.ax.set_xlabel(r"Head ($ft$)")
    return mm


with flopy.plot.styles.USGSPlot():
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.set_aspect("equal")
    ax.legend(
        handles=[
            Patch(color="red", label="Well"),
            Patch(color="blue", label="River"),
        ],
    )
    fig.tight_layout()
    flopy.plot.styles.heading(ax, heading="Head, layer 3")
    plot_heads(ax, gwf, hds[2, :, :])

# ## Tracking models
#
# Now we define a MODPATH 7 particle tracking simulation and an equivalent
# PRT simulation.

# ### MP7 model

mp7_name = f"{example_name}-mp7"
mp7_ws = base_ws / "mp7"
mp7_ws.mkdir(exist_ok=True, parents=True)

mp7_pathline_file_path = mp7_ws / f"{mp7_name}.mppth"

# Create the MP7 model.

mp7 = flopy.modpath.Modpath7(
    modelname=mp7_name,
    flowmodel=gwf,
    exe_name="mp7",
    model_ws=mp7_ws,
    budgetfilename=budgetfile_name,
    headfilename=headfile_name,
)

# Create the basic input data package.

mp7_bas = flopy.modpath.Modpath7Bas(
    mp7, porosity=porosity, defaultiface={"RCH": 6, "EVT": 6}
)

# Define a particle release configuration. We release particles from the
# lateral faces of the well cell using MP7 particle input style 2,
# subdivision style 1.

mp7_face_data = flopy.modpath.FaceDataType(
    verticaldivisions1=5,
    verticaldivisions2=5,
    verticaldivisions3=5,
    verticaldivisions4=5,
    horizontaldivisions1=5,
    horizontaldivisions2=5,
    horizontaldivisions3=5,
    horizontaldivisions4=5,
    rowdivisions5=0,
    rowdivisions6=0,
    columndivisions5=0,
    columndivisions6=0,
)
mp7_particle_data = flopy.modpath.LRCParticleData(
    subdivisiondata=[mp7_face_data], lrcregions=[[[*wel_loc, *wel_loc]]]
)
mp7_pg1 = flopy.modpath.ParticleGroupLRCTemplate(
    particlegroupname="PG1",
    particledata=mp7_particle_data,
    filename=f"{mp7_name}pg1.sloc",
)

# Define a zone map. Zone 0 is active, zone 1 is the well, zone 2 is the
# river. Zones are used to identify where particles originate or terminate.

izone = np.zeros((nlay, nrow, ncol), dtype=int)
for l, r, c in gwf.modelgrid.get_lrc(nodes["well"]):
    izone[l, r, c] = 1
for l, r, c in gwf.modelgrid.get_lrc(nodes["river"]):
    izone[l, r, c] = 2

# Create the MP7 simulation.

mp7_sim = flopy.modpath.Modpath7Sim(
    mp7,
    simulationtype="combined",
    trackingdirection="backward",
    weaksinkoption="pass_through",
    weaksourceoption="pass_through",
    budgetoutputoption="summary",
    referencetime=[0, 0, 0.0],
    stoptimeoption="extend",
    timepointdata=[500, 1000.0],
    zonedataoption="on",
    zones=izone,
    particlegroups=[mp7_pg1],
)

# Write and run the MP7 simulation.

mp7.write_input()
mp7.run_model(silent=False)

# Load MP7 pathlines. FloPy returns a list of recarrays, one per pathline;
# we merge them into a single dataframe.

mp7_pathline_file = flopy.utils.PathlineFile(mp7_pathline_file_path)
mp7_pathlines = pd.concat(
    [pd.DataFrame(pdata) for pdata in mp7_pathline_file.get_alldata()]
)
mp7_pathlines

# Plot MP7 pathlines.

with flopy.plot.styles.USGSPlot():
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.set_aspect("equal")
    fig.tight_layout()
    flopy.plot.styles.heading(ax, heading="MODPATH 7 pathlines")
    mm = plot_heads(ax, gwf, hds[2, :, :])
    mm.plot_pathline(
        mp7_pathlines, layer="all", colors="black", alpha=0.5, linewidth=0.5
    )

# ### PRT model

prt_name = f"{example_name}-prt"
prt_ws = base_ws / "prt"
prt_ws.mkdir(exist_ok=True, parents=True)

# Create the PRT simulation.

prt_sim = flopy.mf6.MFSimulation(
    sim_name=prt_name, exe_name="mf6", version="mf6", sim_ws=prt_ws
)

# Create the temporal discretization.

tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    prt_sim,
    pname="tdis",
    time_units="DAYS",
    nper=nper,
    perioddata=[(perlen, nstp, tsmult)],
)

# Create the PRT model.

prt = flopy.mf6.ModflowPrt(
    prt_sim, modelname=prt_name, model_nam_file=f"{prt_name}.nam"
)

# Create the discretization package.

dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    prt,
    pname="dis",
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    length_units="FEET",
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)

# Create the model input package.

mip = flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity, izone=izone)

# Create the particle release package. FloPy's `to_prp()` utility converts
# the MP7 particle configuration to the format expected by PRT.

release_pts = list(mp7_particle_data.to_prp(gwf.modelgrid))
prp = flopy.mf6.ModflowPrtprp(
    prt,
    nreleasepts=len(release_pts),
    packagedata=release_pts,
    perioddata={0: ["FIRST"]},
    exit_solve_tolerance=1e-5,
    extend_tracking=True,
)

# Create the output control package.
#
# PRT can write pathline output to binary or CSV files. Below we enable
# both, but will only read the CSV output file.
#
# PRT tracking events (all enabled by default):
#
# - release: record at particle release
# - cell exit: record when a particle exits a cell
# - timestep end: record at end of each time step
# - termination: record when a particle terminates
# - weak sink: record when a particle exits a weak sink cell
# - user time: record at user-specified times
#
# Some exercises:
#
# Q: How to configure a PRT model like an MP7 endpoint simulation?
# A: Select release and termination events only.
#
# Q: How to configure a model like an MP7 timeseries simulation?
# A: Select time reporting and provide tracking times.

# +
budgetfile_prt = f"{prt_name}.cbb"
trackfile_prt = f"{prt_name}.trk"
trackcsvfile_prt = f"{prt_name}.trk.csv"
tracktimes = list(range(0, 72000, 1000))

oc = flopy.mf6.ModflowPrtoc(
    prt,
    pname="oc",
    budget_filerecord=[budgetfile_prt],
    track_filerecord=[trackfile_prt],
    trackcsv_filerecord=[trackcsvfile_prt],
    ntracktimes=len(tracktimes),
    tracktimes=[(t,) for t in tracktimes],
    saverecord=[("BUDGET", "ALL")],
)
# -

# Create the flow model interface. PRT does not natively support backward
# tracking, so we reverse the GWF head and budget files with FloPy first.

# +
head_file = flopy.utils.HeadFile(gwf_ws / headfile_name, tdis=gwf_sim.tdis)
budget_file = flopy.utils.CellBudgetFile(
    gwf_ws / budgetfile_name, precision="double", tdis=gwf_sim.tdis
)

headfile_bkwd_name = f"{headfile_name}_bkwd"
budgetfile_bkwd_name = f"{budgetfile_name}_bkwd"

head_file.reverse(prt_ws / headfile_bkwd_name)
budget_file.reverse(prt_ws / budgetfile_bkwd_name)

fmi = flopy.mf6.ModflowPrtfmi(
    prt,
    packagedata=[
        ("GWFHEAD", headfile_bkwd_name),
        ("GWFBUDGET", budgetfile_bkwd_name),
    ],
)
# -

# Create an explicit model solution. PRT uses its own solution procedure,
# not the iterative matrix solvers used by GWF and GWT.

ems = flopy.mf6.ModflowEms(
    prt_sim,
    pname="ems",
    filename=f"{prt_name}.ems",
)
prt_sim.register_solution_package(ems, [prt.name])

# Write and run the PRT simulation.

prt_sim.write_simulation()
prt_sim.run_simulation(silent=False)

# Load PRT pathlines.

prt_pathlines = pd.read_csv(prt_ws / trackcsvfile_prt)
prt_pathlines

# Compare PRT and MP7 pathlines in 3D with PyVista.

try:
    import pyvista as pv
    from flopy.export.vtk import Vtk

    vert_exag = 10

    vtk = Vtk(model=gwf, binary=False, vertical_exageration=vert_exag, smooth=False)
    vtk.add_model(gwf)
    vtk.add_pathline_points(mp7_pathlines.to_records(index=False))
    gwf_mesh, mp7_mesh = vtk.to_pyvista()

    # Vtk only supports one pathline set at a time, so rebuild for PRT.
    vtk = Vtk(model=gwf, binary=False, vertical_exageration=vert_exag, smooth=False)
    vtk.add_model(gwf)
    vtk.add_pathline_points(prt_pathlines.to_records(index=False))
    _, prt_mesh = vtk.to_pyvista()

    def get_nn(k, i, j):
        return k * nrow * ncol + i * ncol + j

    riv_mesh = pv.Box(
        bounds=[
            gwf.modelgrid.extent[1] - delc,
            gwf.modelgrid.extent[1],
            gwf.modelgrid.extent[2],
            gwf.modelgrid.extent[3],
            botm[0] * vert_exag,
            hds[(0, 0, ncol - 1)] * vert_exag,
        ]
    )
    well_cellid = get_nn(0, *wel_loc[1:])
    well_points = gwf.modelgrid.verts[gwf.modelgrid.iverts[well_cellid]]
    well_xs, well_ys = list(zip(*well_points))
    wel_mesh = pv.Box(
        bounds=[
            min(well_xs),
            max(well_xs),
            min(well_ys),
            max(well_ys),
            botm[-1] * vert_exag,
            botm[-2] * vert_exag,
        ]
    )
    bed_mesh = pv.Box(
        bounds=[
            gwf.modelgrid.extent[0],
            gwf.modelgrid.extent[1],
            gwf.modelgrid.extent[2],
            gwf.modelgrid.extent[3],
            botm[1] * vert_exag,
            botm[0] * vert_exag,
        ]
    )

    pv.set_jupyter_backend("static")
    p = pv.Plotter(window_size=[500, 500], notebook=True)
    p.enable_anti_aliasing()
    p.add_mesh(gwf_mesh, opacity=0.025, style="wireframe")
    p.add_mesh(mp7_mesh, point_size=5, line_width=2.5, smooth_shading=True, color="white")
    p.add_mesh(prt_mesh, point_size=5, line_width=2.5, smooth_shading=True, color="gray")
    p.add_mesh(riv_mesh, color="teal", opacity=0.2)
    p.add_mesh(wel_mesh, color="red", opacity=0.3)
    p.add_mesh(bed_mesh, color="tan", opacity=0.1)
    p.show()
except ImportError:
    print("PyVista not available.")

# ## Capture area
#
# Filter pathlines to termination points (ireason == 3) and compute the
# convex hull to estimate the well's capture area.

termination_pts = prt_pathlines[prt_pathlines.ireason == 3].set_index("irpt")[
    ["x", "y", "z"]
]
termination_pts

# Build a multi-point geometry and compute the convex hull.

term_pts = MultiPoint(termination_pts.to_numpy())
chull = convex_hull(term_pts)
chull_area_mi = chull.area / (5280 * 5280)
print(f"Capture area: {chull.area:9.3f} sq {length_units}, {chull_area_mi:9.3f} sq mi")
chull

# Plot pathlines with the capture area overlaid.

with flopy.plot.styles.USGSPlot():
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.set_aspect("equal")
    fig.tight_layout()
    flopy.plot.styles.heading(ax, heading="Pathlines with capture area")
    mm = plot_heads(ax, gwf, hds[2, :, :])
    mm.plot_pathline(
        prt_pathlines, layer="all", colors="black", alpha=0.5, linewidth=0.5
    )
    chull_xs, chull_ys, _ = map(list, zip(*list(chull.exterior.coords)))
    ax.fill(chull_xs, chull_ys, alpha=0.3)
    ax.annotate(f"{chull_area_mi:.3f} sq mi", (1000, 3000), color="red")

# ## Exercises
#
# 1. Determine a travel time distribution and plot endpoints colored by travel time.
# 2. Vary the particle configuration and determine the effect on the capture area.
# 3. Refine the grid around the well and determine the effect on the capture area.

# ### Exercise 1 solution: travel time distribution

termination_pts = prt_pathlines[prt_pathlines.ireason == 3].set_index("irpt")
termination_pts

termination_pts.t.describe()

termination_pts.t.plot(kind="hist")

with flopy.plot.styles.USGSPlot():
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.set_aspect("equal")
    fig.tight_layout()
    flopy.plot.styles.heading(ax, heading="Endpoints colored by travel time")
    mm = plot_heads(ax, gwf, hds[2, :, :], colorbar=False)
    mm.plot_pathline(
        prt_pathlines, layer="all", colors="black", alpha=0.5, linewidth=0.5
    )
    pc = ax.scatter(termination_pts.x, termination_pts.y, c=termination_pts.t)
    cb = plt.colorbar(pc, shrink=0.25, pad=0.1)
    cb.ax.set_xlabel(r"Travel time ($d$)")
