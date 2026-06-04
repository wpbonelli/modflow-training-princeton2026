# %% [markdown]
# # Thermal profile along a particle path
#
# This example demonstrates the combined use of GWF, GWE, and PRT.
# A steady flow field is established over a Voronoi grid, followed by
# a steady-state temperature distribution. Two particles are then tracked
# through the domain and their temperatures interpolated along the path.
#
# The flow system uses a 2000 × 1000 m domain with:
# - constant-head boundaries on the left, right, and bottom edges carrying
#   different temperatures
# - three wells (two pumping, one injection) creating local flow complexity
#
# The Voronoi grid is loaded from a pre-generated binary reference file
# (`ex-gwe-prt-gwf.disv.grb`) to keep the notebook focused on the
# GWF → GWE → PRT coupling rather than grid generation.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import flopy

# %% [markdown]
# Define workspaces.

# %%
example_name = "prt-a"
gwf_name = f"{example_name}-gwf"
gwe_name = f"{example_name}-gwe"
prt_name = f"{example_name}-prt"

base_ws = Path("models") / example_name
gwf_ws = base_ws / "gwf"
gwe_ws = base_ws / "gwe"
prt_ws = base_ws / "prt"
gwf_ws.mkdir(exist_ok=True, parents=True)
gwe_ws.mkdir(exist_ok=True, parents=True)
prt_ws.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# Define model parameters.

# %%
length_units = "meters"
time_units = "days"

# domain
xmin, xmax = 0.0, 2000.0
ymin, ymax = 0.0, 1000.0
top = 1.0
botm = [0.0]
nlay = 1

# aquifer
porosity = 0.1

# thermal properties
strt_temp = 10.0        # initial temperature (°C)
scheme = "TVD"
alh = 0.0               # longitudinal mechanical dispersivity (m)
ath1 = 0.0              # transverse mechanical dispersivity (m)
ktw = 0.56 * 86400      # thermal conductivity of water, W/(m·°C) → J/(m·day·°C)
kts = 2.5 * 86400       # thermal conductivity of solids, W/(m·°C) → J/(m·day·°C)
rhow = 1000.0           # density of water (kg/m³)
cpw = 4180.0            # heat capacity of water (J/(kg·°C))
rhos = 2650.0           # density of dry solid (kg/m³)
cps = 900.0             # heat capacity of dry solid (J/(kg·°C))
lhv = 2500.0            # latent heat of vaporization (J/kg)

# solver
nouter = 1000
ninner = 200
hclose = 1e-6
rclose = 1e-6
relax = 1.0

# particle release points: a line at x=20 m, spaced every 20 m in y
rpts = [[20, i, 0.5] for i in range(1, 999, 20)]

# %% [markdown]
# ## Grid
#
# Load the Voronoi grid from a pre-generated DISV binary reference file.
# This avoids a runtime dependency on the Triangle mesh generator and keeps
# the focus on the GWF/GWE/PRT coupling.
#
# The grid covers a 2000 × 1000 m domain with ~1692 Voronoi cells.

# %%
from flopy.mf6.utils import MfGrdFile
from flopy.utils import GridIntersect
from shapely.geometry import LineString, Point

grb = MfGrdFile(Path("../data/prt_a/ex-gwe-prt-gwf.disv.grb"))
mg = grb.modelgrid

ncpl = mg.ncpl
nvert = len(mg.verts)
vertices = [
    [iv, float(mg.verts[iv][0]), float(mg.verts[iv][1])]
    for iv in range(nvert)
]
cell2d = [
    [icpl, float(mg.xcellcenters[icpl]), float(mg.ycellcenters[icpl]),
     len(mg.iverts[icpl]), *mg.iverts[icpl]]
    for icpl in range(ncpl)
]

gi = GridIntersect(mg)

# %% [markdown]
# ## GWF model
#
# Build a steady-state flow model on the Voronoi grid. Boundary conditions:
# - **CHD left**: head = 2.0, temperature varying linearly 0–100 °C bottom to top
# - **CHD right**: head = 1.0, temperature = 0 °C
# - **CHD bottom**: head = 1.8, temperature = 80 °C (cells not already assigned)
# - **WEL**: two pumping wells and one injection well

# %% [markdown]
# Identify boundary cells with `GridIntersect`, and locate well cells with
# `VertexGrid.intersect`.

# %%
cells_left = gi.intersect(
    LineString([(xmin, ymin), (xmin, ymax)]), geo_dataframe=False
)["cellids"]
cells_right = gi.intersect(
    LineString([(xmax, ymin), (xmax, ymax)]), geo_dataframe=False
)["cellids"]
cells_bottom = gi.intersect(
    LineString([(xmin, ymin), (xmax, ymin)]), geo_dataframe=False
)["cellids"]
well_cells = [
    mg.intersect(p.x, p.y)
    for p in [Point(1200, 500), Point(700, 200), Point(1600, 700)]
]

# %%
gwf_sim = flopy.mf6.MFSimulation(
    sim_name=gwf_name, version="mf6", exe_name="mf6", sim_ws=gwf_ws
)
flopy.mf6.ModflowTdis(gwf_sim, time_units=time_units, perioddata=[[1.0, 1, 1.0]])
gwf = flopy.mf6.ModflowGwf(gwf_sim, modelname=gwf_name, save_flows=True)
flopy.mf6.ModflowIms(
    gwf_sim,
    print_option="SUMMARY",
    complexity="complex",
    outer_dvclose=1.0e-8,
    inner_dvclose=1.0e-8,
    filename=f"{gwf_name}.ims",
)
flopy.mf6.ModflowGwfdisv(
    gwf,
    nlay=nlay,
    ncpl=ncpl,
    nvert=nvert,
    vertices=vertices,
    cell2d=cell2d,
    top=top,
    botm=botm,
)

# %%
# Two pumping wells (Q = −0.05) and one injection well (Q = +5.0).
# Temperature of all injected/infiltrating water is 80 °C.
wells = [[0, c, -0.05, 80.0] for c in well_cells]
wells[2][-2] *= -100  # convert third well to injection, Q = +5.0
flopy.mf6.ModflowGwfwel(
    gwf,
    auxiliary="TEMPERATURE",
    maxbound=len(wells),
    save_flows=True,
    pname="WEL",
    stress_period_data={0: wells},
    filename=f"{gwf_name}.wel",
)
flopy.mf6.ModflowGwfnpf(
    gwf,
    xt3doptions=True,
    k=10.0,
    save_saturation=True,
    save_specific_discharge=True,
)
flopy.mf6.ModflowGwfsto(gwf, ss=0, sy=0, steady_state={0: True})
flopy.mf6.ModflowGwfic(gwf, strt=1.0)

# %%
# Build CHD list. Left-edge cells get a temperature proportional to their
# y-coordinate; right-edge cells get 0 °C; bottom-edge cells not already
# assigned get 80 °C.
chdlist = []
seen = set()
for icpl in cells_left:
    yc = cell2d[icpl][2]
    chdlist.append([(0, icpl), 2.0, 100.0 * yc / ymax])
    seen.add(int(icpl))
for icpl in cells_right:
    chdlist.append([(0, icpl), 1.0, 0.0])
    seen.add(int(icpl))
for icpl in cells_bottom:
    if int(icpl) not in seen:
        chdlist.append([(0, icpl), 1.8, 80.0])

flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary="TEMPERATURE",
    stress_period_data=chdlist,
    pname="CHD",
    filename=f"{gwf_name}.chd",
)
flopy.mf6.ModflowGwfoc(
    gwf,
    budget_filerecord=f"{gwf_name}.cbc",
    head_filerecord=f"{gwf_name}.hds",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
)

# %%
gwf_sim.write_simulation(silent=True)
gwf_sim.run_simulation(silent=True)

# %% [markdown]
# ## GWE model
#
# Solve for the steady-state temperature distribution driven by the CHD and
# WEL source temperatures. The simulation runs for 10⁶ days with 1000
# time steps (multiplier 1.003) to approach steady state.

# %%
gwe_sim = flopy.mf6.MFSimulation(sim_name=gwe_name, sim_ws=gwe_ws, exe_name="mf6")
flopy.mf6.ModflowTdis(
    gwe_sim, time_units=time_units, perioddata=[[1.0e6, 1000, 1.003]]
)

gwe = flopy.mf6.MFModel(
    gwe_sim,
    model_type="gwe6",
    modelname=gwe_name,
    model_nam_file=f"{gwe_name}.nam",
)
imsgwe = flopy.mf6.ModflowIms(
    gwe_sim,
    print_option="SUMMARY",
    outer_dvclose=hclose,
    outer_maximum=nouter,
    under_relaxation="NONE",
    inner_maximum=ninner,
    inner_dvclose=hclose,
    rcloserecord=rclose,
    linear_acceleration="BICGSTAB",
    scaling_method="NONE",
    reordering_method="NONE",
    relaxation_factor=relax,
    filename=f"{gwe_name}.ims",
)
gwe_sim.register_ims_package(imsgwe, [gwe.name])

flopy.mf6.ModflowGwedisv(
    gwe,
    nlay=nlay,
    ncpl=ncpl,
    nvert=nvert,
    vertices=vertices,
    cell2d=cell2d,
    top=top,
    botm=botm,
    pname="DISV",
    filename=f"{gwe_name}.disv",
)
flopy.mf6.ModflowGweic(gwe, strt=strt_temp, pname="IC", filename=f"{gwe_name}.ic")
flopy.mf6.ModflowGweadv(gwe, scheme=scheme, pname="ADV", filename=f"{gwe_name}.adv")
flopy.mf6.ModflowGwecnd(
    gwe,
    alh=alh,
    ath1=ath1,
    ktw=ktw,
    kts=kts,
    pname="CND",
    filename=f"{gwe_name}.cnd",
)
flopy.mf6.ModflowGweest(
    gwe,
    porosity=porosity,
    heat_capacity_water=cpw,
    density_water=rhow,
    latent_heat_vaporization=lhv,
    heat_capacity_solid=cps,
    density_solid=rhos,
    pname="EST",
    filename=f"{gwe_name}.est",
)
flopy.mf6.ModflowGwessm(
    gwe,
    sources=[("WEL", "AUX", "TEMPERATURE"), ("CHD", "AUX", "TEMPERATURE")],
    pname="SSM",
    filename=f"{gwe_name}.ssm",
)
flopy.mf6.ModflowGweoc(
    gwe,
    budget_filerecord=f"{gwe_name}.cbc",
    temperature_filerecord=f"{gwe_name}.ucn",
    saverecord={0: [("TEMPERATURE", "ALL"), ("BUDGET", "ALL")]},
    printrecord=[("TEMPERATURE", "LAST"), ("BUDGET", "LAST")],
)
flopy.mf6.ModflowGwefmi(
    gwe,
    packagedata=[
        ("GWFHEAD", Path(f"../gwf/{gwf_name}.hds"), None),
        ("GWFBUDGET", Path(f"../gwf/{gwf_name}.cbc"), None),
    ],
)

# %%
gwe_sim.write_simulation(silent=True)
gwe_sim.run_simulation(silent=True)

# %% [markdown]
# ## PRT model
#
# Track two particles released near the left edge of the domain.
# Release points 0 and 9 along the `rpts` line are skipped — the first
# few release positions lie in cells that cause a tracking crash in this
# problem; indices 23 and 32 (y ≈ 461 m and y ≈ 641 m) are used instead.

# %%
prpdata = [
    (i, (0, mg.intersect(p[0], p[1])), p[0], p[1], p[2])
    for i, p in enumerate([rpts[23], rpts[32]])
]

prt_sim = flopy.mf6.MFSimulation(
    sim_name=prt_name, version="mf6", exe_name="mf6", sim_ws=prt_ws
)
flopy.mf6.ModflowTdis(prt_sim, time_units=time_units, perioddata=[[1.0, 1, 1.0]])
prt = flopy.mf6.ModflowPrt(prt_sim, modelname=prt_name)
flopy.mf6.ModflowGwfdisv(
    prt,
    nlay=nlay,
    ncpl=ncpl,
    nvert=nvert,
    vertices=vertices,
    cell2d=cell2d,
    top=top,
    botm=botm,
)
flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)
flopy.mf6.ModflowPrtprp(
    prt,
    pname="prp1",
    filename=f"{prt_name}_1.prp",
    nreleasepts=len(prpdata),
    packagedata=prpdata,
    perioddata={0: ["FIRST"]},
    track_filerecord=[f"{prt_name}.prp.trk"],
    trackcsv_filerecord=[f"{prt_name}.prp.trk.csv"],
    boundnames=True,
    stop_at_weak_sink=True,
    exit_solve_tolerance=1e-10,
    extend_tracking=True,
)
flopy.mf6.ModflowPrtoc(
    prt,
    pname="oc",
    track_filerecord=[f"{prt_name}.trk"],
    trackcsv_filerecord=[f"{prt_name}.trk.csv"],
)
flopy.mf6.ModflowPrtfmi(
    prt,
    packagedata=[
        ("GWFHEAD", Path(f"../gwf/{gwf_name}.hds"), None),
        ("GWFBUDGET", Path(f"../gwf/{gwf_name}.cbc"), None),
    ],
)
ems = flopy.mf6.ModflowEms(prt_sim, pname="ems", filename=f"{prt_name}.ems")
prt_sim.register_solution_package(ems, [prt.name])

# %%
prt_sim.write_simulation(silent=True)
prt_sim.run_simulation(silent=True)

# %% [markdown]
# ## Results
#
# Load the steady-state temperature field and particle pathlines.
# Interpolate the GWE temperature at every particle position using a
# Clough–Tocher 2D interpolator built from the Voronoi cell centres.

# %%
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator

temperatures = gwe.output.temperature().get_data()[0, 0]  # shape (ncpl,)

pls = pd.read_csv(prt_ws / f"{prt_name}.trk.csv", na_filter=False)

xc = [cell2d[i][1] for i in range(ncpl)]
yc = [cell2d[i][2] for i in range(ncpl)]
interp = CloughTocher2DInterpolator(list(zip(xc, yc)), temperatures)
pls["therm"] = interp(pls["x"].values, pls["y"].values)

part1 = pls[pls["irpt"] == 1].copy()
part2 = pls[pls["irpt"] == 2].copy()

# %% [markdown]
# Plot the temperature field with flow vectors and particle paths (top panel),
# temperature along the x-axis (middle), and temperature vs. travel time (bottom).

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

spdis = gwf.output.budget().get_data(text="DATA-SPDIS")[0]
qx, qy, _ = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

fig, axes = plt.subplots(
    3, 1,
    figsize=(6, 7),
    tight_layout=True,
    gridspec_kw={"height_ratios": [3, 1, 1]},
)

# map view
ax = axes[0]
ax.set_xlim(0, 2000)
ax.set_ylim(0, 1000)
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(model=gwf, ax=ax)
pmv.plot_grid(alpha=0.25)
tempmesh = pmv.plot_array(temperatures, alpha=0.7)
cv = pmv.contour_array(temperatures, levels=np.linspace(0, 80, 9))
plt.clabel(cv, colors="k")
plt.colorbar(
    tempmesh, shrink=0.5, ax=ax,
    label="Temperature (°C)", location="bottom", fraction=0.1,
)
pmv.plot_vector(qx, qy, normalize=True, alpha=0.25)
pmv.plot_bc(ftype="WEL")
for ipl, (_, pl) in enumerate(pls.groupby(["iprp", "irpt", "trelease"])):
    pl.plot(kind="line", linestyle="--", x="x", y="y", ax=ax, legend=False, color="blue")
    if ipl == 0:
        ax.annotate(
            "Particle 1", xy=(1050, 380), xycoords="data", xytext=(30, -20),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "1.0", "alpha": 0.66},
            arrowprops={"arrowstyle": "->", "shrinkA": 0, "shrinkB": 5,
                        "connectionstyle": "angle,angleA=0,angleB=135,rad=40"},
        )
    else:
        ax.annotate(
            "Particle 2", xy=(1050, 610), xycoords="data", xytext=(-75, 10),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "1.0", "alpha": 0.66},
            arrowprops={"arrowstyle": "->", "shrinkA": 0, "shrinkB": 5,
                        "connectionstyle": "angle,angleA=0,angleB=135,rad=30"},
        )
ax.legend(
    handles=[
        mpl.lines.Line2D(
            [0], [0], marker=">", linestyle="", color="grey",
            markerfacecolor="gray", label="Specific discharge",
        ),
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor="red", label="Well",
        ),
    ],
    loc="lower right",
)

# shared colour norm for the two profile plots
norm = plt.Normalize(
    min(part1["therm"].min(), part2["therm"].min()) - 5,
    max(part1["therm"].max(), part2["therm"].max()) + 5,
)

# temperature vs x-position
for part in [part1, part2]:
    pts = np.array([part["x"], part["therm"]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, norm=norm)
    lc.set_array(part["therm"])
    lc.set_linewidth(3)
    axes[1].add_collection(lc)
axes[1].annotate("Particle 2", xy=(400, 68), xycoords="data")
axes[1].annotate("Particle 1", xy=(400, 50), xycoords="data")
axes[1].set_xlabel("X (m)")
axes[1].set_xlim(0, 2000)
axes[1].set_xticks(np.arange(0, 2100, 250))
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_ylim(40, 80)
axes[1].set_yticks(np.arange(40, 81, 10))

# temperature vs travel time
for part in [part1, part2]:
    pts = np.array([part["t"], part["therm"]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, norm=norm)
    lc.set_array(part["therm"])
    lc.set_linewidth(3)
    axes[2].add_collection(lc)
axes[2].annotate("Particle 2", xy=(15000, 68), xycoords="data")
axes[2].annotate("Particle 1", xy=(15000, 50), xycoords="data")
axes[2].set_xlabel("Time (days)")
axes[2].set_xlim(0, max(part1["t"].max(), part2["t"].max()))
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_ylim(40, 80)
axes[2].set_yticks(np.arange(40, 81, 10))

plt.show()
