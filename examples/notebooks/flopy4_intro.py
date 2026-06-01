# # flopy4 intro
#
# Steady-state groundwater flow on a circular DISV (vertex) grid.
#
# Topics covered:
# * syncing flopy4 package definitions to a MODFLOW 6 release
# * building a `VertexGrid` from a GRB file, then deriving `Disv` from it
# * constructing boundary packages with various period-data input styles
# * assembling a GWF simulation, writing input, and running MODFLOW 6
# * reading head and budget output and plotting results inline

# ### Imports

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu
from flopy.mf6.utils.binarygrid_util import MfGrdFile

import flopy4
import flopy4.mf6
import flopy4.mf6.gwf
import flopy4.mf6.simulation
from flopy4.mf6.gwf import Chd, Ic, Npf, Oc, Rch, Sto
from flopy4.mf6.gwf.disv import Disv
from flopy4.mf6.utils.grid import VertexGrid
from flopy4.mf6.utils.time import Time

# ### Sync
#
# `flopy4 mf6 sync` regenerates the Python package definitions from MODFLOW 6
# DFN files for a given release.  Running it ensures the flopy4 classes
# (`Chd`, `Npf`, `Disv`, ...) match the version of MODFLOW 6 you have installed.
# By default it targets the release matching the `mf6` binary on your PATH;
# pass `--release` to pin a specific version.

!flopy4 mf6 sync

# ### Setup

try:
    NOTEBOOK_ROOT = Path(__file__).parent.parent
except NameError:
    NOTEBOOK_ROOT = Path.cwd().parent

DATA_ROOT = NOTEBOOK_ROOT / "data" / "circle"
WORKSPACE = NOTEBOOK_ROOT / "_output" / "circle"
WORKSPACE.mkdir(parents=True, exist_ok=True)

# ### Timing
#
# One steady-state stress period of length 1 day with a single time step.

time = Time(perlen=[1.0], nstp=[1], tsmult=[1.0], time_units="days")
nper = time.nper

# ### Build the grid
#
# Load an existing GRB (binary grid) file to extract vertex coordinates and
# cell connectivity.  A GRB records everything MODFLOW 6 needs to describe
# the mesh; using it here avoids duplicating the geometry in Python.
#
# We build a `VertexGrid` directly from those arrays, then derive the `Disv`
# discretization package from the grid via `Disv.from_grid()` — the reverse
# of the usual "construct Disv first, call `to_grid()` later" workflow.

grb = MfGrdFile(DATA_ROOT / "disv.disv.grb", verbose=True)

nlay, ncpl = int(grb.nlay), int(grb.ncpl)
top = np.ravel(grb.top)
botm = grb.bot.reshape(nlay, ncpl)
idomain = grb.idomain
vertices, cell2d = grb.cell2d

grid = VertexGrid(
    xoff=573309.700,
    yoff=4102552.000,
    nlay=nlay,
    ncpl=ncpl,
    top=top,
    botm=botm,
    idomain=idomain.reshape(nlay, ncpl),
    iv=np.array([v[0] for v in vertices], dtype=int),
    xv=np.array([v[1] for v in vertices], dtype=float),
    yv=np.array([v[2] for v in vertices], dtype=float),
    cell2d=cell2d,
)

# `Disv.from_grid()` reads vertex coordinates and cell2d directly from the
# VertexGrid — no need to restate the geometry.
disv = Disv.from_grid(grid)
disv.crs = "EPSG:26911"
disv.length_units = "meters"

# ### Plot the grid

ugrid = grid.ugrid

fig, ax = plt.subplots(figsize=(6, 6))
xu.plot.line(ugrid, ax=ax)
ax.set_aspect(1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("DISV grid")
plt.tight_layout()
plt.savefig(WORKSPACE / "grid.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# ### Packages

dims = {
    "nper": nper,
    "nlay": nlay,
    "ncpl": ncpl,
    "nvert": len(vertices),
    "nodes": nlay * ncpl,
}

idomain_uda = xu.UgridDataArray(
    xr.DataArray(
        idomain.reshape(nlay, ncpl),
        coords={"layer": list(range(1, nlay + 1))},
        dims=["layer", ugrid.face_dimension],
    ),
    grid=ugrid,
)

ic = Ic(strt=0.0, dims=dims)

icelltype = xu.full_like(idomain_uda, 0)
k = xu.full_like(idomain_uda, 1.0, dtype=float)
npf = Npf(
    icelltype=icelltype.values.ravel(),
    k=k.values.ravel(),
    k33=k.values.ravel(),
    save_flows=True,
    dims=dims,
)

sto = Sto(
    ss=1.0e-5,
    sy=0.15,
    steady_state=[True],
    iconvert=0,
    dims=dims,
)

# ### Period data input styles
#
# flopy4 stress packages accept period data in several equivalent forms.
# All map stress-period index → cell-id tuple → value.
#
# **Style 1 — explicit period index**
#
# Use an integer key for each stress period.  The cell-id tuple follows the
# discretization: `(layer, icpl)` for DISV, `(layer, row, col)` for DIS.
#
#     head = {0: {(0, 0): 1.0, (0, 5): 0.5}}
#
# **Style 2 — wildcard `"*"` (applies to all periods)**
#
# The sentinel `"*"` means "repeat this block for every stress period".
#
#     recharge = {"*": {(0, j): 0.001 for j in range(ncpl)}}
#
# **Style 3 — per-period variation**
#
# Each period gets its own inner dict.  Useful when boundary values change
# between stress periods (e.g. a pumping schedule or a seasonal recharge).
#
#     rch = {
#         0: {(0, j): 0.001 for j in range(ncpl)},   # wet season
#         1: {(0, j): 0.0005 for j in range(ncpl)},  # transitional
#         2: {(0, j): 0.0001 for j in range(ncpl)},  # dry season
#     }
#
# **Style 4 — DataFrame via `stress_period_data`**
#
# After construction you can read or write period data as a pandas DataFrame
# through the `stress_period_data` property.  Each row is one active record;
# columns are `kper`, the coordinate columns (`layer`/`row`/`col` or `node`),
# and the package-specific value column(s).
#
#     df = chd.stress_period_data       # inspect
#     df["head"] *= 0.9                 # scale all head values
#     chd.stress_period_data = df       # write back

# ### Constant-head boundary (CHD)
#
# Apply head = 1.0 m on the outer ring of cells identified by
# `binary_dilation` — cells that touch the domain boundary.
# Uses **style 1**: explicit period-index dict.

chd_location = xu.zeros_like(idomain_uda.sel(layer=2), dtype=bool).ugrid.binary_dilation(
    border_value=True
)

# Style 1: explicit period-index dict.  Key is 0-based stress period index;
# inner dict maps (layer, icpl) cell-id tuples to head values.
chd = Chd(
    head={0: {(1, int(i)): 1.0 for i in np.where(chd_location)[0]}},
    print_input=True,
    print_flows=True,
    save_flows=True,
    dims=dims,
)

# ### Plot CHD cells

constant_head = xu.full_like(idomain_uda.sel(layer=2), 1.0, dtype=float).where(chd_location)

fig, ax = plt.subplots(figsize=(6, 6))
constant_head.ugrid.plot(ax=ax)
xu.plot.line(ugrid, ax=ax, color="black")
ax.set_aspect(1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("CHD cells")
plt.tight_layout()
plt.savefig(WORKSPACE / "chd.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# ### Recharge (RCH)
#
# Uniform recharge of 0.001 m/day on every top-layer cell.
# Uses **style 2**: wildcard `"*"` — same rate for every stress period.

rch = Rch(
    recharge={"*": {(0, j): 0.001 for j in range(ncpl)}},
    dims=dims,
)

# ### Inspect period data as a DataFrame (style 4)
#
# `stress_period_data` returns a tidy DataFrame — one row per active record.
# Useful for auditing, plotting, or bulk edits before running.

df = chd.stress_period_data
print(df.head())
print(f"\n{len(df)} CHD records, {df['kper'].nunique()} period(s)")

# ### Output control

oc = Oc(
    budget_file="gwf.bud",
    head_file="gwf.hds",
    save_head={0: "all"},
    save_budget={0: "all"},
    dims=dims,
)

# ### GWF model

gwf = flopy4.mf6.gwf.Gwf(
    dis=disv,
    ic=ic,
    npf=npf,
    sto=sto,
    chd=[chd],
    rch=[rch],
    oc=oc,
)

# ### Solver

ims = flopy4.mf6.Ims(
    print_option="summary",
    outer_dvclose=1.0e-4,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-4,
    rclose=flopy4.mf6.Ims.Rclose(inner_rclose=0.001),
    inner_maximum=100,
    linear_acceleration="cg",
    reordering_method=None,
    relaxation_factor=0.97,
    models=["gwf"],
)

# ### Simulation

tdis = flopy4.mf6.simulation.Tdis.from_time(time)

sim = flopy4.mf6.simulation.Simulation(
    name="circle",
    tdis=tdis,
    models={"gwf": gwf},
    solutions={"ims": ims},
    workspace=WORKSPACE,
)

sim.write()
sim.run(verbose=True)

# ### Head and flow vectors
#
# `gwf.output.head` returns a `UgridDataArray` for DISV models.
# `gwf.output.budget` returns a `UgridDataset` keyed by face-flow term.

head = gwf.output.head
cbc = gwf.output.budget

# Assemble face-normal flow components into a dataset and attach
# edge-centre coordinates so `.plot.quiver()` can place the arrows.
cbc_grid = cbc["flow-horizontal-face-x"].grid
ds = xu.UgridDataset(grids=cbc_grid)
ds["u"] = cbc["flow-horizontal-face-x"]
ds["v"] = cbc["flow-horizontal-face-y"]
ds = ds.ugrid.assign_edge_coords()

fig, ax = plt.subplots(figsize=(7, 6))
head.isel(time=0, layer=0).compute().ugrid.plot(ax=ax, cmap="viridis")
ds.isel(time=0, layer=0).plot.quiver(
    x="mesh2d_edge_x", y="mesh2d_edge_y", u="u", v="v", color="white", ax=ax
)
ax.set_aspect(1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_title("Head (m) with flow vectors — layer 1, time step 1")
plt.tight_layout()
plt.savefig(WORKSPACE / "head.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
