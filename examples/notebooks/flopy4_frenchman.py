# # Frenchman Flat — UX and write performance: flopy4 vs. flopy3
#
# The Frenchman Flat, NV regional groundwater model is a real-world
# 10-layer, 87×87 DIS simulation (75,690 cells) with heterogeneous
# hydraulic conductivity, specific storage, and three separate WEL
# packages representing constant-rate pumping, subsurface leakage, and
# water-sampling extraction.
#
# This notebook uses the model to compare flopy4 and flopy3 on two axes:
#
# * **UX** — how each library expresses the same model in Python
# * **Write speed** — how long it takes to serialize the input files to disk
#
# The full model has 33 transient stress periods; we use 4 here so the
# write completes in seconds while still exercising the full grid and
# heterogeneous array data.
#
# Data: https://www.sciencebase.gov/catalog/item/641a1b51d34eb496d1d2a1fd

# ## Imports

from pathlib import Path
from time import perf_counter

import numpy as np
import flopy.mf6 as mf6
import flopy4
from flopy4.mf6 import Ims
from flopy4.mf6.gwf import Dis, Ic, Npf, Sto, Wel, Oc, Gwf
from flopy4.mf6.simulation import Simulation, Tdis
from flopy4.mf6.utils.grid import StructuredGrid
from flopy4.mf6.utils.time import Time as MfTime

# ## Data path
#
# The Frenchman Flat hydraulic-property arrays are distributed with the
# pyphoenix-project repository.  The path below expects a sibling checkout
# alongside this training repository; adjust `FF_ARRAYS` if yours is
# elsewhere.

try:
    _root = Path(__file__).parents[2]
except NameError:
    _root = Path.cwd().parents[1]

FF_ARRAYS = _root.parent / "pyphoenix-project" / "docs" / "examples" / "data" / "frenchman-flat" / "arrays"
assert FF_ARRAYS.exists(), f"Array data not found: {FF_ARRAYS}"

# ## Stress periods
#
# Four transient periods drawn from the original 33-period pumping schedule.
# Even at 4 periods the write touches every package — DIS, NPF, STO, IC, OC,
# and three WEL files — with large (87×87) per-layer arrays.

perlen = [1.17707, 0.84374, 4.61527, 0.41874]
nstp   = [15, 15, 15, 15]
tsmult = [1.1, 1.1, 1.1, 1.1]
nper   = len(perlen)

# ## Grid
#
# 87×87 structured grid with variable column widths (`delr`/`delc`) that
# refine toward the centre of the domain where the wells are located.

nlay, nrow, ncol = 10, 87, 87
shape = (nlay, nrow, ncol)

delr = np.array([
    2500., 2500., 2500., 2150., 1800., 1500., 1250., 1000.,  750.,  750.,
     500.,  500.,  500.,  500.,  500.,  350.,  250.,  200.,  150.,  125.,
     100.,  100.,  100.,  100.,  100.,   75.,   50.,   50.,   50.,   50.,
      50.,   30.,   30.,   15.,   15.,   13.5,  10.,   6.5,   5.,    3.5,
       2.5,   2.,   1.5,   1.,   1.5,   2.,    2.5,   3.5,   5.,    6.5,
      10.,   13.5,  15.,   15.,   30.,   30.,   50.,   50.,   50.,   50.,
      50.,   75.,  100.,  100.,  100.,  100.,  100.,  125.,  150.,  200.,
     250.,  350.,  500.,  500.,  500.,  500.,  500.,  750.,  750., 1000.,
    1250., 1500., 1800., 2150., 2500., 2500., 2500.,
])
delc = delr.copy()

idomain = np.ones(shape, dtype=int)
top     = np.zeros((nrow, ncol), dtype=float)
botm    = np.stack([
    np.full((nrow, ncol), v)
    for v in [-200., -400., -600., -800., -1050., -1350., -1700., -2200., -2950., -3950.]
])

# ## Arrays
#
# Hydraulic conductivity and specific storage loaded from per-layer text
# files.  These large arrays (87×87 × 10 layers) are the bulk of the data
# that both libraries must serialize, making them the main driver of write
# time differences.

icelltype = np.zeros(shape, dtype=int)
k         = np.zeros(shape, dtype=float)
k33       = np.zeros(shape, dtype=float)
ss        = np.zeros(shape, dtype=float)

for l in range(nlay):
    pad   = "000" if l < 9 else "00"
    k[l]  = np.loadtxt(FF_ARRAYS / f"Array.MF-HydK_{pad}{l + 1}.txt")
    k33[l] = k[l] * 0.1   # 10:1 horizontal-to-vertical anisotropy
    ss[l] = np.loadtxt(FF_ARRAYS / f"Array.MF-HydS_{pad}{l + 1}.txt")

# # flopy4
#
# flopy4 builds a simulation from plain Python objects.  A few things to
# notice:
#
# * `StructuredGrid` and `Time` capture spatial and temporal discretization
#   once; `dims` propagates those sizes to every package automatically.
# * Stress-period data is a nested dict `{period: {cellid: value}}` — you
#   write exactly what you mean, with no lists-of-tuples bookkeeping.
# * Packages are keyword arguments to `Gwf`; the model knows its own
#   components at construction time.
# * `sim.write()` is a single, obvious call.

# ## Grid and time

ff_grid = StructuredGrid(
    lenuni="meters",
    xoff=573309.700,
    yoff=4102552.000 - delr.sum(),   # xul/yul → xll/yll
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    top=top,
    botm=botm,
    delr=delr,
    delc=delc,
    idomain=idomain,
    crs="EPSG:26911",
)
ff_time = MfTime(perlen=perlen, nstp=nstp, tsmult=tsmult)
dims    = {"nper": nper, "ncpl": nrow * ncol, **dict(ff_grid.dataset.sizes)}

# ## Packages

dis4 = Dis.from_grid(grid=ff_grid)
ic4  = Ic(strt=0.0, dims=dims)
npf4 = Npf(icelltype=icelltype, k=k, k33=k33, save_flows=True, dims=dims)
sto4 = Sto(ss=ss, iconvert=0, dims=dims)

# Three WEL packages.  Stress-period data is a plain nested dict:
# {period: {(lay, row, col): q}}.  Periods not listed carry the previous
# period forward automatically.
wel4_crt = Wel(
    filename="ff.crt.wel",
    q={
        0: {(1, 43, 43): -30992.50},
        1: {(1, 43, 43):      0.00},
        2: {(1, 43, 43): -30992.50},
        3: {(1, 43, 43):      0.00},
    },
    save_flows=True,
    dims=dims,
)
wel4_leak = Wel(
    filename="ff.leak.wel",
    q={0: {(1, 43, 43): 1e-5}},
    save_flows=True,
    dims=dims,
)
wel4_sampleQ = Wel(
    filename="ff.sampleQ.wel",
    q={0: {(1, 43, 43): 0.0}},
    save_flows=True,
    dims=dims,
)

oc4 = Oc(
    budget_file=Path("ff.cbc"),
    head_file=Path("ff.hds"),
    save_head={"0": "all"},
    save_budget={"0": "last"},
    dims=dims,
)

gwf4 = Gwf(
    dis=dis4,
    ic=ic4,
    npf=npf4,
    sto=sto4,
    oc=oc4,
    wel=[wel4_crt, wel4_leak, wel4_sampleQ],
    dims=dims,
)

ims4 = Ims(
    print_option="summary",
    complexity="moderate",
    outer_dvclose=0.01,
    outer_maximum=50,
    under_relaxation="DBD",
    under_relaxation_theta=0.9,
    under_relaxation_kappa=0.0001,
    under_relaxation_gamma=0.0,
    under_relaxation_momentum=0.0,
    inner_dvclose=0.00001,
    rclose=Ims.Rclose(inner_rclose=0.1),
    inner_maximum=100,
    linear_acceleration="bicgstab",
    number_orthogonalizations=0,
    reordering_method=None,
    models=["ff"],
)
tdis4 = Tdis.from_time(ff_time)

workspace4 = Path("frenchman-flat") / "flopy4"
workspace4.mkdir(parents=True, exist_ok=True)

sim4 = Simulation(
    name="ff",
    tdis=tdis4,
    models={"ff": gwf4},
    solutions={"ims": ims4},
    workspace=workspace4,
)

# ## Write (timed)

t0          = perf_counter()
sim4.write()
flopy4_time = perf_counter() - t0
print(f"flopy4  write: {flopy4_time:.3f} s")

# # flopy3
#
# The same model expressed with the standard `flopy` package.  Some
# differences worth noting:
#
# * Every package is a separate class, each with its own positional-argument
#   conventions.
# * Stress-period data is `{period: [((lay, row, col), q), ...]}` — a dict
#   of lists of tuples — and must be kept in sync with `nper` manually.
# * Multiple WEL packages require an explicit `pname` to avoid name
#   collisions on the model.
# * The write call is `write_simulation()`.

workspace3 = Path("frenchman-flat") / "flopy3"
workspace3.mkdir(parents=True, exist_ok=True)

sim3   = mf6.MFSimulation(sim_name="ff", sim_ws=str(workspace3))
tdis3  = mf6.ModflowTdis(sim3, nper=nper, perioddata=list(zip(perlen, nstp, tsmult)))
ims3   = mf6.ModflowIms(
    sim3,
    print_option="summary",
    complexity="moderate",
    outer_dvclose=0.01,
    outer_maximum=50,
    under_relaxation="dbd",
    under_relaxation_theta=0.9,
    under_relaxation_kappa=0.0001,
    under_relaxation_gamma=0.0,
    under_relaxation_momentum=0.0,
    inner_dvclose=0.00001,
    rcloserecord=0.1,
    inner_maximum=100,
    linear_acceleration="bicgstab",
)
gwf3   = mf6.ModflowGwf(sim3, modelname="ff")
dis3   = mf6.ModflowGwfdis(
    gwf3,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=top, botm=botm,
    idomain=idomain,
)
ic3    = mf6.ModflowGwfic(gwf3, strt=0.0)
npf3   = mf6.ModflowGwfnpf(gwf3, icelltype=icelltype, k=k, k33=k33, save_flows=True)
sto3   = mf6.ModflowGwfsto(gwf3, ss=ss, iconvert=0)

# flopy3 stress_period_data: {period: [((lay, row, col), q), ...]}
wel3_crt = mf6.ModflowGwfwel(
    gwf3,
    stress_period_data={
        0: [((1, 43, 43), -30992.50)],
        1: [((1, 43, 43),      0.00)],
        2: [((1, 43, 43), -30992.50)],
        3: [((1, 43, 43),      0.00)],
    },
    save_flows=True,
    filename="ff.crt.wel",
    pname="wel_crt",
)
wel3_leak = mf6.ModflowGwfwel(
    gwf3,
    stress_period_data={0: [((1, 43, 43), 1e-5)]},
    save_flows=True,
    filename="ff.leak.wel",
    pname="wel_leak",
)
wel3_sampleQ = mf6.ModflowGwfwel(
    gwf3,
    stress_period_data={0: [((1, 43, 43), 0.0)]},
    save_flows=True,
    filename="ff.sampleQ.wel",
    pname="wel_sampleQ",
)
oc3    = mf6.ModflowGwfoc(
    gwf3,
    budget_filerecord="ff.cbc",
    head_filerecord="ff.hds",
    saverecord={0: [("HEAD", "ALL"), ("BUDGET", "LAST")]},
    printrecord={0: [("BUDGET", "LAST")]},
)

# ## Write (timed)

t0          = perf_counter()
sim3.write_simulation()
flopy3_time = perf_counter() - t0
print(f"flopy3  write: {flopy3_time:.3f} s")

# # Results

speedup = flopy3_time / flopy4_time
print()
print(f"  flopy4   {flopy4_time:.3f} s")
print(f"  flopy3   {flopy3_time:.3f} s")
print(f"  speedup  {speedup:.1f}×  (flopy4 faster)")
