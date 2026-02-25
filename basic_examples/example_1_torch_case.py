#!/usr/bin/env python

r"""
Torch-enabled version of example_1_case.py.

Key idea:
- SIMSOPT computes physics objective + analytic gradient wrt SIMSOPT dofs.
- Torch applies geometry constraints/transforms to a raw parameter vector.
- A custom torch.autograd.Function bridges SIMSOPT J/dJ into Torch backward.
"""

import os
from pathlib import Path
from numpy.random import PCG64DXSM, Generator
import numpy as np
import torch

from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
    MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation, GaussianSampler, CurvePerturbed, PerturbationSample
)
from simsopt.objectives import QuadraticPenalty, MPIObjective, SquaredFlux
from simsopt.util import in_github_actions, proc0_print, comm_world


# -----------------------------
# Same user parameters as example_1_case.py
# -----------------------------
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 24

LENGTH_WEIGHT = 1e-6
DISTANCE_THRESHOLD = 0.1
DISTANCE_WEIGHT = 10
CURVATURE_THRESHOLD = 5.0
CURVATURE_WEIGHT = 1e-6
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6
ARCLENGTH_WEIGHT = 1e-2

SIGMA = 1e-3
L = 0.5
N_SAMPLES = 16
N_OOS = 256

MAXITER = 50 if in_github_actions else 400

TEST_DIR = (Path(__file__).parent / ".." / "data" / "test_files").resolve()
filename = TEST_DIR / "input.LandremanPaul2021_QA"

OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Build physics model (same as original)
# -----------------------------
nphi = 64
ntheta = 16
s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)

base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for _ in range(ncoils)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = CurveCurveDistance(curves, DISTANCE_THRESHOLD, num_basecurves=ncoils)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]

seed = 0
rg = Generator(PCG64DXSM(seed))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)

Jfs = []
curves_pert = []
for i in range(N_SAMPLES):
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils_sym = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils_sym]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jfs.append(SquaredFlux(s, bs_pert))

Jmpi = MPIObjective(Jfs, comm_world, needs_splitting=True)

for i in range(len(curves_pert)):
    curves_to_vtk(curves_pert[i], OUT_DIR + f"curves_init_{i}")

JF = (
    Jmpi
    + LENGTH_WEIGHT * sum(Jls)
    + DISTANCE_WEIGHT * Jdist
    + CURVATURE_WEIGHT * sum(Jcs)
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
    + ARCLENGTH_WEIGHT * sum(Jals)
)


# -----------------------------
# Torch bridge: SIMSOPT J/dJ -> autograd
# -----------------------------
class SimsoptObjectiveTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dofs_torch):
        dofs_np = dofs_torch.detach().cpu().numpy()
        JF.x = dofs_np
        J = float(JF.J())
        dJ = np.asarray(JF.dJ(), dtype=np.float64)
        dJ_torch = torch.from_numpy(dJ).to(device=dofs_torch.device, dtype=dofs_torch.dtype)
        ctx.save_for_backward(dJ_torch)
        return torch.tensor(J, device=dofs_torch.device, dtype=dofs_torch.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (dJ_torch,) = ctx.saved_tensors
        return grad_output * dJ_torch


def physics_loss_from_dofs(dofs_torch):
    return SimsoptObjectiveTorch.apply(dofs_torch)


# -----------------------------
# Put your torch constraints here
# -----------------------------
def apply_geometry_constraints(raw_params):
    """
    Map unconstrained torch params -> valid SIMSOPT dofs.
    Replace with your own torch operations.
    Keep output shape identical to JF.x.

    Examples:
      - positivity: x = torch.nn.functional.softplus(raw)
      - bounded range: x = xmin + (xmax-xmin)*torch.sigmoid(raw)
      - frozen subsets via masking
    """
    # Identity map by default:
    return raw_params


def report_metrics(loss_value):
    jf = Jmpi.J()
    outstr = f"J={loss_value:.1e}, ⟨Jf⟩={jf:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += (
        f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, "
        f"ϰ=[{kap_string}], ∫ϰ²/L>=[{msc_string}], C-C-Sep={Jdist.shortest_distance():.2f}"
    )
    grad_norm = np.linalg.norm(JF.dJ())
    outstr += f", ║∇J║={grad_norm:.1e}"
    proc0_print(outstr, flush=True)


# -----------------------------
# Taylor test (same style)
# -----------------------------
proc0_print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")

x0_np = np.array(JF.x, dtype=np.float64)
np.random.seed(1)
h = np.random.uniform(size=x0_np.shape)

def f_np(dofs_np):
    JF.x = dofs_np
    return JF.J(), JF.dJ()

J0, dJ0 = f_np(x0_np)
dJh = float(np.dot(dJ0, h))
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f_np(x0_np + eps * h)
    J2, _ = f_np(x0_np - eps * h)
    proc0_print("err", (J1 - J2) / (2 * eps) - dJh)


# -----------------------------
# Torch optimization
# -----------------------------
proc0_print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

dtype = torch.float64
device = torch.device("cpu")

raw = torch.tensor(x0_np, dtype=dtype, device=device, requires_grad=True)

optimizer = torch.optim.LBFGS(
    [raw],
    lr=1.0,
    max_iter=MAXITER,
    max_eval=MAXITER * 5,
    tolerance_grad=1e-15,
    tolerance_change=1e-15,
    history_size=400,
    line_search_fn="strong_wolfe",
)

def closure():
    optimizer.zero_grad()
    dofs = apply_geometry_constraints(raw)
    loss = physics_loss_from_dofs(dofs)
    loss.backward()
    report_metrics(float(loss.item()))
    return loss

optimizer.step(closure)

x_opt = apply_geometry_constraints(raw).detach().cpu().numpy()
JF.x = x_opt

alen_string = ", ".join(
    [f"{np.max(c.incremental_arclength()) / np.min(c.incremental_arclength()) - 1:.2e}" for c in base_curves]
)
proc0_print(f"Final arclength variation max(|ℓ|)/min(|ℓ|) - 1=[{alen_string}]")


# -----------------------------
# Evaluate optimized coils (same style)
# -----------------------------
proc0_print("""
################################################################################
### Evaluate the obtained coils ################################################
################################################################################
""")

curves_to_vtk(curves, OUT_DIR + "curves_opt")
for i in range(len(curves_pert)):
    curves_to_vtk(curves_pert[i], OUT_DIR + f"curves_opt_{i}")

pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

Jf.x = x_opt
proc0_print(f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}")
proc0_print(f"Flux Objective for exact coils coils      : {Jf.J():.3e}")

rg = Generator(PCG64DXSM(seed + 1))
val = 0.0
for _ in range(N_OOS):
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils_sym = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils_sym]
    bs_pert = BiotSavart(coils_pert)
    val += SquaredFlux(s, bs_pert).J()

val *= 1.0 / N_OOS
proc0_print(f"Out-of-sample flux value                  : {val:.3e}")