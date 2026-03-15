#!/usr/bin/env python
"""
Sweep over (ncoils, n_unique_shapes) combinations, running the Stage-II
coil optimisation from the notebook for each pair and saving results into
separate output directories.

Usage:
    python run_sweep.py                   # run full sweep
    python run_sweep.py --maxiter 50      # override LBFGS max iterations
    python run_sweep.py --dry-run         # list combos without running
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from simsopt.geo import (
    SurfaceRZFourier,
    create_equally_spaced_curves,
    CurveLength,
    curves_to_vtk,
    RotatedCurve,
)
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty


# ---------------------------------------------------------------------------
# PyTorch / SIMSOPT bridge (same as notebook)
# ---------------------------------------------------------------------------
class SimsoptObjective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dofs, objective_obj):
        dofs_np = dofs.detach().cpu().numpy()
        objective_obj.x = dofs_np
        loss_val = objective_obj.J()
        ctx.objective_obj = objective_obj
        return torch.tensor(loss_val, dtype=dofs.dtype, device=dofs.device)

    @staticmethod
    def backward(ctx, grad_output):
        objective_obj = ctx.objective_obj
        grad_np = objective_obj.dJ()
        grad = torch.from_numpy(grad_np).to(grad_output.device).type(grad_output.dtype)
        return grad * grad_output, None


def simsopt_loss(dofs, objective_obj):
    return SimsoptObjective.apply(dofs, objective_obj)


# ---------------------------------------------------------------------------
# Single-run optimisation
# ---------------------------------------------------------------------------
def run_single(ncoils: int, n_unique_shapes: int, out_dir: str,
               maxiter: int = 100, length_weight: float = 1.0,
               length_target: float = 18.0, lr: float = 0.10,
               order: int = 5, R0: float = 1.0, R1: float = 0.5):
    """Run one optimisation and save results into *out_dir*."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # ----- surface -----
    test_dir = Path(__file__).resolve().parent / ".." / "data" / "test_files"
    filename = test_dir / "input.LandremanPaul2021_QA"

    nphi, ntheta = 32, 32
    s = SurfaceRZFourier.from_vmec_input(str(filename), range="full torus",
                                          nphi=nphi, ntheta=ntheta)

    # ----- coils -----
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True,
                                                R0=R0, R1=R1, order=order)

    delta_angle = (2 * np.pi) / (2 * s.nfp * ncoils)
    for i in range(n_unique_shapes, ncoils):
        template_idx = i % n_unique_shapes
        angle_diff = (i - template_idx) * delta_angle
        base_curves[i] = RotatedCurve(base_curves[template_idx], angle_diff,
                                       flip=False)

    base_currents = [Current(1.0) * 1e5 for _ in range(ncoils)]
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))

    curves = [c.curve for c in coils]
    curves_to_vtk(curves, os.path.join(out_dir, "curves_init"), close=True)
    pointData = {
        "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(),
                       axis=2)[:, :, None]
    }
    s.to_vtk(os.path.join(out_dir, "surf_init"), extra_data=pointData)

    # ----- objective -----
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    JF = Jf + length_weight * QuadraticPenalty(sum(Jls), length_target, "max")

    B_dot_n_init = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(),
                           axis=2)
    init_max_Bn = float(np.max(np.abs(B_dot_n_init)))

    # ----- optimise -----
    dofs_init = JF.x.copy()
    dofs_tensor = torch.tensor(dofs_init, dtype=torch.float64, device=device,
                                requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [dofs_tensor], lr=lr, max_iter=maxiter, max_eval=maxiter * 2,
        tolerance_grad=1e-9, tolerance_change=1e-9,
        history_size=100, line_search_fn="strong_wolfe",
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()
        loss = simsopt_loss(dofs_tensor, JF)
        loss.backward()
        loss_history.append(loss.item())
        return loss

    t0 = time.time()
    optimizer.step(closure)
    elapsed = time.time() - t0

    # ----- save results -----
    JF.x = dofs_tensor.detach().cpu().numpy()
    curves_to_vtk(curves, os.path.join(out_dir, "curves_opt"), close=True)
    pointData = {
        "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(),
                       axis=2)[:, :, None]
    }
    s.to_vtk(os.path.join(out_dir, "surf_opt"), extra_data=pointData)
    bs.save(os.path.join(out_dir, "biot_savart_opt.json"))

    B_dot_n_final = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(),
                            axis=2)
    final_max_Bn = float(np.max(np.abs(B_dot_n_final)))
    total_curve_length = float(sum(func.J() for func in Jls))

    summary = {
        "ncoils": ncoils,
        "n_unique_shapes": n_unique_shapes,
        "order": order,
        "maxiter": maxiter,
        "length_weight": length_weight,
        "length_target": length_target,
        "lr": lr,
        "n_dofs": len(dofs_init),
        "init_max_Bn": init_max_Bn,
        "final_max_Bn": final_max_Bn,
        "total_base_curve_length": total_curve_length,
        "final_loss": loss_history[-1] if loss_history else None,
        "n_iters": len(loss_history),
        "elapsed_s": round(elapsed, 2),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savetxt(os.path.join(out_dir, "loss_history.txt"), loss_history)

    return summary


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------
def get_divisors(n):
    """Return sorted list of divisors of n."""
    return sorted(d for d in range(1, n + 1) if n % d == 0)


def build_sweep_combos(ncoils_list):
    """For each ncoils, pair with every valid n_unique_shapes."""
    combos = []
    for nc in ncoils_list:
        for nuq in get_divisors(nc):
            combos.append((nc, nuq))
    return combos


def main():
    parser = argparse.ArgumentParser(description="Sweep ncoils × n_unique_shapes")
    parser.add_argument("--maxiter", type=int, default=100,
                        help="LBFGS max iterations per run")
    parser.add_argument("--lr", type=float, default=0.10,
                        help="LBFGS learning rate")
    parser.add_argument("--output-root", type=str, default="sweep_output",
                        help="Root directory for all sweep outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print combos and exit")
    args = parser.parse_args()

    ncoils_list = [1, 2, 3, 4, 6, 8]
    combos = build_sweep_combos(ncoils_list)

    if args.dry_run:
        print(f"{'ncoils':>6}  {'n_unique':>8}  {'dir'}")
        print("-" * 40)
        for nc, nuq in combos:
            d = f"ncoils{nc}_nuniq{nuq}"
            print(f"{nc:>6}  {nuq:>8}  {d}")
        print(f"\nTotal runs: {len(combos)}")
        return

    os.makedirs(args.output_root, exist_ok=True)
    all_summaries = []

    for idx, (nc, nuq) in enumerate(combos, 1):
        run_dir = os.path.join(args.output_root, f"ncoils{nc}_nuniq{nuq}")
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(combos)}]  ncoils={nc}, n_unique_shapes={nuq}")
        print(f"  Output: {run_dir}")
        print(f"{'='*60}")

        summary = run_single(
            ncoils=nc,
            n_unique_shapes=nuq,
            out_dir=run_dir,
            maxiter=args.maxiter,
            lr=args.lr,
        )
        all_summaries.append(summary)

        print(f"  -> final_max_Bn={summary['final_max_Bn']:.4e}, "
              f"loss={summary['final_loss']:.4e}, "
              f"time={summary['elapsed_s']}s")

    # Save combined summary
    with open(os.path.join(args.output_root, "sweep_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'ncoils':>6} {'n_uniq':>6} {'n_dofs':>6} {'max|Bn| init':>14} "
          f"{'max|Bn| final':>14} {'loss':>12} {'time(s)':>8}")
    print("-" * 70)
    for s in all_summaries:
        print(f"{s['ncoils']:>6} {s['n_unique_shapes']:>6} {s['n_dofs']:>6} "
              f"{s['init_max_Bn']:>14.4e} {s['final_max_Bn']:>14.4e} "
              f"{s['final_loss']:>12.4e} {s['elapsed_s']:>8.1f}")
    print(f"{'='*70}")
    print(f"Results saved to {args.output_root}/sweep_summary.json")


if __name__ == "__main__":
    main()
