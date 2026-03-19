# tme/analysis.py
"""Extract MHD quantities from VMEC output and run Boozer analysis.

Implements REQ-ANALYSIS-001 through REQ-ANALYSIS-003.
"""
import os
import json
import numpy as np
import netCDF4 as nc


def extract_vmec_results(wout_path: str) -> dict:
    """
    Extract key quantities from VMEC wout NetCDF file.

    Returns dict with:
        iotaf: list[float] - rotational transform on full mesh
        presf: list[float] - pressure on full mesh
        betatot: float - volume-averaged beta
        DMerc: list[float] - Mercier stability criterion
        Vp: list[float] - V'(s), derivative of volume w.r.t. flux
        ier_flag: int - VMEC convergence flag (0 = success)
        fsql: float - final force residual
        ns: int - number of radial surfaces
        nfp: int - number of field periods
        mpol: int - poloidal mode count
        ntor: int - toroidal mode count
    """
    ds = nc.Dataset(wout_path)

    results = {
        "iotaf": ds.variables["iotaf"][:].tolist(),
        "presf": ds.variables["presf"][:].tolist(),
        "betatot": float(ds.variables["betatotal"][:]),
        "DMerc": ds.variables["DMerc"][:].tolist(),
        "Vp": ds.variables["vp"][:].tolist(),
        "ier_flag": int(ds.variables["ier_flag"][:]),
        "fsql": float(ds.variables["fsql"][:]),
        "ns": int(len(ds.variables["iotaf"][:])),
        "nfp": int(ds.variables["nfp"][:]),
        "mpol": int(ds.variables["mpol"][:]),
        "ntor": int(ds.variables["ntor"][:]),
    }

    ds.close()
    return results


def run_boozer_analysis(wout_path: str, mboz: int = 32,
                        nboz: int = 32) -> dict:
    """
    Run BOOZ_XFORM via simsopt.mhd.Boozer.

    Args:
        wout_path: Path to VMEC wout file.
        mboz: Number of poloidal Boozer modes.
        nboz: Number of toroidal Boozer modes.

    Returns dict with:
        bmnc_b: dict mapping "(m,n)" → amplitude for dominant modes
        epsilon_eff: list[float] - effective ripple per surface
    """
    from simsopt.mhd import Vmec, Boozer

    vmec = Vmec(wout_path)

    # Boozer constructor takes mpol/ntor
    boozer = Boozer(vmec, mpol=mboz, ntor=nboz)

    # Register surfaces before running
    ns = vmec.wout.ns
    s_surfaces = np.linspace(0, 1, ns)[1:]  # exclude magnetic axis
    boozer.register(s_surfaces)
    boozer.run()

    # Extract Boozer spectrum on LCFS
    bmnc = boozer.bx.bmnc_b
    xm_b = boozer.bx.xm_b
    xn_b = boozer.bx.xn_b

    # Find dominant modes (top 10 by amplitude on LCFS)
    lcfs_amplitudes = np.abs(bmnc[-1, :])
    top_indices = np.argsort(lcfs_amplitudes)[-10:][::-1]
    dominant_modes = {}
    for idx in top_indices:
        m, n = int(xm_b[idx]), int(xn_b[idx])
        dominant_modes[f"({m},{n})"] = float(bmnc[-1, idx])

    # Effective ripple from BOOZ_XFORM
    epsilon_eff = []
    if hasattr(boozer.bx, "epsilon_eff"):
        epsilon_eff = boozer.bx.epsilon_eff.tolist()

    return {
        "bmnc_b": dominant_modes,
        "epsilon_eff": epsilon_eff,
    }


def save_results(results: dict, metadata: dict, output_dir: str,
                 wout_path: str = None):
    """
    Save combined VMEC results and metadata as JSON.

    Creates output_dir/results.json with schema:
    {
        "metadata": {...},
        "vmec": {...},
        "boozer": {...} (if present in results),
        "wout_path": str
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    vmec_results = {k: v for k, v in results.items()
                    if k not in ("bmnc_b", "epsilon_eff")}
    vmec_results["converged"] = results.get("ier_flag", -1) == 0

    output = {
        "metadata": metadata,
        "vmec": vmec_results,
    }

    if wout_path:
        output["wout_path"] = wout_path

    # Include Boozer results if present
    if "bmnc_b" in results or "epsilon_eff" in results:
        output["boozer"] = {
            "bmnc_b": results.get("bmnc_b", {}),
            "epsilon_eff": results.get("epsilon_eff", []),
        }

    # Check beta deviation (REQ-VMEC-002)
    beta_target = metadata.get("beta_target", 0.02)
    betatot = results.get("betatot", 0)
    if betatot > 0 and abs(betatot - beta_target) / beta_target > 0.2:
        import warnings
        warnings.warn(
            f"Achieved beta {betatot:.4f} deviates >20% from target {beta_target}"
        )

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
