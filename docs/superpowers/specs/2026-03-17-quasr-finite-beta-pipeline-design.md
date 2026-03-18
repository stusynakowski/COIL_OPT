# QUASR Finite-Beta Pipeline Design

**Date:** 2026-03-17
**Status:** Draft
**Working directory:** `tme/`

## Overview

Load 3 QA stellarator equilibria from the QUASR database into SIMSOPT, visualize them, and run fixed-boundary VMEC to obtain finite-beta (2%) equilibria with reactor-like profiles. Extract MHD-relevant quantities for future ML surrogate work.

## Requirements

- **REQ-INSTALL-001:** Install VMEC2000 and BOOZ_XFORM from hiddenSymmetries forks. Rebuild SIMSOPT with VMEC and Boozer support.
- **REQ-INSTALL-002:** Verify installation: `from simsopt.mhd import Vmec, Boozer` imports successfully and a trivial VMEC run completes.
- **REQ-LOAD-001:** Load QUASR equilibria from serialized SIMSOPT JSON files via `simsopt.load()`, returning surface, coils (with currents), and BiotSavart field object.
- **REQ-LOAD-002:** Maintain structured metadata per equilibrium (model_id, symmetry_type, nfp, source_url, beta_target, profile_type).
- **REQ-VIZ-001:** Visualize vacuum stage: 3D boundary surface + coils + B·n colormap (plotly).
- **REQ-VIZ-002:** Static matplotlib cross-sections at toroidal angles φ = 0, π/(2·nfp), π/nfp.
- **REQ-VIZ-003:** Post-VMEC static plots: iota(s), p(s), Mercier(s), magnetic well V''(s), flux surface cross-sections.
- **REQ-VIZ-004:** Post-VMEC 3D: finite-beta LCFS colored by |B|.
- **REQ-VMEC-001:** Prepare fixed-boundary VMEC input with reactor-like profiles for QA configurations. Use existing `input.*` files as base templates and inject profiles.
- **REQ-VMEC-002:** Pressure profile: p(s) = p0 · (1-s)². Estimate p0 analytically from target beta using p0 ≈ β_target · B0² / (2μ₀ · ⟨(1-s)²⟩_V). Verify achieved beta from wout; if >20% deviation from target, log a warning.
- **REQ-VMEC-003:** Current profile (QA): NCURR=1, prescribed CURTOR and AC coefficients approximating bootstrap contribution. Estimate CURTOR from the Shaing-Callen large-aspect-ratio bootstrap formula or use ~10-20% of the vacuum rotational transform as a guide for the enclosed current.
- **REQ-VMEC-004:** Current profile (QH, future): NCURR=0, fixed iota from vacuum geometry.
- **REQ-VMEC-005:** Resolution: MPOL=10, NTOR=10, NS_ARRAY=[16, 49].
- **REQ-VMEC-006:** Run VMEC via `simsopt.mhd.Vmec`, verify convergence (check `ier_flag == 0` and `fsql < 1e-8`).
- **REQ-ANALYSIS-001:** Extract from wout: iota profile, pressure profile, volume-averaged beta, Mercier criterion (DMerc), magnetic well (Vp).
- **REQ-ANALYSIS-002:** Run BOOZ_XFORM via `simsopt.mhd.Boozer` to obtain Boozer spectrum and effective ripple (ε_eff).
- **REQ-ANALYSIS-003:** Save all results + metadata as structured JSON per equilibrium with defined schema (see Output Schema section).
- **REQ-PIPE-001:** Pipeline must handle per-equilibrium failures gracefully: log the error, skip the failed equilibrium, and continue processing remaining configurations.

## Equilibria

| Model ID | Symmetry | NFP | Source | Profile Strategy |
|----------|----------|-----|--------|------------------|
| 0010273  | QA       | 2   | quasr.flatironinstitute.org/model/0010273 | bootstrap-like |
| 0019548  | QA       | 2   | quasr.flatironinstitute.org/model/0019548 | bootstrap-like |
| 0358936  | QA       | 2   | quasr.flatironinstitute.org/model/0358936 | bootstrap-like |

Data provenance: equilibria downloaded from QUASR database. Source URLs stored in `tme/QUASR_eq/URLs.txt`.

## Architecture

Modular Python scripts + thin orchestration notebook.

```
tme/
├── QUASR_eq/                  # existing equilibrium data (input.* + serial*.json)
├── load_quasr.py              # REQ-LOAD-001, REQ-LOAD-002
├── vmec_runner.py             # REQ-VMEC-001 through REQ-VMEC-006
├── visualize.py               # REQ-VIZ-001 through REQ-VIZ-004
├── analysis.py                # REQ-ANALYSIS-001 through REQ-ANALYSIS-003
├── run_all.ipynb              # orchestration notebook (REQ-PIPE-001)
└── output/                    # wout files, plots, metadata JSONs
    ├── 0010273/
    ├── 0019548/
    └── 0358936/
```

## Module Design

### `load_quasr.py`

```python
def load_quasr_equilibrium(json_path: str, symmetry_type: str = "QA") -> dict:
    """
    Load QUASR equilibrium from SIMSOPT serialized JSON.

    Uses simsopt.load() to deserialize. Extracts surfaces and coils from
    the object graph. Constructs BiotSavart field from coils.

    Args:
        json_path: Path to serial*.json file
        symmetry_type: "QA", "QH", or "QP"

    Returns:
        {
            "surface": SurfaceXYZTensorFourier (first/outermost surface),
            "coils": list[Coil] (includes currents),
            "bs": BiotSavart (field from coils, ready to evaluate),
            "metadata": {
                "model_id": str,        # e.g. "0010273"
                "symmetry_type": str,   # "QA", "QH", "QP"
                "nfp": int,
                "source_url": str,
                "beta_target": 0.02,
                "profile_type": str     # "bootstrap" if QA, "fixed_iota" if QH
            }
        }
    """
```

- Model ID parsed from filename (`serial0010273.json` → `"0010273"`)
- Profile type determined by symmetry: QA → `"bootstrap"`, QH/QP → `"fixed_iota"`

### `vmec_runner.py`

```python
def prepare_vmec_input(surface, metadata, output_dir: str,
                       input_template: str = None) -> str:
    """
    Create VMEC input file with reactor-like profiles.

    Strategy: Start from the existing input.* file as a template (preserves
    boundary RBC/ZBS coefficients), then inject pressure and current profiles.
    This avoids the SurfaceXYZTensorFourier → RBC/ZBS conversion entirely.

    Pressure: AM coefficients for p(s) = p0 * (1-s)^2, with p0 estimated
    from target beta. PRES_SCALE used for final normalization.

    Current (QA): NCURR=1, CURTOR estimated from bootstrap scaling,
    AC coefficients for a profile peaked off-axis.

    Current (QH): NCURR=0, AI coefficients from vacuum iota.

    Returns path to written input file.
    """

def run_vmec(input_path: str) -> str:
    """
    Run VMEC via simsopt.mhd.Vmec.
    Checks ier_flag == 0 and fsql < 1e-8.
    Returns path to wout file.
    Raises RuntimeError on non-convergence with diagnostic info.
    """
```

### `visualize.py`

```python
def plot_vacuum_3d(surface, coils, bs: BiotSavart) -> go.Figure:
    """3D plotly figure: surface colored by B·n + coils."""

def plot_cross_sections(surface, nfp: int) -> plt.Figure:
    """Static matplotlib cross-sections at φ = 0, π/(2·nfp), π/nfp."""

def plot_vmec_profiles(wout_path: str) -> plt.Figure:
    """Iota, pressure, Mercier, magnetic well vs normalized flux. 2x2 subplot."""

def plot_flux_surfaces(wout_path: str) -> plt.Figure:
    """Cross-section flux surfaces at multiple toroidal angles."""

def plot_finite_beta_3d(wout_path: str) -> go.Figure:
    """3D LCFS colored by |B|."""
```

All plotting functions accept an optional `save_path` parameter for batch/headless use.

### `analysis.py`

```python
def extract_vmec_results(wout_path: str) -> dict:
    """
    Extract key quantities from wout NetCDF.
    Returns dict with keys: iotaf, presf, betatot, DMerc, Vp,
    ier_flag, fsql, ns, nfp, mpol, ntor.
    Array values stored as lists for JSON serialization.
    """

def run_boozer_analysis(wout_path: str, mboz: int = 32, nboz: int = 32) -> dict:
    """
    Run BOOZ_XFORM via simsopt.mhd.Boozer.
    Returns dict with keys: bmnc_b (Boozer spectrum), epsilon_eff.
    """

def save_results(results: dict, metadata: dict, output_dir: str):
    """Save combined results + metadata as JSON to output_dir/results.json."""
```

### `run_all.ipynb`

Thin notebook that loops over the 3 equilibria:
1. Load each with `load_quasr_equilibrium()`
2. Visualize vacuum (3D + cross-sections + B·n)
3. Prepare and run VMEC (try/except per equilibrium, log failures, continue)
4. Extract results + Boozer analysis
5. Visualize finite-beta results
6. Save everything to `output/{model_id}/`

## Output Schema

Per-equilibrium `results.json`:
```json
{
    "metadata": {
        "model_id": "0010273",
        "symmetry_type": "QA",
        "nfp": 2,
        "source_url": "https://quasr.flatironinstitute.org/model/0010273",
        "beta_target": 0.02,
        "profile_type": "bootstrap"
    },
    "vmec": {
        "converged": true,
        "ier_flag": 0,
        "fsql": 1e-12,
        "betatot": 0.0198,
        "iotaf": [0.15, "... (ns values)"],
        "presf": [1000.0, "... (ns values)"],
        "DMerc": [0.01, "... (ns values)"],
        "Vp": [0.5, "... (ns values)"]
    },
    "boozer": {
        "epsilon_eff": [0.001, "... (per surface)"],
        "dominant_modes": {"(0,0)": 1.0, "(1,0)": 0.95}
    },
    "wout_path": "output/0010273/wout_0010273.nc"
}
```

## Installation Plan

1. Check/install prerequisites: `gfortran`, MPI (`mpich` or `openmpi`), `cmake`, `netcdf-fortran`
2. Clone and build VMEC2000 from `github.com/hiddenSymmetries/VMEC2000`
3. Clone and build BOOZ_XFORM from `github.com/hiddenSymmetries/BOOZ_XFORM`
4. Reinstall SIMSOPT from source with VMEC + Boozer support
5. Verify: `from simsopt.mhd import Vmec, Boozer` succeeds and a trivial run completes

## Future Work (Out of Scope)

- Free-boundary VMEC with coil re-optimization (Stage II against finite-beta boundary)
- Self-consistent bootstrap current iteration (VMEC ↔ neoclassical transport)
- Q value calculation and ML surrogate training
- QH equilibrium handling (fixed iota profiles)
