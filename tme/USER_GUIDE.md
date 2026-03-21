# QUASR Finite-Beta Pipeline — User Guide

Pipeline for loading QUASR vacuum stellarator equilibria, running fixed-boundary VMEC at finite beta, and extracting MHD stability quantities.

## Quick Start

Activate the environment and run the notebook:

```bash
conda activate coil_opt_spring_2026
cd tme/
jupyter notebook run_all.ipynb
```

Or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute run_all.ipynb --output run_all_executed.ipynb
```

## Environment

- **Conda env:** `coil_opt_spring_2026`
- **Key packages:** SIMSOPT (v1.10.3.dev, from source with VMEC support), VMEC2000 (hiddenSymmetries fork), booz_xform (v0.0.9), f90nml, netCDF4, plotly, matplotlib

## Modules

### `load_quasr.py` — Load QUASR Equilibria

```python
from load_quasr import load_quasr_equilibrium

eq = load_quasr_equilibrium(
    "QUASR_eq/serial0010273.json",
    symmetry_type="QA",   # "QA", "QH", or "QP"
    beta_target=0.02,     # target volume-averaged beta
)

eq["surface"]   # SurfaceXYZTensorFourier boundary
eq["coils"]     # list of Coil objects
eq["bs"]        # BiotSavart field from coils
eq["metadata"]  # dict: model_id, nfp, symmetry_type, B0, R0, beta_target, profile_type
```

- Parses model ID from filename (`serial0010273.json` -> `"0010273"`)
- Computes external coil field B0 and major radius R0 from the surface/coils
- Sets `profile_type` based on symmetry: QA -> `"bootstrap"`, QH/QP -> `"fixed_iota"`

### `vmec_runner.py` — Run VMEC

Three main functions:

**`prepare_vmec_input(surface, metadata, output_dir, input_template)`**

Creates a VMEC input file by modifying a template. Injects:
- Pressure profile: p(s) = p0 * (1-s)^2 via AM = [1, -2, 1, 0, ...]
- For QA (bootstrap): NCURR=1, toroidal current with j(s) ~ (1-s)
- For QH/QP (fixed iota): NCURR=0
- Resolution: NS_ARRAY=[16,49], MPOL=NTOR=10

```python
from vmec_runner import prepare_vmec_input, run_vmec

input_path = prepare_vmec_input(
    eq["surface"], eq["metadata"],
    output_dir="output/0010273",
    input_template="QUASR_eq/input.0010273",
)
wout_path = run_vmec(input_path)
```

**`run_vmec(input_path)`**

Runs VMEC via SIMSOPT. Returns wout path. Raises `RuntimeError` if ier_flag != 0.

**`run_vmec_to_target_beta(surface, metadata, output_dir, input_template)`** (recommended)

Iteratively adjusts PRES_SCALE to hit the target beta. Runs a calibration pass with the initial estimate, measures achieved beta, scales PRES_SCALE linearly, and re-runs. Typically converges in 2 VMEC runs.

```python
from vmec_runner import run_vmec_to_target_beta

wout_path = run_vmec_to_target_beta(
    eq["surface"], eq["metadata"],
    output_dir="output/0010273",
    input_template="QUASR_eq/input.0010273",
)
```

This is the recommended way to run VMEC because the initial pressure estimate (based on external coil B0) undershoots significantly. The iterative approach achieves beta within ~10% of target.

### `visualize.py` — Plotting

| Function | Output | Description |
|---|---|---|
| `plot_vacuum_3d(surface, coils, bs)` | plotly Figure | 3D surface colored by B*n + coil curves |
| `plot_cross_sections(surface, nfp)` | matplotlib Figure | Boundary cross-sections at 3 toroidal angles |
| `plot_vmec_profiles(wout_path)` | matplotlib Figure | 2x2: iota, pressure, Mercier, V'(s) |
| `plot_flux_surfaces(wout_path)` | matplotlib Figure | Nested flux surfaces at 3 toroidal angles |
| `plot_finite_beta_3d(wout_path)` | plotly Figure | 3D LCFS colored by |B| |

All functions accept an optional `save_path` argument to write the figure to disk.

### `analysis.py` — MHD Analysis

**`extract_vmec_results(wout_path)`**

Returns a dict with: `iotaf`, `presf`, `betatot`, `DMerc`, `Vp`, `ier_flag`, `fsql`, `ns`, `nfp`, `mpol`, `ntor`.

**`run_boozer_analysis(wout_path, mboz=32, nboz=32)`**

Runs BOOZ_XFORM. Returns `bmnc_b` (top 10 Boozer mode amplitudes on LCFS) and `epsilon_eff` (effective ripple per surface).

**`save_results(results, metadata, output_dir, wout_path=None)`**

Saves `results.json` with structure:

```json
{
  "metadata": {"model_id": "0010273", "symmetry_type": "QA", "nfp": 2, ...},
  "vmec": {"betatot": 0.021, "iotaf": [...], "presf": [...], "DMerc": [...], "converged": true, ...},
  "boozer": {"bmnc_b": {"(0,0)": 5.14, ...}, "epsilon_eff": [...]},
  "wout_path": "..."
}
```

## Data Layout

```
tme/
  QUASR_eq/               # Input data
    serial0010273.json     # SIMSOPT-serialized surface + coils
    input.0010273          # VMEC input template (boundary RBC/ZBS)
    serial0019548.json
    input.0019548
    serial0358936.json
    input.0358936
  output/                  # Pipeline outputs
    0010273/
      input.0010273        # Generated VMEC input (with profiles)
      results.json         # Structured MHD results
    0019548/
    0358936/
  load_quasr.py
  vmec_runner.py
  visualize.py
  analysis.py
  run_all.ipynb            # Orchestration notebook
  run_all_executed.ipynb   # Executed notebook with outputs
  tests/                   # 21 unit tests
```

## Current Equilibria

| Model | Type | NFP | Beta achieved | Status |
|---|---|---|---|---|
| 0010273 | QA | 2 | ~2.1% | Converged |
| 0019548 | QA | 2 | ~2.0% | Converged |
| 0358936 | QA | 2 | ~2.1% | Converged |

## Adding New Equilibria

1. Download the SIMSOPT JSON and VMEC input template from QUASR
2. Place them in `QUASR_eq/` as `serial{MODEL_ID}.json` and `input.{MODEL_ID}`
3. Add an entry to the `EQUILIBRIA` dict in `run_all.ipynb`:
   ```python
   EQUILIBRIA = {
       "NEW_ID": {"symmetry_type": "QA"},  # or "QH"
   }
   ```
4. Run the notebook. QH equilibria automatically use fixed-iota profiles (NCURR=0) instead of bootstrap current.

## Tests

```bash
cd tme/
python -m pytest tests/ -v
```

21 tests covering all modules including actual VMEC runs (~90s total).

## Known Limitations

- **Beta targeting uses linear scaling:** accurate at low beta (<5%) but may need more iterations at high beta where nonlinear effects matter.
- **Profile shapes are fixed:** p(s) = p0*(1-s)^2 and j(s) ~ (1-s). Not self-consistent bootstrap current.
- **Fixed-boundary only:** no coil optimization or free-boundary VMEC.
- **VMEC wout files land in the working directory** (SIMSOPT convention), not in `output/`. The path is recorded in `results.json`.
