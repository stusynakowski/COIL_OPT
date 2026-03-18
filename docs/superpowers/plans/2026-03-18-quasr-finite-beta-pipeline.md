# QUASR Finite-Beta Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline to load 3 QA QUASR equilibria into SIMSOPT, visualize them, run fixed-boundary VMEC at 2% beta with reactor-like profiles, and extract MHD quantities.

**Architecture:** Modular Python scripts in `tme/` (load_quasr.py, vmec_runner.py, visualize.py, analysis.py) orchestrated by a thin Jupyter notebook (`run_all.ipynb`). Each module has focused responsibility and is independently testable.

**Tech Stack:** SIMSOPT (with VMEC + Boozer), VMEC2000 (hiddenSymmetries fork), BOOZ_XFORM (hiddenSymmetries fork), matplotlib, plotly, netCDF4, numpy, scipy, pandas, f90nml.

**Spec:** `docs/superpowers/specs/2026-03-17-quasr-finite-beta-pipeline-design.md`

---

## File Structure

```
tme/
├── QUASR_eq/                  # existing (untouched)
│   ├── input.0010273
│   ├── input.0019548
│   ├── input.0358936
│   ├── serial0010273.json
│   ├── serial0019548.json
│   ├── serial0358936.json
│   └── URLs.txt
├── load_quasr.py              # load JSON → SIMSOPT objects + metadata
├── vmec_runner.py             # prepare VMEC input, run VMEC
├── visualize.py               # vacuum + finite-beta plotting
├── analysis.py                # extract wout quantities, run Boozer
├── run_all.ipynb              # orchestration notebook
├── tests/
│   ├── test_load_quasr.py
│   ├── test_vmec_runner.py
│   ├── test_visualize.py
│   └── test_analysis.py
└── output/                    # created at runtime
    ├── 0010273/
    ├── 0019548/
    └── 0358936/
```

---

## Chunk 1: Installation

### Task 1: Install System Dependencies

**Context:** WSL2 Ubuntu, miniconda3, Python 3.13.11. No Fortran compiler, MPI, cmake, or netCDF currently installed.

- [ ] **Step 1: Install build toolchain via apt**

```bash
sudo apt-get update
sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin cmake libnetcdf-dev libnetcdff-dev liblapack-dev libblas-dev git
```

- [ ] **Step 2: Verify compilers and tools**

```bash
gfortran --version   # expect GNU Fortran 11+
mpirun --version     # expect Open MPI
cmake --version      # expect 3.16+
nc-config --all      # expect netCDF paths
```

- [ ] **Step 3: Install mpi4py and netCDF4 Python packages**

```bash
pip install mpi4py netCDF4 f90nml
```

`f90nml` is for reading/writing Fortran namelists (VMEC input files).

- [ ] **Step 4: Verify Python packages**

```bash
python -c "from mpi4py import MPI; print('mpi4py OK')"
python -c "import netCDF4; print('netCDF4 OK')"
python -c "import f90nml; print('f90nml OK')"
```

- [ ] **Step 5: Commit notes (no code to commit yet)**

No commit needed — system-level installs only.

---

### Task 2: Build VMEC2000 from hiddenSymmetries

**Context:** The hiddenSymmetries VMEC2000 fork builds with cmake. It produces a shared library that SIMSOPT links against.

- [ ] **Step 1: Clone VMEC2000**

```bash
cd /home/telder1
git clone https://github.com/hiddenSymmetries/VMEC2000.git
cd VMEC2000
```

- [ ] **Step 2: Build with cmake**

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local
make -j$(nproc)
make install
```

If cmake fails to find netCDF-Fortran, try:
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local -DNETCDF_F_INCLUDE_DIRS=/usr/include -DNETCDF_F_LIBRARIES=/usr/lib/x86_64-linux-gnu/libnetcdff.so
```

- [ ] **Step 3: Verify the VMEC library was built**

```bash
ls $HOME/local/lib/libvmec*   # expect libvmec.so or libvmec.a
```

- [ ] **Step 4: Set environment variables**

```bash
export CMAKE_PREFIX_PATH=$HOME/local:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
```

Add these to `~/.bashrc` for persistence.

---

### Task 3: Install BOOZ_XFORM Python Package

**Context:** The hiddenSymmetries BOOZ_XFORM is a Python package with pybind11 bindings. It must be pip-installed (not cmake-built) to produce the `booz_xform` Python module that simsopt imports.

- [ ] **Step 1: Install booz_xform via pip**

```bash
pip install booz_xform
```

If this fails (e.g., build issues), install from source:
```bash
cd /home/telder1
git clone https://github.com/hiddenSymmetries/BOOZ_XFORM.git
cd BOOZ_XFORM
pip install -e .
```

- [ ] **Step 2: Verify**

```bash
python -c "import booz_xform; print('booz_xform OK, version:', booz_xform.__version__)"
```

---

### Task 4: Reinstall SIMSOPT with VMEC + Boozer Support

**Context:** Current simsopt 1.10.6 is pure-Python pip install. Need to rebuild from source with compiled extensions.

- [ ] **Step 1: Uninstall current simsopt**

```bash
pip uninstall simsopt -y
```

- [ ] **Step 2: Clone and install simsopt from source**

```bash
cd /home/telder1
git clone https://github.com/hiddenSymmetries/simsopt.git
cd simsopt
pip install -e ".[vmec]"
```

The cmake-based build should automatically find VMEC and BOOZ_XFORM via `CMAKE_PREFIX_PATH`.

- [ ] **Step 3: Verify VMEC support (REQ-INSTALL-002)**

```python
python -c "
from simsopt.mhd import Vmec
print('Vmec import OK')
"
```

- [ ] **Step 4: Verify Boozer support**

```python
python -c "
from simsopt.mhd import Boozer
print('Boozer import OK')
"
```

- [ ] **Step 5: Run a trivial VMEC test (REQ-INSTALL-002)**

```python
python -c "
from simsopt.mhd import Vmec
v = Vmec('/home/telder1/COIL_OPT/tme/QUASR_eq/input.0010273')
v.run()
print('VMEC ran successfully, wout at:', v.output_file)
"
```

This confirms the full toolchain works end-to-end.

- [ ] **Step 6: Commit environment notes**

No code to commit, but if any config files were created, commit them.

---

## Chunk 2: Data Loading Module

### Task 5: Write test_load_quasr.py

**Files:**
- Create: `tme/tests/test_load_quasr.py`

- [ ] **Step 1: Create tests directory**

```bash
mkdir -p /home/telder1/COIL_OPT/tme/tests
```

- [ ] **Step 2: Write failing tests**

```python
# tme/tests/test_load_quasr.py
"""Tests for QUASR equilibrium loading. Maps to REQ-LOAD-001, REQ-LOAD-002."""
import os
import pytest

QUASR_DIR = os.path.join(os.path.dirname(__file__), "..", "QUASR_eq")
SERIAL_FILE = os.path.join(QUASR_DIR, "serial0010273.json")


def test_load_returns_required_keys():
    """TEST-LOAD-001: load returns surface, coils, bs, metadata."""
    from load_quasr import load_quasr_equilibrium

    result = load_quasr_equilibrium(SERIAL_FILE)
    assert "surface" in result
    assert "coils" in result
    assert "bs" in result
    assert "metadata" in result


def test_surface_type():
    """TEST-LOAD-002: surface is a SIMSOPT surface object."""
    from load_quasr import load_quasr_equilibrium

    result = load_quasr_equilibrium(SERIAL_FILE)
    # Should be a surface with gamma() method
    gamma = result["surface"].gamma()
    assert gamma.shape[2] == 3  # (nphi, ntheta, 3)


def test_coils_nonempty():
    """TEST-LOAD-003: coils list is non-empty."""
    from load_quasr import load_quasr_equilibrium

    result = load_quasr_equilibrium(SERIAL_FILE)
    assert len(result["coils"]) > 0


def test_biot_savart_evaluates():
    """TEST-LOAD-004: BiotSavart can evaluate B field."""
    from load_quasr import load_quasr_equilibrium
    import numpy as np

    result = load_quasr_equilibrium(SERIAL_FILE)
    bs = result["bs"]
    # Evaluate at surface points
    surface = result["surface"]
    bs.set_points(surface.gamma().reshape(-1, 3))
    B = bs.B()
    assert B.shape[1] == 3
    assert np.all(np.isfinite(B))


def test_metadata_fields():
    """TEST-LOAD-005: metadata has all required fields (REQ-LOAD-002)."""
    from load_quasr import load_quasr_equilibrium

    result = load_quasr_equilibrium(SERIAL_FILE)
    meta = result["metadata"]
    assert meta["model_id"] == "0010273"
    assert meta["symmetry_type"] == "QA"
    assert meta["nfp"] == 2
    assert "source_url" in meta
    assert meta["beta_target"] == 0.02
    assert meta["profile_type"] == "bootstrap"


def test_metadata_qh_profile():
    """TEST-LOAD-006: QH symmetry type gives fixed_iota profile."""
    from load_quasr import load_quasr_equilibrium

    result = load_quasr_equilibrium(SERIAL_FILE, symmetry_type="QH")
    assert result["metadata"]["profile_type"] == "fixed_iota"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_load_quasr.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'load_quasr'`

- [ ] **Step 4: Commit test file**

```bash
git add tme/tests/test_load_quasr.py
git commit -m "test: add tests for QUASR equilibrium loading (REQ-LOAD-001, REQ-LOAD-002)"
```

---

### Task 6: Implement load_quasr.py

**Files:**
- Create: `tme/load_quasr.py`

- [ ] **Step 1: Write implementation**

```python
# tme/load_quasr.py
"""Load QUASR equilibria from SIMSOPT serialized JSON files.

Implements REQ-LOAD-001 and REQ-LOAD-002.
"""
import os
import re
import simsopt
from simsopt.field import BiotSavart, Coil
from simsopt.geo import SurfaceXYZTensorFourier

QUASR_BASE_URL = "https://quasr.flatironinstitute.org/model"

PROFILE_TYPE_MAP = {
    "QA": "bootstrap",
    "QH": "fixed_iota",
    "QP": "fixed_iota",
}


def load_quasr_equilibrium(json_path: str, symmetry_type: str = "QA",
                           beta_target: float = 0.02) -> dict:
    """
    Load a QUASR equilibrium from a SIMSOPT serialized JSON file.

    Args:
        json_path: Path to serial*.json file.
        symmetry_type: One of "QA", "QH", "QP".
        beta_target: Target volume-averaged beta.

    Returns:
        dict with keys:
            surface: SurfaceXYZTensorFourier (outermost/boundary surface)
            coils: list[Coil]
            bs: BiotSavart field object
            metadata: dict with model_id, symmetry_type, nfp, source_url,
                      beta_target, profile_type
    """
    objs = simsopt.load(json_path)

    # simsopt.load() returns a list of two lists:
    # objs[0] = list of SurfaceXYZTensorFourier objects
    # objs[1] = list of Coil objects
    surfaces = objs[0]
    coils_list = objs[1]

    if not surfaces:
        raise ValueError(f"No surfaces found in {json_path}")
    if not coils_list:
        raise ValueError(f"No coils found in {json_path}")

    # Use the first surface as the boundary
    surface = surfaces[0]
    coils = coils_list

    # Build BiotSavart field
    bs = BiotSavart(coils)

    # Parse model ID from filename: serial0010273.json → "0010273"
    basename = os.path.basename(json_path)
    match = re.search(r"serial(\d+)\.json", basename)
    model_id = match.group(1) if match else basename

    metadata = {
        "model_id": model_id,
        "symmetry_type": symmetry_type,
        "nfp": int(surface.nfp),
        "source_url": f"{QUASR_BASE_URL}/{model_id}",
        "beta_target": beta_target,
        "profile_type": PROFILE_TYPE_MAP.get(symmetry_type, "fixed_iota"),
    }

    return {
        "surface": surface,
        "coils": coils,
        "bs": bs,
        "metadata": metadata,
    }
```

- [ ] **Step 2: Run tests**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_load_quasr.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tme/load_quasr.py
git commit -m "feat: implement QUASR equilibrium loader (REQ-LOAD-001, REQ-LOAD-002)"
```

---

## Chunk 3: Visualization Module

### Task 7: Write test_visualize.py

**Files:**
- Create: `tme/tests/test_visualize.py`

- [ ] **Step 1: Write failing tests**

```python
# tme/tests/test_visualize.py
"""Tests for visualization functions. Maps to REQ-VIZ-001 through REQ-VIZ-004."""
import os
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

QUASR_DIR = os.path.join(os.path.dirname(__file__), "..", "QUASR_eq")
SERIAL_FILE = os.path.join(QUASR_DIR, "serial0010273.json")


@pytest.fixture
def loaded_equilibrium():
    from load_quasr import load_quasr_equilibrium
    return load_quasr_equilibrium(SERIAL_FILE)


def test_plot_vacuum_3d_returns_figure(loaded_equilibrium):
    """TEST-VIZ-001: vacuum 3D plot returns a plotly Figure."""
    import plotly.graph_objects as go
    from visualize import plot_vacuum_3d

    eq = loaded_equilibrium
    fig = plot_vacuum_3d(eq["surface"], eq["coils"], eq["bs"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # has at least one trace


def test_plot_cross_sections_returns_figure(loaded_equilibrium):
    """TEST-VIZ-002: cross-section plot returns matplotlib Figure."""
    import matplotlib.pyplot as plt
    from visualize import plot_cross_sections

    fig = plot_cross_sections(loaded_equilibrium["surface"], nfp=2)
    assert isinstance(fig, plt.Figure)


def test_plot_vacuum_3d_save(loaded_equilibrium, tmp_path):
    """TEST-VIZ-003: vacuum 3D plot can save to file."""
    from visualize import plot_vacuum_3d

    eq = loaded_equilibrium
    save_path = str(tmp_path / "vacuum_3d.html")
    fig = plot_vacuum_3d(eq["surface"], eq["coils"], eq["bs"],
                         save_path=save_path)
    assert os.path.exists(save_path)


def test_plot_cross_sections_save(loaded_equilibrium, tmp_path):
    """TEST-VIZ-004: cross-section plot can save to file."""
    from visualize import plot_cross_sections

    save_path = str(tmp_path / "cross_sections.png")
    fig = plot_cross_sections(loaded_equilibrium["surface"], nfp=2,
                              save_path=save_path)
    assert os.path.exists(save_path)
```

Note: `test_vmec_profiles` and `test_finite_beta_3d` tests require wout files, so they will be added in the analysis chunk after VMEC runs are available.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_visualize.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'visualize'`

- [ ] **Step 3: Commit**

```bash
git add tme/tests/test_visualize.py
git commit -m "test: add tests for vacuum visualization (REQ-VIZ-001, REQ-VIZ-002)"
```

---

### Task 8: Implement visualize.py (vacuum functions)

**Files:**
- Create: `tme/visualize.py`

- [ ] **Step 1: Write implementation**

```python
# tme/visualize.py
"""Visualization functions for QUASR equilibria.

Implements REQ-VIZ-001 through REQ-VIZ-004.
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from simsopt.field import BiotSavart


def plot_vacuum_3d(surface, coils, bs: BiotSavart,
                   save_path: str = None) -> go.Figure:
    """
    3D plotly figure of boundary surface colored by B·n, with coils.

    REQ-VIZ-001.
    """
    # Evaluate B·n on the surface
    bs.set_points(surface.gamma().reshape(-1, 3))
    Bfield = bs.B().reshape(surface.gamma().shape)
    normal = surface.unitnormal()
    Bdotn = np.sum(Bfield * normal, axis=2)

    # Surface mesh
    gamma = surface.gamma()
    nphi, ntheta, _ = gamma.shape
    x = gamma[:, :, 0]
    y = gamma[:, :, 1]
    z = gamma[:, :, 2]

    fig = go.Figure()

    # Surface colored by B·n
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=Bdotn,
        colorscale="RdBu",
        cmid=0,
        colorbar=dict(title="B·n [T]"),
        name="Boundary",
    ))

    # Coils
    for i, coil in enumerate(coils):
        curve_data = coil.curve.gamma()
        # Close the curve
        curve_closed = np.vstack([curve_data, curve_data[0:1, :]])
        fig.add_trace(go.Scatter3d(
            x=curve_closed[:, 0],
            y=curve_closed[:, 1],
            z=curve_closed[:, 2],
            mode="lines",
            line=dict(color="black", width=3),
            name=f"Coil {i}",
            showlegend=(i == 0),
        ))

    fig.update_layout(
        title="Vacuum Boundary (B·n) + Coils",
        scene=dict(aspectmode="data"),
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_cross_sections(surface, nfp: int,
                        save_path: str = None) -> plt.Figure:
    """
    Static matplotlib cross-sections at φ = 0, π/(2·nfp), π/nfp.

    REQ-VIZ-002.
    """
    gamma = surface.gamma()
    nphi = gamma.shape[0]

    # Toroidal angles for cross-sections
    phi_targets = [0, np.pi / (2 * nfp), np.pi / nfp]
    phi_labels = ["φ = 0", f"φ = π/{2*nfp}", f"φ = π/{nfp}"]

    # Map target phi to nearest index
    # Surface quadpoints cover one field period [0, 1/nfp] (normalized)
    # or [0, 2π/nfp] in radians, depending on SIMSOPT version
    if hasattr(surface, 'quadpoints_phi'):
        qp = np.array(surface.quadpoints_phi)
        # quadpoints_phi is in [0, 1], convert to radians
        phi_grid = qp * 2 * np.pi
    else:
        phi_grid = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, phi_target, label in zip(axes, phi_targets, phi_labels):
        idx = np.argmin(np.abs(phi_grid - phi_target))
        R = np.sqrt(gamma[idx, :, 0]**2 + gamma[idx, :, 1]**2)
        Z = gamma[idx, :, 2]
        ax.plot(R, Z, "b-", linewidth=2)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Boundary Cross-Sections")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_vmec_profiles(wout_path: str,
                       save_path: str = None) -> plt.Figure:
    """
    Iota, pressure, Mercier, magnetic well vs normalized flux.

    REQ-VIZ-003. Requires wout NetCDF file from VMEC.
    """
    import netCDF4 as nc

    ds = nc.Dataset(wout_path)
    ns = ds.dimensions["radius"].size
    s = np.linspace(0, 1, ns)

    iotaf = ds.variables["iotaf"][:]
    presf = ds.variables["presf"][:]
    DMerc = ds.variables["DMerc"][:]
    Vp = ds.variables["vp"][:]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(s, iotaf, "b-", linewidth=2)
    axes[0, 0].set_ylabel("ι")
    axes[0, 0].set_title("Rotational Transform")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(s, presf, "r-", linewidth=2)
    axes[0, 1].set_ylabel("p [Pa]")
    axes[0, 1].set_title("Pressure Profile")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(s[1:], DMerc[1:], "g-", linewidth=2)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_ylabel("D_Mercier")
    axes[1, 0].set_title("Mercier Criterion (>0 stable)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(s, Vp, "m-", linewidth=2)
    axes[1, 1].set_ylabel("V'(s)")
    axes[1, 1].set_title("Magnetic Well")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("s (norm. flux)")

    fig.suptitle("VMEC Equilibrium Profiles")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    ds.close()
    return fig


def plot_flux_surfaces(wout_path: str,
                       save_path: str = None) -> plt.Figure:
    """
    Cross-section flux surfaces at multiple toroidal angles.

    REQ-VIZ-003.
    """
    import netCDF4 as nc

    ds = nc.Dataset(wout_path)
    rmnc = ds.variables["rmnc"][:]
    zmns = ds.variables["zmns"][:]
    xm = ds.variables["xm"][:]
    xn = ds.variables["xn"][:]
    nfp = int(ds.variables["nfp"][:])
    ns = rmnc.shape[0]
    ds.close()

    theta = np.linspace(0, 2 * np.pi, 100)
    phi_targets = [0, np.pi / (2 * nfp), np.pi / nfp]
    phi_labels = ["φ = 0", f"φ = π/{2*nfp}", f"φ = π/{nfp}"]

    # Plot every 5th surface
    surface_indices = np.arange(0, ns, max(1, ns // 10))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, phi_val, label in zip(axes, phi_targets, phi_labels):
        for si in surface_indices:
            R = np.zeros_like(theta)
            Z = np.zeros_like(theta)
            for j in range(len(xm)):
                angle = xm[j] * theta - xn[j] * phi_val
                R += rmnc[si, j] * np.cos(angle)
                Z += zmns[si, j] * np.sin(angle)
            ax.plot(R, Z, "b-", linewidth=0.8, alpha=0.7)

        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("VMEC Flux Surfaces")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_finite_beta_3d(wout_path: str,
                        save_path: str = None) -> go.Figure:
    """
    3D LCFS colored by |B|.

    REQ-VIZ-004.
    """
    import netCDF4 as nc

    ds = nc.Dataset(wout_path)
    rmnc = ds.variables["rmnc"][:]
    zmns = ds.variables["zmns"][:]
    bmnc = ds.variables["bmnc"][:]
    xm = ds.variables["xm"][:]
    xn = ds.variables["xn"][:]
    xm_nyq = ds.variables["xm_nyq"][:]
    xn_nyq = ds.variables["xn_nyq"][:]
    nfp = int(ds.variables["nfp"][:])
    ds.close()

    nphi, ntheta = 128, 64
    theta = np.linspace(0, 2 * np.pi, ntheta)
    phi = np.linspace(0, 2 * np.pi, nphi)
    theta2d, phi2d = np.meshgrid(theta, phi)

    # LCFS is last radial surface
    R = np.zeros_like(theta2d)
    Z = np.zeros_like(theta2d)
    for j in range(len(xm)):
        angle = xm[j] * theta2d - xn[j] * phi2d
        R += rmnc[-1, j] * np.cos(angle)
        Z += zmns[-1, j] * np.sin(angle)

    # |B| on LCFS
    modB = np.zeros_like(theta2d)
    for j in range(len(xm_nyq)):
        angle = xm_nyq[j] * theta2d - xn_nyq[j] * phi2d
        modB += bmnc[-1, j] * np.cos(angle)

    X = R * np.cos(phi2d)
    Y = R * np.sin(phi2d)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=modB,
        colorscale="Viridis",
        colorbar=dict(title="|B| [T]"),
    ))
    fig.update_layout(
        title="Finite-Beta LCFS colored by |B|",
        scene=dict(aspectmode="data"),
    )

    if save_path:
        fig.write_html(save_path)

    return fig
```

- [ ] **Step 2: Run tests**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_visualize.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tme/visualize.py
git commit -m "feat: implement vacuum + finite-beta visualization (REQ-VIZ-001 through REQ-VIZ-004)"
```

---

## Chunk 4: VMEC Runner Module

### Task 9: Write test_vmec_runner.py

**Files:**
- Create: `tme/tests/test_vmec_runner.py`

- [ ] **Step 1: Write tests**

Tests are split: input preparation tests work without VMEC, execution tests require VMEC.

```python
# tme/tests/test_vmec_runner.py
"""Tests for VMEC input preparation and execution.
Maps to REQ-VMEC-001 through REQ-VMEC-006."""
import os
import pytest
import f90nml

QUASR_DIR = os.path.join(os.path.dirname(__file__), "..", "QUASR_eq")
SERIAL_FILE = os.path.join(QUASR_DIR, "serial0010273.json")
INPUT_TEMPLATE = os.path.join(QUASR_DIR, "input.0010273")


@pytest.fixture
def loaded_equilibrium():
    from load_quasr import load_quasr_equilibrium
    return load_quasr_equilibrium(SERIAL_FILE)


@pytest.fixture
def vmec_input_path(loaded_equilibrium, tmp_path):
    from vmec_runner import prepare_vmec_input
    return prepare_vmec_input(
        loaded_equilibrium["surface"],
        loaded_equilibrium["metadata"],
        str(tmp_path),
        input_template=INPUT_TEMPLATE,
    )


def test_prepare_vmec_input_creates_file(vmec_input_path):
    """TEST-VMEC-001: prepare_vmec_input creates an input file."""
    assert os.path.exists(vmec_input_path)


def test_vmec_input_has_pressure(vmec_input_path):
    """TEST-VMEC-002: input file has non-zero pressure (REQ-VMEC-002)."""
    nml = f90nml.read(vmec_input_path)
    indata = nml["indata"]
    pres_scale = indata.get("pres_scale", 1.0)
    assert pres_scale > 0


def test_vmec_input_has_current_for_qa(vmec_input_path):
    """TEST-VMEC-003: QA config has NCURR=1 with CURTOR (REQ-VMEC-003)."""
    nml = f90nml.read(vmec_input_path)
    indata = nml["indata"]
    assert indata["ncurr"] == 1
    assert abs(indata.get("curtor", 0)) > 0


def test_vmec_input_resolution(vmec_input_path):
    """TEST-VMEC-004: resolution matches spec (REQ-VMEC-005)."""
    nml = f90nml.read(vmec_input_path)
    indata = nml["indata"]
    assert indata["mpol"] >= 10
    assert indata["ntor"] >= 10


def test_vmec_input_fixed_iota_for_qh(loaded_equilibrium, tmp_path):
    """TEST-VMEC-005: QH config gets NCURR=0 (REQ-VMEC-004)."""
    from vmec_runner import prepare_vmec_input

    meta = loaded_equilibrium["metadata"].copy()
    meta["symmetry_type"] = "QH"
    meta["profile_type"] = "fixed_iota"

    path = prepare_vmec_input(
        loaded_equilibrium["surface"],
        meta,
        str(tmp_path / "qh"),
        input_template=INPUT_TEMPLATE,
    )
    nml = f90nml.read(path)
    assert nml["indata"]["ncurr"] == 0


# --- Tests below require VMEC to be installed ---

def vmec_available():
    try:
        from simsopt.mhd import Vmec
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not vmec_available(), reason="VMEC not installed")
def test_run_vmec_produces_wout(vmec_input_path):
    """TEST-VMEC-006: VMEC runs and produces wout file (REQ-VMEC-006)."""
    from vmec_runner import run_vmec
    wout_path = run_vmec(vmec_input_path)
    assert os.path.exists(wout_path)


@pytest.mark.skipif(not vmec_available(), reason="VMEC not installed")
def test_run_vmec_converges(vmec_input_path):
    """TEST-VMEC-007: VMEC converges (ier_flag == 0) (REQ-VMEC-006)."""
    import netCDF4 as nc
    from vmec_runner import run_vmec
    wout_path = run_vmec(vmec_input_path)
    ds = nc.Dataset(wout_path)
    ier_flag = int(ds.variables["ier_flag"][:])
    ds.close()
    assert ier_flag == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_vmec_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vmec_runner'`

- [ ] **Step 3: Commit**

```bash
git add tme/tests/test_vmec_runner.py
git commit -m "test: add tests for VMEC input preparation and execution (REQ-VMEC-001 through REQ-VMEC-006)"
```

---

### Task 10: Implement vmec_runner.py

**Files:**
- Create: `tme/vmec_runner.py`

- [ ] **Step 1: Write implementation**

```python
# tme/vmec_runner.py
"""Prepare and run fixed-boundary VMEC equilibria.

Implements REQ-VMEC-001 through REQ-VMEC-006.
"""
import os
import shutil
import numpy as np
import f90nml


def estimate_p0(beta_target: float, B0: float = 1.0) -> float:
    """
    Estimate peak pressure for a target beta.

    For p(s) = p0 * (1-s)^2, the volume average <p> = p0/3.
    beta = 2 * mu0 * <p> / B0^2
    So p0 = 3 * beta * B0^2 / (2 * mu0).

    Note: This assumes uniform V'(s) (cylindrical approximation).
    Actual stellarator geometry will shift the result. The beta
    deviation check in save_results() (REQ-VMEC-002) catches
    cases where the approximation is too far off.
    """
    mu0 = 4e-7 * np.pi
    return 3 * beta_target * B0**2 / (2 * mu0)


def estimate_curtor(beta_target: float, iota_frac: float = 0.15,
                    R0: float = 1.0, B0: float = 1.0) -> float:
    """
    Estimate toroidal current for bootstrap-like contribution.

    Rough scaling: CURTOR ~ iota_frac * 2π * R0 * B0 / mu0
    where iota_frac is the fraction of iota from plasma current.
    """
    mu0 = 4e-7 * np.pi
    return iota_frac * 2 * np.pi * R0 * B0 / mu0


def prepare_vmec_input(surface, metadata: dict, output_dir: str,
                       input_template: str = None) -> str:
    """
    Create VMEC input file with reactor-like profiles.

    Uses existing input.* file as template (preserves boundary RBC/ZBS).
    Injects pressure and current profiles based on symmetry type.

    Args:
        surface: SIMSOPT surface object (used for B0/R0 estimates if needed).
        metadata: dict with symmetry_type, beta_target, model_id, profile_type.
        output_dir: Directory to write the input file.
        input_template: Path to existing VMEC input file to use as template.

    Returns:
        Path to the written input file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_id = metadata["model_id"]
    output_path = os.path.join(output_dir, f"input.{model_id}")

    if input_template is None:
        raise ValueError("input_template is required")

    # Read the template
    nml = f90nml.read(input_template)
    indata = nml["indata"]

    # Estimate B0 and R0 from the surface geometry
    gamma = surface.gamma()
    R_vals = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
    R0 = float(np.mean(R_vals))
    B0 = 1.0  # default for QUASR normalized equilibria

    beta_target = metadata["beta_target"]

    # --- Pressure profile: p(s) = p0 * (1 - s)^2 ---
    # VMEC AM array: pressure = PRES_SCALE * sum(AM(i) * s^i)
    # (1-s)^2 = 1 - 2s + s^2 → AM = [1, -2, 1, 0, ...]
    p0 = estimate_p0(beta_target, B0)
    indata["pres_scale"] = float(p0)
    indata["am"] = [1.0, -2.0, 1.0] + [0.0] * 8
    indata["am_aux_s"] = None  # clear aux arrays if present
    indata["am_aux_f"] = None

    # --- Resolution ---
    indata["mpol"] = 10
    indata["ntor"] = 10
    indata["ns_array"] = [16, 49]
    indata["ftol_array"] = [1e-8, 1e-12]
    indata["niter_array"] = [2000, 5000]

    # --- Current/iota profile ---
    if metadata.get("profile_type") == "bootstrap":
        # QA: prescribe toroidal current
        indata["ncurr"] = 1
        curtor = estimate_curtor(beta_target, iota_frac=0.15, R0=R0, B0=B0)
        indata["curtor"] = float(curtor)
        # Current profile shape: j(s) ~ (1 - s), peaked on axis
        # AC array for VMEC power series
        indata["ac"] = [1.0, -1.0] + [0.0] * 9
        # Remove any iota prescriptions
        indata["ai"] = None
        indata["piota_type"] = None
    else:
        # QH/QP: fixed iota
        indata["ncurr"] = 0
        indata["curtor"] = 0.0
        # Keep existing AI coefficients from template if present,
        # otherwise VMEC will use its defaults

    # --- Other VMEC settings ---
    indata["lfreeb"] = False  # fixed boundary
    indata["nstep"] = 200

    # Write the namelist, filtering out None values
    clean_indata = {k: v for k, v in indata.items() if v is not None}
    nml["indata"] = clean_indata

    # f90nml write
    nml.write(output_path, force=True)

    return output_path


def run_vmec(input_path: str) -> str:
    """
    Run VMEC via simsopt.mhd.Vmec.

    Args:
        input_path: Path to VMEC input file.

    Returns:
        Path to wout output file.

    Raises:
        RuntimeError: If VMEC does not converge.
    """
    from simsopt.mhd import Vmec

    vmec = Vmec(input_path)
    vmec.run()

    # Check convergence
    wout_path = vmec.output_file
    if not os.path.exists(wout_path):
        raise RuntimeError(f"VMEC did not produce output file: {wout_path}")

    import netCDF4 as nc
    ds = nc.Dataset(wout_path)
    ier_flag = int(ds.variables["ier_flag"][:])
    fsql = float(ds.variables["fsql"][:])
    ds.close()

    if ier_flag != 0:
        raise RuntimeError(
            f"VMEC did not converge: ier_flag={ier_flag}, fsql={fsql}"
        )

    if fsql > 1e-8:
        import warnings
        warnings.warn(f"VMEC fsql={fsql:.2e} > 1e-8, convergence may be poor")

    return wout_path
```

- [ ] **Step 2: Run tests**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_vmec_runner.py -v
```

Expected: 5 preparation tests PASS, 2 VMEC execution tests SKIP (if VMEC not yet installed) or PASS (if installed).

- [ ] **Step 3: Commit**

```bash
git add tme/vmec_runner.py
git commit -m "feat: implement VMEC input preparation and runner (REQ-VMEC-001 through REQ-VMEC-006)"
```

---

## Chunk 5: Analysis Module

### Task 11: Write test_analysis.py

**Files:**
- Create: `tme/tests/test_analysis.py`

- [ ] **Step 1: Write tests**

Tests for extraction functions are split: wout-dependent tests skip if no wout exists.

```python
# tme/tests/test_analysis.py
"""Tests for analysis module. Maps to REQ-ANALYSIS-001 through REQ-ANALYSIS-003."""
import os
import json
import pytest

QUASR_DIR = os.path.join(os.path.dirname(__file__), "..", "QUASR_eq")
SERIAL_FILE = os.path.join(QUASR_DIR, "serial0010273.json")
INPUT_TEMPLATE = os.path.join(QUASR_DIR, "input.0010273")

# Look for a pre-existing wout file from a test run
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "0010273")
WOUT_GLOB = os.path.join(OUTPUT_DIR, "wout_*.nc")


def find_wout():
    """Find a wout file if one exists from a prior run."""
    import glob
    files = glob.glob(WOUT_GLOB)
    return files[0] if files else None


def vmec_available():
    try:
        from simsopt.mhd import Vmec
        return True
    except ImportError:
        return False


@pytest.fixture
def wout_path():
    """Get or create a wout file for testing."""
    existing = find_wout()
    if existing:
        return existing

    if not vmec_available():
        pytest.skip("No wout file and VMEC not installed")

    # Generate one
    from load_quasr import load_quasr_equilibrium
    from vmec_runner import prepare_vmec_input, run_vmec

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eq = load_quasr_equilibrium(SERIAL_FILE)
    input_path = prepare_vmec_input(
        eq["surface"], eq["metadata"], OUTPUT_DIR,
        input_template=INPUT_TEMPLATE,
    )
    return run_vmec(input_path)


def test_extract_vmec_results_keys(wout_path):
    """TEST-ANALYSIS-001: extract returns required keys (REQ-ANALYSIS-001)."""
    from analysis import extract_vmec_results

    results = extract_vmec_results(wout_path)
    required_keys = ["iotaf", "presf", "betatot", "DMerc", "Vp",
                     "ier_flag", "fsql"]
    for key in required_keys:
        assert key in results, f"Missing key: {key}"


def test_extract_vmec_results_beta(wout_path):
    """TEST-ANALYSIS-002: extracted beta is finite and positive."""
    from analysis import extract_vmec_results

    results = extract_vmec_results(wout_path)
    assert results["betatot"] > 0
    assert results["betatot"] < 1  # sanity


def test_save_results(wout_path, tmp_path):
    """TEST-ANALYSIS-003: save_results writes valid JSON (REQ-ANALYSIS-003)."""
    from analysis import extract_vmec_results, save_results

    results = extract_vmec_results(wout_path)
    metadata = {
        "model_id": "0010273",
        "symmetry_type": "QA",
        "nfp": 2,
        "beta_target": 0.02,
    }
    save_results(results, metadata, str(tmp_path), wout_path=wout_path)

    json_path = os.path.join(str(tmp_path), "results.json")
    assert os.path.exists(json_path)

    with open(json_path) as f:
        data = json.load(f)
    assert "metadata" in data
    assert "vmec" in data
    assert "converged" in data["vmec"]
    assert "wout_path" in data


@pytest.mark.skipif(not vmec_available(), reason="VMEC not installed")
def test_run_boozer_analysis(wout_path):
    """TEST-ANALYSIS-004: Boozer analysis returns epsilon_eff (REQ-ANALYSIS-002)."""
    from analysis import run_boozer_analysis

    results = run_boozer_analysis(wout_path)
    assert "epsilon_eff" in results
    assert "bmnc_b" in results
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_analysis.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'analysis'`

- [ ] **Step 3: Commit**

```bash
git add tme/tests/test_analysis.py
git commit -m "test: add tests for VMEC analysis and Boozer (REQ-ANALYSIS-001 through REQ-ANALYSIS-003)"
```

---

### Task 12: Implement analysis.py

**Files:**
- Create: `tme/analysis.py`

- [ ] **Step 1: Write implementation**

```python
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
        "betatot": float(ds.variables["betatot"][:]),
        "DMerc": ds.variables["DMerc"][:].tolist(),
        "Vp": ds.variables["vp"][:].tolist(),
        "ier_flag": int(ds.variables["ier_flag"][:]),
        "fsql": float(ds.variables["fsql"][:]),
        "ns": int(ds.dimensions["radius"].size),
        "nfp": int(ds.variables["nfp"][:]),
        "mpol": int(ds.variables["mpol"][:]),
        "ntor": int(ds.variables["ntor"][:]),
    }

    # Check beta vs target
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
    import booz_xform

    vmec = Vmec(wout_path)

    # Boozer constructor takes mpol/ntor, not mboz/nboz
    boozer = Boozer(vmec, mpol=mboz, ntor=nboz)

    # Must register surfaces before running
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
```

- [ ] **Step 2: Run tests**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/test_analysis.py -v
```

Expected: Tests requiring wout files SKIP if VMEC not yet installed, or PASS if wout exists.

- [ ] **Step 3: Commit**

```bash
git add tme/analysis.py
git commit -m "feat: implement VMEC analysis and Boozer extraction (REQ-ANALYSIS-001 through REQ-ANALYSIS-003)"
```

---

## Chunk 6: Orchestration Notebook

### Task 13: Create run_all.ipynb

**Files:**
- Create: `tme/run_all.ipynb`

- [ ] **Step 1: Write the orchestration notebook**

The notebook has these cells:

**Cell 1 — Imports and config:**
```python
import os
import sys
import glob
import warnings

# Ensure tme/ modules are importable (run notebook from tme/ directory)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from load_quasr import load_quasr_equilibrium
from vmec_runner import prepare_vmec_input, run_vmec
from visualize import (
    plot_vacuum_3d, plot_cross_sections,
    plot_vmec_profiles, plot_flux_surfaces, plot_finite_beta_3d,
)
from analysis import extract_vmec_results, run_boozer_analysis, save_results

# Configuration
QUASR_DIR = "QUASR_eq"
OUTPUT_BASE = "output"
EQUILIBRIA = {
    "0010273": {"symmetry_type": "QA"},
    "0019548": {"symmetry_type": "QA"},
    "0358936": {"symmetry_type": "QA"},
}
```

**Cell 2 — Load and visualize vacuum (loop):**
```python
equilibria = {}

for model_id, config in EQUILIBRIA.items():
    print(f"\n{'='*60}")
    print(f"Loading model {model_id} ({config['symmetry_type']})")
    print(f"{'='*60}")

    json_path = os.path.join(QUASR_DIR, f"serial{model_id}.json")
    eq = load_quasr_equilibrium(json_path, symmetry_type=config["symmetry_type"])
    equilibria[model_id] = eq

    output_dir = os.path.join(OUTPUT_BASE, model_id)
    os.makedirs(output_dir, exist_ok=True)

    # Vacuum 3D visualization with B·n
    print("  Plotting vacuum 3D (B·n)...")
    fig_3d = plot_vacuum_3d(
        eq["surface"], eq["coils"], eq["bs"],
        save_path=os.path.join(output_dir, "vacuum_3d.html"),
    )
    fig_3d.show()

    # Cross-sections
    print("  Plotting cross-sections...")
    fig_cs = plot_cross_sections(
        eq["surface"], nfp=eq["metadata"]["nfp"],
        save_path=os.path.join(output_dir, "cross_sections.png"),
    )

    print(f"  Model {model_id}: {len(eq['coils'])} coils, nfp={eq['metadata']['nfp']}")
```

**Cell 3 — Run VMEC (loop):**
```python
wout_paths = {}

for model_id, eq in equilibria.items():
    print(f"\n{'='*60}")
    print(f"Running VMEC for model {model_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(OUTPUT_BASE, model_id)
    input_template = os.path.join(QUASR_DIR, f"input.{model_id}")

    try:
        input_path = prepare_vmec_input(
            eq["surface"], eq["metadata"], output_dir,
            input_template=input_template,
        )
        print(f"  Input written to: {input_path}")

        wout_path = run_vmec(input_path)
        wout_paths[model_id] = wout_path
        print(f"  VMEC converged! wout: {wout_path}")

    except Exception as e:
        print(f"  VMEC FAILED for {model_id}: {e}")
        warnings.warn(f"Skipping {model_id} due to VMEC failure")
        continue
```

**Cell 4 — Extract results and Boozer analysis:**
```python
all_results = {}

for model_id, wout_path in wout_paths.items():
    print(f"\n{'='*60}")
    print(f"Analyzing model {model_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(OUTPUT_BASE, model_id)

    # VMEC results
    results = extract_vmec_results(wout_path)
    print(f"  Beta = {results['betatot']:.4f}")
    print(f"  ier_flag = {results['ier_flag']}, fsql = {results['fsql']:.2e}")

    # Boozer analysis
    try:
        boozer_results = run_boozer_analysis(wout_path)
        results.update(boozer_results)
        print(f"  Boozer analysis complete, {len(boozer_results['bmnc_b'])} dominant modes")
    except Exception as e:
        print(f"  Boozer analysis failed: {e}")

    # Save
    save_results(results, equilibria[model_id]["metadata"], output_dir,
                 wout_path=wout_path)
    all_results[model_id] = results
    print(f"  Results saved to {output_dir}/results.json")
```

**Cell 5 — Visualize finite-beta results:**
```python
for model_id, wout_path in wout_paths.items():
    print(f"\n{'='*60}")
    print(f"Visualizing finite-beta results for {model_id}")
    print(f"{'='*60}")

    output_dir = os.path.join(OUTPUT_BASE, model_id)

    # Profiles
    fig_prof = plot_vmec_profiles(
        wout_path,
        save_path=os.path.join(output_dir, "profiles.png"),
    )

    # Flux surfaces
    fig_fs = plot_flux_surfaces(
        wout_path,
        save_path=os.path.join(output_dir, "flux_surfaces.png"),
    )

    # 3D LCFS with |B|
    fig_3d = plot_finite_beta_3d(
        wout_path,
        save_path=os.path.join(output_dir, "finite_beta_3d.html"),
    )
    fig_3d.show()
```

**Cell 6 — Summary table:**
```python
import pandas as pd

summary = []
for model_id, results in all_results.items():
    summary.append({
        "Model": model_id,
        "Beta": f"{results['betatot']:.4f}",
        "Iota(0)": f"{results['iotaf'][0]:.4f}",
        "Iota(edge)": f"{results['iotaf'][-1]:.4f}",
        "Converged": results["ier_flag"] == 0,
    })

pd.DataFrame(summary)
```

- [ ] **Step 2: Verify notebook structure is correct**

```bash
cd /home/telder1/COIL_OPT/tme
jupyter nbconvert --to script run_all.ipynb --stdout | head -20
```

Or simply open in Jupyter and verify cells parse.

- [ ] **Step 3: Commit**

```bash
git add tme/run_all.ipynb
git commit -m "feat: add orchestration notebook for QUASR finite-beta pipeline (REQ-PIPE-001)"
```

---

## Chunk 7: Integration Test and End-to-End Run

### Task 14: Run the full pipeline

- [ ] **Step 1: Verify all unit tests pass**

```bash
cd /home/telder1/COIL_OPT/tme
python -m pytest tests/ -v
```

- [ ] **Step 2: Run the notebook end-to-end**

Open `tme/run_all.ipynb` in Jupyter and run all cells. Verify:
- All 3 equilibria load successfully
- Vacuum 3D plots show reasonable surface shapes with B·n coloring
- Cross-section plots show expected stellarator shapes
- VMEC converges for all 3 (ier_flag == 0)
- Achieved beta is within 20% of 0.02 target
- Profiles (iota, pressure, Mercier, well) look physically reasonable
- Boozer analysis completes
- Results JSON files are written to `output/{model_id}/results.json`

- [ ] **Step 3: If VMEC doesn't converge, adjust parameters**

Common fixes:
- Reduce beta target (try 1% first)
- Increase NITER
- Add intermediate NS step: `NS_ARRAY=[16, 33, 49]`
- Adjust FTOL

- [ ] **Step 4: Final commit with any fixes**

```bash
git add tme/
git commit -m "feat: complete QUASR finite-beta pipeline, all 3 equilibria processed"
```
