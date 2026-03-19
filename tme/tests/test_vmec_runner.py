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
