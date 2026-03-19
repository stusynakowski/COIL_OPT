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
