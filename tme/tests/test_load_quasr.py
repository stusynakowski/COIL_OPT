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
