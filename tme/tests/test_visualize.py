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
