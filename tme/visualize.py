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

    # Surface quadpoints cover one field period [0, 1/nfp] (normalized)
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
    ns = len(ds.variables["iotaf"][:])
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

    # Plot every 10th surface
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
