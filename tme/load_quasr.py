# tme/load_quasr.py
"""Load QUASR equilibria from SIMSOPT serialized JSON files.

Implements REQ-LOAD-001 and REQ-LOAD-002.
"""
import os
import re
import numpy as np
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
    coils = objs[1]

    if not surfaces:
        raise ValueError(f"No surfaces found in {json_path}")
    if not coils:
        raise ValueError(f"No coils found in {json_path}")

    # Use the first surface as the boundary
    surface = surfaces[0]

    # Build BiotSavart field
    bs = BiotSavart(coils)

    # Estimate B0 on the magnetic axis (average |B| over toroidal angle at
    # the geometric center of each cross-section)
    gamma = surface.gamma()
    R_vals = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
    R0 = float(np.mean(R_vals))
    # Evaluate |B| at the centroid of each toroidal cross-section
    centroids = np.mean(gamma, axis=1)  # (nphi, 3)
    bs.set_points(centroids)
    B0 = float(np.mean(np.linalg.norm(bs.B(), axis=1)))

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
        "B0": B0,
        "R0": R0,
    }

    return {
        "surface": surface,
        "coils": coils,
        "bs": bs,
        "metadata": metadata,
    }
