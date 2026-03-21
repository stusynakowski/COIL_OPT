# tme/vmec_runner.py
"""Prepare and run fixed-boundary VMEC equilibria.

Implements REQ-VMEC-001 through REQ-VMEC-006.
"""
import os
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

    # Use B0 and R0 from metadata if available, otherwise default
    B0 = metadata.get("B0", 1.0)
    gamma = surface.gamma()
    R_vals = np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2)
    R0 = metadata.get("R0", float(np.mean(R_vals)))

    beta_target = metadata["beta_target"]

    # --- Pressure profile: p(s) = p0 * (1 - s)^2 ---
    # VMEC AM array: pressure = PRES_SCALE * sum(AM(i) * s^i)
    # (1-s)^2 = 1 - 2s + s^2 → AM = [1, -2, 1, 0, ...]
    p0 = estimate_p0(beta_target, B0)
    indata["pres_scale"] = float(p0)
    indata["am"] = [1.0, -2.0, 1.0] + [0.0] * 8

    # --- Resolution ---
    indata["mpol"] = 10
    indata["ntor"] = 10
    indata["ns_array"] = [16, 49]
    indata["ftol_array"] = [1e-8, 1e-10]
    indata["niter_array"] = [2000, 10000]

    # --- Current/iota profile ---
    if metadata.get("profile_type") == "bootstrap":
        # QA: prescribe toroidal current
        indata["ncurr"] = 1
        curtor = estimate_curtor(beta_target, iota_frac=0.15, R0=R0, B0=B0)
        indata["curtor"] = float(curtor)
        # Current profile shape: j(s) ~ (1 - s), peaked on axis
        indata["ac"] = [1.0, -1.0] + [0.0] * 9
    else:
        # QH/QP: fixed iota
        indata["ncurr"] = 0
        indata["curtor"] = 0.0

    # --- Other VMEC settings ---
    indata["lfreeb"] = False  # fixed boundary
    indata["nstep"] = 200

    # Write the namelist
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

    if fsql > 1e-6:
        import warnings
        warnings.warn(f"VMEC fsql={fsql:.2e} > 1e-8, convergence may be poor")

    return wout_path


def run_vmec_to_target_beta(surface, metadata: dict, output_dir: str,
                            input_template: str, max_iterations: int = 3) -> str:
    """
    Run VMEC iteratively, scaling PRES_SCALE to achieve the target beta.

    Does a calibration run, measures achieved beta, rescales PRES_SCALE
    proportionally, and re-runs. Beta scales linearly with PRES_SCALE
    at low beta, so this converges in 1-2 iterations.

    Returns:
        Path to final wout file.
    """
    import netCDF4 as nc

    beta_target = metadata["beta_target"]
    pres_scale_override = None

    for iteration in range(max_iterations):
        input_path = prepare_vmec_input(
            surface, metadata, output_dir, input_template=input_template,
        )

        # Override PRES_SCALE if we have a correction from a previous iteration
        if pres_scale_override is not None:
            nml = f90nml.read(input_path)
            nml["indata"]["pres_scale"] = pres_scale_override
            nml.write(input_path, force=True)

        wout_path = run_vmec(input_path)

        ds = nc.Dataset(wout_path)
        beta_achieved = float(ds.variables["betatotal"][:])
        ds.close()

        if beta_achieved <= 0:
            import warnings
            warnings.warn("VMEC returned beta=0, cannot rescale")
            return wout_path

        ratio = beta_target / beta_achieved
        if abs(ratio - 1.0) < 0.2:  # within 20% of target
            return wout_path

        # Scale PRES_SCALE linearly (beta ∝ PRES_SCALE at low beta)
        current_nml = f90nml.read(input_path)
        current_pres = current_nml["indata"]["pres_scale"]
        pres_scale_override = float(current_pres * ratio)
        print(f"  Iteration {iteration+1}: beta={beta_achieved:.4e}, "
              f"target={beta_target:.4e}, ratio={ratio:.1f}x, "
              f"new PRES_SCALE={pres_scale_override:.1f}")

    return wout_path
