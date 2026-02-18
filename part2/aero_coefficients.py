"""
Aerodynamic model of the aircraft.
"""

from __future__ import annotations
import math
import time


def run_aero(alpha_deg: float) -> tuple[float, float]:
    """
    Aerodynamic calculation.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.

    Returns
    -------
    CL, CD : float, float
        Lift and drag coefficients.
    """
    # Artificial delay to simulating the CFD
    # Please do not change this! In weblab this is fixed.
    delay_s = 0.02
    time.sleep(delay_s)

    # Convert AoA to radians
    a = math.radians(alpha_deg)

    # --- Nonlinear "physics-ish" model ---
    # Compressibility-ish lift slope reduction with Mach (very rough)
    # keep a mild compressibility-ish effect as a fixed factor (no Mach input)
    beta = 0.9
    cl_alpha_0 = 2.0 * math.pi  # per rad (thin airfoil)
    cl_alpha = cl_alpha_0 * (0.85 + 0.15 * beta)

    # Mild camber shift and Mach effects
    alpha0 = math.radians(-1.0)
    CL_lin = cl_alpha * (a - alpha0)

    # Soft stall saturation (smoothly limits CL)
    CL_max = 1.35
    CL = CL_max * math.tanh(CL_lin / max(1e-6, CL_max))

    # Drag polar: CD = CD0(M) + k(M)*CL^2 + wave_drag(M, CL) + small nonlinear AoA term
    CD0 = 0.018
    k = 0.045

    # Wave drag ramp near transonic
    wave = 0.0

    # Extra AoA penalty (separation-ish)
    sep = 0.004 * (abs(alpha_deg) / 10.0) ** 3

    CD = CD0 + k * (CL**2) + wave + sep

    return float(CL), float(CD)
