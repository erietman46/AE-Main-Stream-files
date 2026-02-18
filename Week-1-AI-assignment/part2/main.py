#!/usr/bin/env python3

"""
Simple point-mass mission simulation that repeatedly calls an aero model.


What it does:
- Builds (or loads) a surrogate from sampled expensive calls
- Runs the same mission twice (expensive vs surrogate)
- Prints runtime + key outputs and compares them
"""

from __future__ import annotations
import time
import numpy as np
from tqdm.auto import tqdm

from aero_coefficients import run_aero

"""
TODO:
Part 2, Step 5(a):
    - Add import of the SurrogateModel from file surrogate_model.py
"""



# ----------------------------
# Aircraft + atmosphere helpers
# ----------------------------
def isa_density(alt_m: float) -> float:
    """
    Very rough density model for low-to-mid altitudes.
    """
    rho0 = 1.225
    h_scale = 8500.0
    return rho0 * np.exp(-alt_m / h_scale)


def speed_of_sound(alt_m: float) -> float:
    """Rough constant a (m/s)."""
    return 340.0


# ----------------------------
# Mission simulation
# ----------------------------
def run_mission(run_aero_fn, n_steps: int = 200, dt: float = 0.1) -> dict:
    """
    Point-mass forward sim.
    State: altitude, speed
    Controls: hold a target Mach schedule and compute AoA needed for lift

    We intentionally call run_aero_fn() every step to emulate a main sim that
    needs aero data constantly.

    If target_range_m is provided, an ETA is computed as remaining_range / current_speed.
    Otherwise, ETA is computed as remaining_sim_time = (n_steps - k - 1) * dt.
    """
    # Aircraft parameters
    m = 65000.0          # kg
    S = 122.6            # m^2
    g = 9.80665

    # Thrust model
    T_sl = 2.2e5         # [N] sea-level max
    thrust_lapse = 0.75  # altitude lapse rate for thrust

    # Initialize
    alt = 0.0            # m
    V = 130.0            # m/s
    x = 0.0              # "range" proxy (m)
    target_range_m = 500e3  # m

    # Mission target schedule: accelerate/climb then cruise
    # We'll impose a target Mach that ramps up and then holds
    Mach_target = np.linspace(0.25, 0.78, n_steps)
    Mach_target[int(0.35 * n_steps):] = 0.78

    # Outputs
    cl_list = []
    cd_list = []
    alpha_list = []
    alt_list = []
    V_list = []
    eta_list = []

    iterator = range(n_steps)
    # show progress bar
    iterator = tqdm(iterator, desc="Simulating mission", unit="step")

    for k in iterator:
        rho = float(isa_density(alt))
        a = float(speed_of_sound(alt))
        M = float(np.clip(V / a, 0.05, 0.85))

        # Simple speed control: accelerate/decelerate toward Mach_target
        M_cmd = float(Mach_target[k])
        V_cmd = M_cmd * a
        V_err = V_cmd - V
        throttle = float(np.clip(0.5 + 0.003 * V_err, 0.0, 1.0))

        # Compute required CL for quasi-level lift
        q = 0.5 * rho * V * V
        L_req = m * g * 1.05
        CL_req = float(L_req / max(1e-6, q * S))

        # Invert CL -> alpha using an approximate relation
        cl_alpha_guess = 5.5
        alpha_rad_guess = CL_req / max(1e-6, cl_alpha_guess)
        alpha_deg = float(np.clip(np.degrees(alpha_rad_guess), -6.0, 14.0))

        # Aero call
        CL, CD = run_aero_fn(alpha_deg)

        # Forces
        D = q * S * CD

        # Thrust available
        T = throttle * T_sl * (1.0 - thrust_lapse * (alt / 12000.0))
        T = float(np.clip(T, 0.0, T_sl))

        # Longitudinal acceleration
        a_long = (T - D) / m

        # Crude climb rate proxy using excess specific power
        dhdt = ((T - D) * V) / (m * g)
        dhdt = float(np.clip(dhdt, -5.0, 25.0))

        # Integrate
        V = float(max(60.0, V + a_long * dt))
        alt = float(max(0.0, alt + dhdt * dt))
        x = float(x + V * dt)

        # Estimated time until arrival
        if target_range_m is None:
            eta_s = (n_steps - k - 1) * dt
        else:
            remaining_range = max(0.0, target_range_m - x)
            eta_s = remaining_range / max(1e-6, V)

        cl_list.append(CL)
        cd_list.append(CD)
        alpha_list.append(alpha_deg)
        alt_list.append(alt)
        V_list.append(V)
        eta_list.append(eta_s)

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(eta_s=f"{eta_s:.1f}")

    return {
        "final_alt_m": alt,
        "final_speed_mps": V,
        "range_proxy_m": x,
        "mean_CL": float(np.mean(cl_list)),
        "mean_CD": float(np.mean(cd_list)),
        "mean_alpha_deg": float(np.mean(alpha_list)),
        "alt_trace": np.array(alt_list),
        "V_trace": np.array(V_list),
        "eta_trace_s": np.array(eta_list),
    }

def main():
    t0 = time.perf_counter()
    out_exp = run_mission(lambda a: run_aero(a), n_steps=200)
    t1 = time.perf_counter()
    # Report
    print("\n=== Mission outputs ===")
    print(f"First Run:")
    print(f"final_alt={out_exp['final_alt_m']:.1f} m, "
          f"range~={out_exp['range_proxy_m']/1000:.2f} km, "
          f"mean_CD={out_exp['mean_CD']:.4f}, "
          f"eta={out_exp['eta_trace_s'][-1]/60:.1f} min")

    print("\n=== Timing ===")
    print(f"First Mission runtime: {(t1 - t0):.3f} s")

    """
    TODO:
    Part 2, Step 5(d):
        - Run the same mission again using the surrogate and compare outputs + runtime
        (uncomment the lines below and complete them)
        - Change the number of steps and see how surrogate vs expensive runtimes scale
    """

    # build a surrogate model
    # surrogate_model = SurrogateModel(degree = 4, N_alpha = 10, alpha_min = -6.0, alpha_max = 14.0)
    # train the surrogate model
    # surrogate_model.train()
    # run the mission wit the surrogate model
    # ...


if __name__ == "__main__":
    main()
