"""Sanity checks: solver vs closed-form/brute-force, plus the checks from the drafts.

Run from the repo root:  python verify.py
Not part of the submission pipeline - a development check (grading guide: 'compare
closed forms with the solver', 'sanity-check before interpreting').
"""
import numpy as np

from src.data_loader import load_question
from src.model import FlexibleConsumerModel, Q2LinearModel, Q2QuadraticModel

TOL = 1e-6


def procurement_cost(l, pv_avail, pi_i, pi_e, c_pv):
    """Cheapest way to serve a fixed load l in one hour (analytical dispatch).

    Serve the load from PV if PV is cheaper than importing; export leftover PV only if
    the effective export price beats the PV production cost.
    """
    a = min(pv_avail, l) if c_pv < pi_i else 0.0          # PV used for own load
    imp = l - a                                            # remainder imported
    # extra PV produced to export
    e = (pv_avail - a) if pi_e > c_pv else 0.0
    return c_pv * (a + e) + pi_i * imp - pi_e * e


def brute_force_hour(objective_of_load, lo, hi, extra_candidates=()):
    """Best load value on a fine grid plus the analytic candidate points."""
    grid = np.concatenate(
        [np.linspace(lo, hi, 6001), np.asarray(extra_candidates)])
    grid = grid[(grid >= lo - 1e-12) & (grid <= hi + 1e-12)]
    vals = [objective_of_load(l) for l in grid]
    i = int(np.argmin(vals))
    return grid[i], vals[i]


def common_checks(name, r):
    h = r.hourly
    bal = np.max(np.abs(h["pv"] + h["import"] - h["load"] - h["export"]))
    overlap = float(np.max(np.minimum(h["import"], h["export"])))
    inside = (h["load"].min() >= -TOL) and (h["load"].max() <= 6 + 1e-6)
    print(f"[{name}] balance residual {bal:.2e} | max simultaneous imp*exp overlap {overlap:.2e} kWh"
          f" | load within bounds: {inside}")
    assert bal < 1e-6 and overlap < 1e-6 and inside


def dual_checks(name, model, r):
    """Duals land in Pi, not RC (the bug the old lb=0 duplicate bound caused).

    * ``load`` is a free variable (lb=-GRB.INFINITY), so its reduced cost must be 0 in
      every hour - any bound multiplier hiding in RC means the bound exists twice.
    * Q1 stationarity of the Lagrangian in load_t (drafts/Q1a, written for the model
      as solved - max, constraints arranged as in src/model.py):
          u_L - lambda_t + mu_min_t - mu_max_t = 0    for every t,
      with lambda = Pi(balance), mu_min = Pi(load_min), mu_max = Pi(load_max).
    """
    h = r.hourly
    rc = np.array([model.var["load"][t].RC for t in range(24)])
    print(f"[{name}] max |RC(load)| {np.max(np.abs(rc)):.2e} (free var: must be 0)")
    assert np.max(np.abs(rc)) < TOL

    if name.startswith("Q1"):
        u = model.data.consumption_utility
        resid = u - h["dual_balance"] + h["dual_load_min"] - h["dual_load_max"]
        print(f"[{name}] stationarity max |u - lambda + mu_min - mu_max| {np.max(np.abs(resid)):.2e}"
              f" | Pi(load_min) in [{h['dual_load_min'].min():.2f}, {h['dual_load_min'].max():.2f}]"
              f" (nonzero where load=0 binds)")
        assert np.max(np.abs(resid)) < TOL
        # arrangement L_min - load <= 0, max => Pi >= 0
        assert h["dual_load_min"].min() > -TOL
        # the bound must actually show up in Pi somewhere (it binds in the peak hours)
        assert h["dual_load_min"].max() > TOL


def check_q1(case):
    d = load_question(case)
    model = FlexibleConsumerModel(d).build()
    r = model.solve()
    common_checks(case, r)
    dual_checks(case, model, r)
    pi_i, pi_e = d.energy_price + d.import_tariff, d.energy_price - d.export_tariff
    u, c_pv = d.consumption_utility, d.pv_marginal_cost
    total = 0.0
    for t in range(24):
        def f(l): return -(u * l) + procurement_cost(l,
                                                     d.pv_available[t], pi_i[t], pi_e[t], c_pv)
        _, v = brute_force_hour(
            f, d.load_min_kWh, d.load_max_kWh, (d.pv_available[t],))
        # back to surplus (max) convention
        total += -v
    print(f"[{case}] surplus: gurobi {r.objective:.4f} vs brute force {total:.4f} "
          f"(diff {abs(r.objective - total):.2e})")
    assert abs(r.objective - total) < 1e-3
    return r


def check_q2_linear():
    d = load_question("Q2_linear")
    model = Q2LinearModel(d).build()
    r = model.solve()
    common_checks("Q2_linear", r)
    dual_checks("Q2_linear", model, r)
    h = r.hourly
    pi_i, pi_e = d.energy_price + d.import_tariff, d.energy_price - d.export_tariff
    c_L, c_pv, ref = d.linear_disutility, d.pv_marginal_cost, d.reference_load

    # s_t == |load - ref| at the optimum (reformulation exact)
    gap = np.max(
        np.abs(h["deviation"] - np.abs(h["load"] - h["reference_load"])))
    # H3: never above the reference
    above = float(np.max(h["load"] - h["reference_load"]))
    print(
        f"[Q2_linear] s=|dev| gap {gap:.2e} | max load-above-reference {above:.2e} kWh (H3)")
    assert gap < 1e-6 and above < 1e-6

    # brute force objective
    total = 0.0
    for t in range(24):
        def f(l): return c_L * abs(l -
                                   ref[t]) + procurement_cost(l, d.pv_available[t], pi_i[t], pi_e[t], c_pv)
        _, v = brute_force_hour(
            f, d.load_min_kWh, d.load_max_kWh, (ref[t], d.pv_available[t]))
        total += v
    print(f"[Q2_linear] cost: gurobi {r.objective:.4f} vs brute force {total:.4f} "
          f"(diff {abs(r.objective - total):.2e})")
    assert abs(r.objective - total) < 1e-3

    # H1/H4: per-hour location of the optimum among the candidate values
    print(f"[Q2_linear] hour-by-hour (c_L = {c_L}):")
    print(f"{'t':>2} {'pi_imp':>6} {'pi_exp':>6} {'PV':>5} {'ref':>5} {'load':>6}  position")
    for t in range(24):
        l, rf, pv = h['load'][t], ref[t], d.pv_available[t]
        pos = ("= ref" if abs(l - rf) < 1e-6 else
               "= 0 (lower bound)" if l < 1e-6 else
               "= PV avail" if abs(l - pv) < 1e-6 else
               f"INTERIOR {l:.3f} <-- unexpected under H1?")
        print(
            f"{t:>2} {pi_i[t]:>6.2f} {pi_e[t]:>6.2f} {pv:>5.2f} {rf:>5.2f} {l:>6.2f}  {pos}")
    return r


def check_q2_quadratic():
    d = load_question("Q2_quadratic")
    model = Q2QuadraticModel(d).build()
    r = model.solve()
    common_checks("Q2_quadratic", r)
    dual_checks("Q2_quadratic", model, r)
    h = r.hourly
    pi_i, pi_e = d.energy_price + d.import_tariff, d.energy_price - d.export_tariff
    c_Q, c_pv, ref = d.quadratic_disutility, d.pv_marginal_cost, d.reference_load
    total = 0.0
    for t in range(24):
        def f(l): return c_Q * (l - ref[t]) ** 2 + procurement_cost(
            l, d.pv_available[t], pi_i[t], pi_e[t], c_pv)
        _, v = brute_force_hour(
            f, d.load_min_kWh, d.load_max_kWh, (ref[t], d.pv_available[t]))
        total += v
    interior = int(np.sum((h["load"] > 1e-6) & (np.abs(h["load"] - ref) > 1e-6)
                          & (h["load"] < d.load_max_kWh - 1e-6)))
    print(f"[Q2_quadratic] cost: gurobi {r.objective:.4f} vs brute force {total:.4f} "
          f"(diff {abs(r.objective - total):.2e}) | interior-solution hours: {interior} (dimmer)")
    assert abs(r.objective - total) < 1e-3
    return r


if __name__ == "__main__":
    check_q1("Q1_caseA")
    check_q1("Q1_caseB")
    check_q2_linear()
    check_q2_quadratic()
    print("\nAll checks passed.")
