"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import Results, model_for_case
from src.plotting import (plot_disutility_sweep, plot_duals, plot_inputs, plot_load_comparison,
                          plot_scenario_comparison, plot_schedule,
                          plot_schedule_q3_comparison, plot_duals_q3)
from src.scenarios import scale_prices, scale_pv, set_disutility, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    try:
        # the class registered for this case
        ModelClass = model_for_case(question)
        model = ModelClass(data).build()
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    results_q2c = None
    if "Q3" in question:
        print("--> Q3 question detected, solving Q2.(c) (without E_min constraint) simultaneously for comparison curves...")
        model_q2c = Q2QuadraticModel(data).build()
        results_q2c = model_q2c.solve()
        print(f"Q2.(c) solved. Unconstrained energy consumption: {results_q2c.hourly['load'].sum():.2f} kWh (Q3 mandatory constraint is {data.min_daily_energy_kWh:.2f} kWh)")

    print(results, "\n")
    results.save(out)

    if "Q3" in question and results_q2c is not None:
        plot_schedule_q3_comparison(results, results_q2c, data, save_to=out / "schedule.png")
        plot_duals_q3(results, data, save_to=out / "duals.png")
    else:
        plot_schedule(results, data, save_to=out / "schedule.png")
        plot_duals(results, data, save_to=out / "duals.png")

    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    ModelClass = model_for_case(question)
    for name, data in scenarios.items():
        results = ModelClass(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(
        runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def _disutility_sweep(case: str, coeff: str, grid, out: Path, xlabel: str,
                      price_lines: bool = True, logx: bool = False):
    """Shared code of the disutility sweeps Questions 2.(b).iv and 2.(c).iv.

    Solves the case once per coefficient value in ``grid`` and collects the metric table
    required by the assignment: daily procurement cost, total disutility, daily energy
    consumed, total absolute deviation, and the number of hours with a non-zero deviation.
    """
    import numpy as np
    import pandas as pd

    base = load_question(case)
    ModelClass = model_for_case(case)
    rows = []
    for c in grid:
        res = ModelClass(set_disutility(
            base, **{coeff: float(c)})).build().solve()
        h = res.hourly
        dev = (h["load"] - h["reference_load"]).abs()
        rows.append({
            coeff: float(c),
            "procurement_DKK": res.meta["objective_terms"]["procurement_cost"],
            "disutility_DKK": res.meta["objective_terms"]["disutility"],
            "total_cost_DKK": res.objective,
            "energy_kWh": h["load"].sum(),
            "abs_deviation_kWh": dev.sum(),
            "hours_deviating": int((dev > 1e-6).sum()),
        })
    df = pd.DataFrame(rows).set_index(coeff)
    stem = "sweep_c_L" if coeff == "linear" else "sweep_c_Q"
    df.to_csv(out / f"{stem}.csv")
    plot_disutility_sweep(
        df,
        price_import=(base.energy_price +
                      base.import_tariff) if price_lines else (),
        price_export=(base.energy_price -
                      base.export_tariff) if price_lines else (),
        base_value=getattr(base, f"{coeff}_disutility"),
        xlabel=xlabel, logx=logx, save_to=out / f"{stem}.png")
    print(df.iloc[::4].to_string(float_format=lambda v: f"{v:8.2f}"))
    print(f"\nSweep written to {out / stem}.csv/.png")
    return df


def run_q2b_sweep(out: Path):
    '''Question 2b.iv, to sweep c_L on Q2_linear. The range spans from below the smallest supply
    margin (deviating everywhere), to above the largest one (never deviate). Those margins are the
    hourly effective prices and c_PV'''
    import numpy as np
    base = load_question('Q2_linear')
    grid = np.round(np.arange(0.05, (base.energy_price +
                                    base.import_tariff).max() + 0.35, 0.05), 2)
    return _disutility_sweep('Q2_linear', 'linear', grid, out, xlabel='c_L [DKK/kWh]')


def run_q2c_sweep(out: Path):
    '''Question 2c.iv: sweep c_Q on Q2_quadratic (same metric table as 2b.iv).

    This uses a geometric grid: the interior response scales like (price margin)/(2 c_Q),
    so equal *ratios* of c_Q... not equal increments. It probes evenly. Chosen relative to
    the data: at c_Q = 0.02 the implied deviations far exceed the load bounds (near
    cost-minimizer behavior), at c_Q = 10 they are ~0.1 kWh (near reference-tracking).
    No vertical price lines: c_Q is in DKK/kWh^2, not comparable with prices, and the
    hypothesis is a smooth response with no thresholds.'''
    import numpy as np
    grid = np.round(np.geomspace(0.02, 10.0, 40), 4)
    return _disutility_sweep('Q2_quadratic', 'quadratic', grid, out,
                             xlabel="c_Q [DKK/kWh^2]", price_lines=False, logx=True)


def run_q2d_comparison(out: Path):
    '''Question 2d: compare the three objective functions on the same day.

    Solves the base case of each model (Q1 linear utility on Q1_caseA; linear and
    quadratic disutility on their Q2 cases - identical prices, tariffs and PV, cf. the
    data README), overlays the optimal load profiles, and tabulates the shared metrics.
    The deviation of the Q1 consumer is measured against the Q2 reference profile for
    comparability (Q1 itself has no reference; same load bounds in all cases).
    '''
    import numpy as np
    import pandas as pd

    # carries the reference profile
    ref_data = load_question("Q2_linear")
    cases = {
        "Q1 (linear utility)": "Q1_caseA",
        "linear disutility": "Q2_linear",
        "quadratic disutility": "Q2_quadratic",
    }
    runs, rows = {}, []
    for label, case in cases.items():
        res = model_for_case(case)(load_question(case)).build().solve()
        runs[label] = res
        h = res.hourly
        dev = (h["load"] - ref_data.reference_load).abs()
        terms = res.meta["objective_terms"]
        rows.append({
            "model": label,
            "procurement_DKK": terms["procurement_cost"],
            "preference_term_DKK": terms.get("disutility", -terms.get("utility", np.nan)),
            "energy_kWh": h["load"].sum(),
            "import_kWh": h["import"].sum(),
            "export_kWh": h["export"].sum(),
            "abs_dev_vs_ref_kWh": dev.sum(),
            "hours_deviating": int((dev > 1e-6).sum()),
        })
    df = pd.DataFrame(rows).set_index("model")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "q2d_model_comparison.csv")
    plot_load_comparison(runs, ref_data, save_to=out / "q2d_load_profiles.png")
    print(df.to_string(float_format=lambda v: f"{v:8.2f}"))
    print(f"\nComparison written to {out}")
    return df

from src.model import Q2QuadraticModel
from src.q3_model import ModelQ3

def run_q3_analysis(question: str = "Q3_base", out: Path = RESULTS_DIR / "Q3_base"):
    out.mkdir(parents=True, exist_ok=True)
    data = load_question(question)
    
    res_q2c = Q2QuadraticModel(data).build().solve()
    
    res_q3 = ModelQ3(data).build().solve()
    
    res_q3.save(out)

    plot_schedule_q3_comparison(res_q3, res_q2c, data, save_to=out / "schedule.png")
    plot_duals_q3(res_q3, data, save_to=out / "duals.png")
    
    print(f"Q3 analysis completed, figures saved to: {out}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA",
                        choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true",
                        help="also run the example sensitivity scenarios")
    parser.add_argument("--sweep-cl", action="store_true",
                        help="run the c_L sweep of Question 2.(b).iv (case Q2_linear)")
    parser.add_argument("--sweep-cq", action="store_true",
                        help="run the c_Q sweep of Question 2.(c).iv (case Q2_quadratic)")
    parser.add_argument("--compare-q2", action="store_true",
                        help="run the multi-model comparison of Question 2.(d)")
    parser.add_argument("--show", action="store_true",
                        help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
