"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid",
             label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff,
             where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff,
             where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid",
                     alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid",
                 color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7,
               label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid",
                color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-",
                label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid",
                color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h",
           title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3",
             lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid",
            color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid",
            color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh",
           title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_disutility_sweep(
        df, price_import=(), price_export=(), base_value: float | None = None,
        xlabel: str = 'disutility coefficient', logx: bool = False,
        save_to: Path | str | None = None,
) -> plt.Figure:
    '''Metrics of a disutility-coefficient sweep. For 2(b) and 2(c), iv.

    Top: total absolute deviation (left axis, kWh) and number of deviating hours (right).
    Bottom: cost components (DKK). Vertical grey lines mark the distinct hourly effective
    import (dashed) and export (dotted) prices - the values where the hypotheses of
    2.(b).ii predict the steps of the linear model.
    '''
    x = df.index.to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

    for ax in (ax1, ax2):
        for v in np.unique(price_import):
            ax.axvline(v, color="grey", ls="--", lw=0.5, alpha=0.6)
        for v in np.unique(price_export):
            ax.axvline(v, color="grey", ls=":", lw=0.5, alpha=0.6)
        if base_value is not None:
            ax.axvline(base_value, color="C3", lw=1.0, alpha=0.8)

    ax1.plot(x, df["abs_deviation_kWh"], ".-",
             color="C0", label="total |deviation| [kWh]")
    ax1.plot(x, df["energy_kWh"], ".-", color="C2",
             label="daily energy consumed [kWh]")
    ax1.set_ylabel("kWh")
    ax1b = ax1.twinx()
    ax1b.plot(x, df["hours_deviating"], ".-",
              color="C1", label="# deviating hours")
    ax1b.set_ylabel("hours", color="C1")
    lines, labels = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, fontsize=8, loc="upper right")
    title = "Sweep of the disutility coefficient"
    if len(np.atleast_1d(price_import)) or len(np.atleast_1d(price_export)):
        title += " (grey lines: hourly effective prices)"
    if base_value is not None:
        title += " - red: base value"
    ax1.set_title(title, fontsize=10)

    ax2.plot(x, df["procurement_DKK"], ".-", label="procurement cost")
    ax2.plot(x, df["disutility_DKK"], ".-", label="total disutility")
    ax2.plot(x, df["total_cost_DKK"], "k.-", label="total cost (objective)")
    ax2.set(xlabel=xlabel, ylabel="DKK")
    ax2.legend(fontsize=8)
    if logx:
        ax1.set_xscale("log")
    return _finish(fig, save_to)


def plot_load_comparison(runs, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """For Question 2(d) --> the optimal load of several models on one shared axis, versus the
    reference profile and the available PV, with prices on a secondary axis.

    ``runs`` maps a label (e.g. "Q1 (linear utility)") to a :class:`Results`.
    """
    fig, ax = plt.subplots(figsize=(11, 4.4))
    h = data.hours
    ax.step(h, data.reference_load, where="mid", color="k",
            ls=":", lw=1.4, label="reference profile")
    ax.step(h, data.pv_available, where="mid", color="orange",
            ls="--", lw=1.1, label="PV available")
    for (label, res), color in zip(runs.items(), ("C0", "C2", "C3", "C4")):
        ax.step(h, res.hourly["load"], where="mid",
                color=color, lw=1.6, label=f"load - {label}")
    ax.set(xlabel="hour", ylabel="kWh/h",
           title="Optimal load profiles across objective functions (base cases)")

    ax2 = ax.twinx()
    ax2.step(h, data.energy_price, where="mid", color="grey",
             lw=0.9, alpha=0.7, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="grey")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)
