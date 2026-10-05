"""Post-solve analysis of a solved run: turns the analytical results of the assignment into
tables that can go straight into the report.

Every function here takes a solved :class:`~src.model.Results` and the
:class:`~src.data_loader.InputData` it was solved on, and returns a ``pandas.DataFrame``.
None of them touches the gurobipy model, so they apply to any run - including the derived
scenarios of :mod:`src.scenarios`.

    from src.analysis import check_price_ladder

    results = FlexibleConsumerModel(data).build().solve()
    ladder = check_price_ladder(results, data)              # Question 1.(f)
    ladder[["regime", "load_hat", "load", "lambda_lo", "lambda", "lambda_hi"]]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import InputData
from .model import Results


def check_price_ladder(results: Results, data: InputData, tol: float = 1e-6) -> pd.DataFrame:
    """Verify the price ladder of Question 1.(d) hour by hour and bound the balance dual.

    Two independent checks per hour, both built from the solved values only:

    * **primal** - the regime and the optimal load and PV output predicted by (d).ii-(d).iv,
      from the comparison of the consumption utility ``u`` and the PV marginal cost ``c_PV``
      with the effective import and export prices, against the solved ``load`` and ``pv``.
    * **dual** - the interval ``[lambda_lo, lambda_hi]`` of values of the balance dual that the
      KKT conditions admit for the solved primal. A strictly positive width means ``lambda_t``
      is not uniquely determined - what Question 1.(f) asks to point out. The three
      stationarity conditions used are, with the model as arranged in ``src/model.py``:
      ``lambda = pi_imp - nu_imp``, ``lambda = pi_exp + nu_exp``, ``lambda = u + mu_min - mu_max``
      and ``lambda = c_PV + mu_pv - nu_pv``, every multiplier being non-negative and zero
      unless its own constraint binds.

    Parameters
    ----------
    results : solved :class:`Results`; needs the columns ``load``, ``pv``, ``import``,
        ``export`` and ``dual_balance``.
    data : the :class:`InputData` it was solved on (Question 1 cases only: needs
        ``consumption_utility``).
    tol : tolerance of the strict price comparisons and of the "variable sits at a bound" tests.

    Returns
    -------
    pandas.DataFrame indexed by hour; energies in kWh/h, prices and duals in DKK/kWh.
    ``match`` is True where the predicted load and PV equal the solved ones, and
    ``lambda_unique`` is False where the admissible interval has a strictly positive width.
    """
    hr = results.hourly
    u, c_pv = data.consumption_utility, data.pv_marginal_cost
    l_min, l_max = data.load_min_kWh, data.load_max_kWh
    pi_imp = data.energy_price + data.import_tariff
    pi_exp = data.energy_price - data.export_tariff

    rows = []
    for t in range(data.n_hours):
        pv_cap = data.pv_available[t]

        # --- predicted regime, load and PV output: (d).ii - (d).iv
        if u > pi_imp[t] + tol:          # consuming is worth more than importing costs
            regime, lam_hat, load_hat = "import", pi_imp[t], l_max
        elif u < pi_exp[t] - tol:        # exporting pays more than consuming is worth
            regime, lam_hat, load_hat = "export", pi_exp[t], l_min
        else:                            # pi_exp < u < pi_imp: neither, pure self-consumption
            regime, lam_hat, load_hat = "self-consume", u, None
        pv_hat = pv_cap if lam_hat > c_pv + tol else 0.0          # (d).iii: curtail or not
        if load_hat is None:             # the load follows the PV actually produced, clipped
            load_hat = float(np.clip(pv_hat, l_min, l_max))       # by the load bounds: (d).iv

        # --- interval of lambda_t admitted by the KKT conditions at the solved primal
        imp, exp = hr["import"][t], hr["export"][t]
        load, pv = hr["load"][t], hr["pv"][t]
        lo, hi = -np.inf, np.inf
        hi = min(hi, pi_imp[t])                       # import_t >= 0  =>  lambda <= pi_imp
        if imp > tol:                                 # importing      =>  lambda == pi_imp
            lo = max(lo, pi_imp[t])
        lo = max(lo, pi_exp[t])                       # export_t >= 0  =>  lambda >= pi_exp
        if exp > tol:                                 # exporting      =>  lambda == pi_exp
            hi = min(hi, pi_exp[t])
        at_lo, at_hi = load <= l_min + tol, load >= l_max - tol
        if at_lo and not at_hi:                       # load at L_min  =>  lambda >= u
            lo = max(lo, u)
        elif at_hi and not at_lo:                     # load at L_max  =>  lambda <= u
            hi = min(hi, u)
        elif not at_lo and not at_hi:                 # load interior  =>  lambda == u
            lo, hi = max(lo, u), min(hi, u)
        pv_lo, pv_hi = pv <= tol, pv >= pv_cap - tol
        if pv_hi and not pv_lo:                       # PV at capacity =>  lambda >= c_PV
            lo = max(lo, c_pv)
        elif pv_lo and not pv_hi:                     # PV curtailed   =>  lambda <= c_PV
            hi = min(hi, c_pv)
        elif not pv_lo and not pv_hi:                 # PV interior    =>  lambda == c_PV
            lo, hi = max(lo, c_pv), min(hi, c_pv)
        # (when a variable sits at two coinciding bounds - PV_max = 0, L_min = L_max - the
        #  hour carries no information on lambda and neither branch above applies)

        rows.append({
            "pi_imp": pi_imp[t], "pi_exp": pi_exp[t], "pv_available": pv_cap, "regime": regime,
            "load_hat": load_hat, "load": load, "pv_hat": pv_hat, "pv": pv,
            "lambda_lo": lo, "lambda": hr["dual_balance"][t], "lambda_hi": hi,
        })

    ladder = pd.DataFrame(rows, index=hr.index)
    ladder["match"] = (np.isclose(ladder.load_hat, ladder.load, atol=1e-6)
                       & np.isclose(ladder.pv_hat, ladder.pv, atol=1e-6))
    ladder["lambda_unique"] = ladder.lambda_hi - ladder.lambda_lo <= tol
    return ladder
