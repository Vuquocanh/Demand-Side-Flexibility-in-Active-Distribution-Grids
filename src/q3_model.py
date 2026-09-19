"""Question 3 model extending FlexibleConsumerModel."""
from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from .model import FlexibleConsumerModel


class ModelQ3(FlexibleConsumerModel):
    """Question 3: Quadratic disutility with a daily minimum energy requirement constraint.

    Direction: MINIMIZE total cost (procurement cost + disutility penalty)
    """

    def build(self) -> "ModelQ3":
        d, m, T = self.data, self.m, self.T

        self._add_common_variables()

        ref_load = getattr(d, "reference_load", getattr(d, "l_ref", None))
        if ref_load is None:
            ref_load = [0.0] * len(T)

        c_Q = getattr(d, "quadratic_disutility", getattr(d, "c_Q", getattr(d, "c_q", 1.0)))
        E_min = getattr(d, "min_daily_energy_kWh", getattr(d, "E_min", getattr(d, "min_daily_energy", 0.0)))

        self.expr["disutility"] = gp.quicksum(
            c_Q * (self.var["load"][t] - ref_load[t]) ** 2 for t in T
        )
        self.expr["procurement_cost"] = gp.quicksum(
            self._money_terms(t) for t in T
        )
        m.setObjective(
            self.expr["disutility"] + self.expr["procurement_cost"], GRB.MINIMIZE
        )

        self._add_common_constraints()


        self.con["min_daily_energy"] = m.addConstr(
            E_min - gp.quicksum(self.var["load"][t] for t in T) <= 0,
            name="min_daily_energy"
        )

        m.update()
        return self