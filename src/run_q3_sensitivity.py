import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .data_loader import load_question
from .q3_model import ModelQ3
from .plotting import plot_q3_cq_profiles, plot_q3_emin_sensitivity


def run_e_min_sweep(base_data):
    """Experiment 1: Monitoring the lowest electricity consumption of the day E_min"""
    e_min_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140]
    records = []

    for e_val in e_min_values:
        data = copy.deepcopy(base_data)
        data.min_daily_energy_kWh = e_val  # alternative E_min parameter

        model = ModelQ3(data).build()
        try:
            res = model.solve()
            total_load = res.hourly["load"].sum()
            # \mu^{daily} >= 0
            mu_daily = abs(res.duals.get("min_daily_energy", 0.0))

            # Calculate total deviation: sum((L_t - L_ref_t)^2)
            ref_load = data.reference_load
            sq_dev = ((res.hourly["load"] - ref_load) ** 2).sum()

            records.append(
                {
                    "E_min": e_val,
                    "status": res.status,
                    "objective": res.objective,
                    "total_load": total_load,
                    "mu_daily": mu_daily,
                    "squared_dev": sq_dev,
                }
            )
        except RuntimeError:
            print(f"E_min = {e_val} Beyond the feasible region (Infeasible)")

    return pd.DataFrame(records)


def run_c_q_sweep(base_data):
    """Experiment 2: Monitoring the coefficient of dissatisfaction c_Q"""
    cq_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    records = []

    for cq in cq_values:
        data = copy.deepcopy(base_data)
        data.quadratic_disutility = cq  # alternative c_Q parameter

        model = ModelQ3(data).build()
        res = model.solve()

        records.append(
            {
                "c_Q": cq,
                "objective": res.objective,
                "total_load": res.hourly["load"].sum(),
                "mu_daily": res.duals.get("min_daily_energy", 0.0),
                "max_load_peak": res.hourly["load"].max(),
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    base_data = load_question("Q3")

    df_emin = run_e_min_sweep(base_data)
    print("--- E_min Sensitivity Results ---")
    print(df_emin)

    df_cq = run_c_q_sweep(base_data)
    print("\n--- c_Q Sensitivity Results ---")
    print(df_cq)

# ----------- plotting --------------
out_dir = Path("results/Q3_sensitivity")
out_dir.mkdir(parents=True, exist_ok=True)

base_data = load_question("Q3")

# 1.  E_min
df_emin = run_e_min_sweep(base_data)
plot_q3_emin_sensitivity(df_emin, save_to=out_dir / "emin_sensitivity.png")

# 2. select c_Q (e.g. 0.1, 1.0, 5.0)
selected_cq = [0.1, 1.0, 5.0]
cq_results = {}
for cq in selected_cq:
    data = copy.deepcopy(base_data)
    data.disutility_coeff_quadratic = cq
    res = ModelQ3(data).build().solve()
    cq_results[cq] = res.hourly

plot_q3_cq_profiles(cq_results, base_data, save_to=out_dir / "cq_load_profiles.png")
