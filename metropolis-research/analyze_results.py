
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("results")
DATA_FILE = RESULTS / "observables_summary.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "observables_summary.csv not found. "
        "Run the upgraded main.py first."
    )

with DATA_FILE.open(newline="") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    for key in row:
        if key != "L":
            row[key] = float(row[key])
    row["L"] = int(row["L"])

lattice_sizes = sorted({row["L"] for row in rows})
Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))

print(f"Exact infinite-lattice Tc = {Tc_exact:.5f}")
print("\nPeak estimates from simulated temperature grid")
print("-" * 70)
print(f"{'L':>5} {'Cv peak T':>12} {'Chi peak T':>12} {'Binder at peak':>16}")

for L in lattice_sizes:
    subset = sorted(
        [row for row in rows if row["L"] == L],
        key=lambda row: row["T"],
    )

    cv_peak = max(subset, key=lambda row: row["heat_capacity"])
    chi_peak = max(subset, key=lambda row: row["susceptibility"])

    print(
        f"{L:5d} "
        f"{cv_peak['T']:12.4f} "
        f"{chi_peak['T']:12.4f} "
        f"{cv_peak['binder_cumulant']:16.4f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for L in lattice_sizes:
    subset = sorted(
        [row for row in rows if row["L"] == L],
        key=lambda row: row["T"],
    )

    T = [row["T"] for row in subset]
    U = [row["binder_cumulant"] for row in subset]
    U_se = [row.get("binder_cumulant_se", np.nan) for row in subset]
    Cv = [row["heat_capacity"] for row in subset]

    axes[0].errorbar(
        T, U, yerr=U_se, marker="o", markersize=3,
        capsize=2, label=f"L={L}"
    )
    axes[1].plot(T, Cv, marker="o", markersize=3, label=f"L={L}")

for ax in axes:
    ax.axvline(Tc_exact, linestyle="--", color="black",
               label=f"Exact Tc={Tc_exact:.4f}")
    ax.set_xlabel("Temperature T")
    ax.grid(alpha=0.25)
    ax.legend()

axes[0].set_title("Binder cumulant")
axes[0].set_ylabel("U4")
axes[1].set_title("Heat capacity")
axes[1].set_ylabel("Cv per spin")

fig.tight_layout()
fig.savefig(RESULTS / "critical_analysis.png", dpi=200)
plt.show()

print(f"\nSaved: {RESULTS / 'critical_analysis.png'}")
