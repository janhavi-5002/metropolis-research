
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("results")
INPUT = RESULTS / "observables_summary.csv"
TC_EXACT = 2.0 / math.log(1.0 + math.sqrt(2.0))

if not INPUT.exists():
    raise FileNotFoundError(
        "results/observables_summary.csv nahi mili. "
        "Project root se script run karo."
    )

with INPUT.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = reader.fieldnames or []

def find_column(candidates, required=True):
    for name in candidates:
        if name in columns:
            return name
    if required:
        raise ValueError(
            f"Required column nahi mili. Available columns: {columns}"
        )
    return None

lcol = find_column(["L", "lattice_size"])
tcol = find_column(["T", "temperature", "temp"])
ucol = find_column(
    ["binder_cumulant", "binder", "U4", "u4"]
)
secol = find_column(
    ["binder_cumulant_se", "binder_cumulant_sem",
     "binder_cumulant_std_error", "binder_cumulant_error",
     "binder_se", "U4_se"],
    required=False
)

data = {}

for row in rows:
    try:
        L = int(float(row[lcol]))
        T = float(row[tcol])
        U = float(row[ucol])
        error = (
            float(row[secol])
            if secol and row.get(secol, "")
            else float("nan")
        )
        data.setdefault(L, []).append((T, U, error))
    except (ValueError, TypeError):
        continue

for L in data:
    data[L].sort()

sizes = sorted(data)

if len(sizes) < 2:
    raise ValueError(
        f"At least 2 lattice sizes required. Found: {sizes}"
    )

def interpolate(series, temperature):
    temperatures = [point[0] for point in series]
    values = [point[1] for point in series]
    return float(np.interp(temperature, temperatures, values))

crossings = []

# Find approximate pairwise intersections.
for i, L1 in enumerate(sizes):
    for L2 in sizes[i + 1:]:
        curve1 = data[L1]
        curve2 = data[L2]

        low = max(curve1[0][0], curve2[0][0])
        high = min(curve1[-1][0], curve2[-1][0])

        grid = sorted(set(
            [p[0] for p in curve1 if low <= p[0] <= high]
            + [p[0] for p in curve2 if low <= p[0] <= high]
        ))

        if len(grid) < 2:
            continue

        differences = [
            interpolate(curve1, T) - interpolate(curve2, T)
            for T in grid
        ]

        for j in range(len(grid) - 1):
            d0 = differences[j]
            d1 = differences[j + 1]

            if d0 * d1 < 0:
                T0, T1 = grid[j], grid[j + 1]
                T_cross = T0 - d0 * (T1 - T0) / (d1 - d0)
                crossings.append(
                    (L1, L2, T_cross, T_cross - TC_EXACT)
                )
            elif d0 == 0:
                crossings.append(
                    (L1, L2, grid[j], grid[j] - TC_EXACT)
                )

RESULTS.mkdir(exist_ok=True)

with (RESULTS / "binder_crossings.csv").open(
    "w", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "L1", "L2", "estimated_crossing_T",
        "difference_from_exact_Tc"
    ])
    writer.writerows(crossings)

fig, ax = plt.subplots(figsize=(9, 6))

for L in sizes:
    curve = data[L]
    temperatures = [p[0] for p in curve]
    values = [p[1] for p in curve]
    errors = [p[2] for p in curve]

    if secol and all(np.isfinite(errors)):
        ax.errorbar(
            temperatures, values, yerr=errors,
            marker="o", capsize=3, label=f"L={L}"
        )
    else:
        ax.plot(
            temperatures, values,
            marker="o", label=f"L={L}"
        )

ax.axvline(
    TC_EXACT, color="black", linestyle="--",
    label=f"Exact Tc={TC_EXACT:.5f}"
)
ax.set_xlabel("Temperature T (J = kB = 1)")
ax.set_ylabel("Binder cumulant U4")
ax.set_title("Finite-size Binder Cumulant Analysis")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "finite_size_analysis.png", dpi=180)
plt.close(fig)

print(f"Exact infinite-lattice Tc = {TC_EXACT:.5f}")
print("Lattice sizes:", sizes)
print("Saved results/binder_crossings.csv")
print("Saved results/finite_size_analysis.png")

if crossings:
    print("\nApproximate pairwise intersections:")
    for L1, L2, Tcross, difference in crossings:
        print(
            f"L={L1} vs L={L2}: "
            f"T_cross={Tcross:.5f}, "
            f"difference from exact Tc={difference:+.5f}"
        )
else:
    print(
        "No sign-changing intersection found on the sampled grid."
    )

print(
    "\nNote: linear-interpolation crossings are exploratory, "
    "not precision critical-temperature estimates."
)
