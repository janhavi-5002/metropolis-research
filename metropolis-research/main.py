
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ising import simulate_temperature

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONFIG_PATH = ROOT / "configs" / "default.json"

METRICS = [
    "energy_per_spin",
    "abs_magnetization_per_spin",
    "heat_capacity",
    "susceptibility",
    "abs_m_susceptibility",
    "binder_cumulant",
    "acceptance_rate",
    "tau_energy",
    "tau_abs_magnetization",
]


def load_config():
    with CONFIG_PATH.open() as file:
        return json.load(file)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No data to save: {path}")

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_and_se(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(values))
    se = (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) >= 2 else float("nan")
    )
    return mean, se


def aggregate_runs(raw_rows):
    groups = {}

    for row in raw_rows:
        groups.setdefault((row["L"], row["T"]), []).append(row)

    summary = []

    for (L, T), group in sorted(groups.items()):
        record = {"L": L, "T": T, "independent_runs": len(group)}

        for metric in METRICS:
            mean, se = mean_and_se([r[metric] for r in group])
            record[metric] = mean
            record[f"{metric}_se"] = se

        summary.append(record)

    return summary


def plot_observables(summary, lattice_sizes):
    plots = [
        ("energy_per_spin", "Energy per spin", "E/N"),
        ("abs_magnetization_per_spin",
         "Absolute magnetization per spin", "<|M|>/N"),
        ("heat_capacity", "Heat capacity per spin", "Cv"),
        ("susceptibility", "Magnetic susceptibility", "Chi"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, plots):
        for L in lattice_sizes:
            subset = sorted(
                [r for r in summary if r["L"] == L],
                key=lambda r: r["T"],
            )

            ax.errorbar(
                [r["T"] for r in subset],
                [r[metric] for r in subset],
                yerr=[r[f"{metric}_se"] for r in subset],
                marker="o",
                markersize=3,
                capsize=2,
                label=f"L={L}",
            )

        ax.set_title(title)
        ax.set_xlabel("Temperature T (kB = 1)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    for ax in axes:
        ax.axvline(Tc, linestyle="--", color="black", alpha=0.6)

    fig.tight_layout()
    fig.savefig(RESULTS / "observables_errorbars.png", dpi=200)
    plt.close(fig)


def plot_binder(summary, lattice_sizes):
    fig, ax = plt.subplots(figsize=(8, 5))

    for L in lattice_sizes:
        subset = sorted(
            [r for r in summary if r["L"] == L],
            key=lambda r: r["T"],
        )

        ax.errorbar(
            [r["T"] for r in subset],
            [r["binder_cumulant"] for r in subset],
            yerr=[r["binder_cumulant_se"] for r in subset],
            marker="o",
            markersize=3,
            capsize=2,
            label=f"L={L}",
        )

    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    ax.axvline(Tc, linestyle="--", color="black", label=f"Exact Tc={Tc:.4f}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Binder cumulant U4")
    ax.set_title("Binder Cumulant with Independent-Run Errors")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS / "binder_errorbars.png", dpi=200)
    plt.close(fig)


def plot_final_lattice(config):
    result = simulate_temperature(
        L=config["example_lattice_size"],
        T=config["example_temperature"],
        equil_sweeps=config["equil_sweeps"],
        sample_sweeps=max(100, config["sample_sweeps"] // 2),
        seed=config["seed"] + 999999,
        J=config["coupling"],
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        result["final_spins"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_title("Final spin configuration")
    fig.colorbar(im, ax=ax, label="Spin")
    fig.tight_layout()
    fig.savefig(RESULTS / "final_lattice.png", dpi=200)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    config = load_config()

    sizes = config["lattice_sizes"]
    temperatures = np.linspace(
        config["temperature_min"],
        config["temperature_max"],
        config["temperature_points"],
    )
    repeats = config["independent_runs"]

    if repeats < 2:
        raise ValueError("independent_runs must be at least 2")

    total = len(sizes) * len(temperatures) * repeats
    raw_rows = []
    start = time.time()
    completed = 0

    print(f"Starting {total} independent simulations")

    for L in sizes:
        for ti, T in enumerate(temperatures):
            for run in range(repeats):
                completed += 1
                seed = config["seed"] + L * 1_000_000 + ti * 10_000 + run

                result = simulate_temperature(
                    L=L,
                    T=float(T),
                    equil_sweeps=config["equil_sweeps"],
                    sample_sweeps=config["sample_sweeps"],
                    seed=seed,
                    J=config["coupling"],
                )

                row = {"L": L, "T": float(T), "run": run + 1, "seed": seed}
                row.update({m: result[m] for m in METRICS})
                raw_rows.append(row)

                if completed % 10 == 0 or completed == total:
                    print(f"Completed {completed}/{total}")

    summary = aggregate_runs(raw_rows)

    write_csv(RESULTS / "independent_runs.csv", raw_rows)
    write_csv(RESULTS / "observables_summary.csv", summary)

    plot_observables(summary, sizes)
    plot_binder(summary, sizes)
    plot_final_lattice(config)

    print(f"\nCompleted in {time.time() - start:.1f} seconds")
    print("Saved independent_runs.csv")
    print("Saved observables_summary.csv")
    print("Saved observables_errorbars.png")
    print("Saved binder_errorbars.png")
    print("Saved final_lattice.png")


if __name__ == "__main__":
    main()
