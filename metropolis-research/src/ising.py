
import numpy as np


def initialize_lattice(L, seed=None, ordered=False):
    """Create an L x L lattice of Ising spins."""
    if L < 2:
        raise ValueError("L must be at least 2")

    if ordered:
        return np.ones((L, L), dtype=np.int8)

    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(L, L)).astype(np.int8)


def total_energy(spins, J=1.0):
    """
    Total energy for a 2D square-lattice Ising model.
    Periodic boundary conditions; each bond counted once.
    """
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    return float(-J * np.sum(spins * (right + down)))


def magnetization(spins):
    """Total magnetization M = sum(s_i)."""
    return int(np.sum(spins))


def local_delta_energy(spins, i, j, J=1.0):
    """Energy difference caused by flipping spin (i, j)."""
    L = spins.shape[0]

    neighbours = (
        spins[(i + 1) % L, j]
        + spins[(i - 1) % L, j]
        + spins[i, (j + 1) % L]
        + spins[i, (j - 1) % L]
    )

    return float(2.0 * J * spins[i, j] * neighbours)


def metropolis_step(spins, T, rng, J=1.0):
    """
    Attempt one random spin flip.
    k_B = 1. Return True if accepted, False otherwise.
    """
    if T <= 0:
        raise ValueError("T must be positive")

    L = spins.shape[0]
    i = rng.integers(L)
    j = rng.integers(L)

    dE = local_delta_energy(spins, i, j, J)

    if dE <= 0 or rng.random() < np.exp(-dE / T):
        spins[i, j] *= -1
        return True

    return False


def monte_carlo_sweep(spins, T, rng, J=1.0):
    """Perform L^2 random spin-flip attempts."""
    L = spins.shape[0]
    accepted = 0

    for _ in range(L * L):
        accepted += metropolis_step(spins, T, rng, J)

    return accepted / (L * L)


def autocorrelation_time(series, max_lag=None):
    """
    Estimate integrated autocorrelation time using the
    initial-positive-sequence approach.

    This is a practical diagnostic, not a substitute for
    a full uncertainty analysis near criticality.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)

    if n < 4:
        return 0.5

    x = x - np.mean(x)
    variance = np.dot(x, x) / n

    if variance == 0:
        return 0.5

    if max_lag is None:
        max_lag = min(n // 2, 1000)

    tau = 0.5

    for lag in range(1, max_lag + 1):
        rho = np.dot(x[:-lag], x[lag:]) / ((n - lag) * variance)

        if rho <= 0:
            break

        tau += rho

    return float(max(0.5, tau))


def simulate_temperature(
    L,
    T,
    equil_sweeps=1000,
    sample_sweeps=3000,
    seed=42,
    J=1.0,
    ordered=False,
):
    """
    Simulate one temperature.

    Each measurement is recorded after one full Monte Carlo
    sweep. Results include raw samples for further analysis.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if equil_sweeps < 0 or sample_sweeps < 2:
        raise ValueError("Invalid sweep counts")

    rng = np.random.default_rng(seed)
    spins = initialize_lattice(L, seed=seed, ordered=ordered)
    N = L * L

    for _ in range(equil_sweeps):
        monte_carlo_sweep(spins, T, rng, J)

    energies = np.empty(sample_sweeps)
    mags = np.empty(sample_sweeps)
    acceptance = np.empty(sample_sweeps)

    for sweep in range(sample_sweeps):
        acceptance[sweep] = monte_carlo_sweep(spins, T, rng, J)
        energies[sweep] = total_energy(spins, J)
        mags[sweep] = magnetization(spins)

    E = energies
    M = mags
    abs_M = np.abs(M)

    mean_E = np.mean(E)
    mean_E2 = np.mean(E ** 2)
    mean_M = np.mean(M)
    mean_M2 = np.mean(M ** 2)
    mean_abs_M = np.mean(abs_M)
    mean_M4 = np.mean(M ** 4)

    energy_per_spin = mean_E / N
    abs_m_per_spin = mean_abs_M / N

    # Heat capacity per spin, with k_B = 1.
    heat_capacity = (mean_E2 - mean_E**2) / (N * T**2)

    # Conventional fluctuation susceptibility.
    susceptibility = (mean_M2 - mean_M**2) / (N * T)

    # Useful below Tc, where finite simulations may flip
    # between positive and negative magnetization states.
    abs_m_susceptibility = (
        mean_M2 - mean_abs_M**2
    ) / (N * T)

    # Binder fourth-order cumulant.
    binder = np.nan
    if mean_M2 > 0:
        binder = 1.0 - mean_M4 / (3.0 * mean_M2**2)

    tau_E = autocorrelation_time(E)
    tau_M = autocorrelation_time(abs_M)

    return {
        "L": L,
        "T": float(T),
        "energy_per_spin": float(energy_per_spin),
        "abs_magnetization_per_spin": float(abs_m_per_spin),
        "heat_capacity": float(max(0.0, heat_capacity)),
        "susceptibility": float(max(0.0, susceptibility)),
        "abs_m_susceptibility": float(max(0.0, abs_m_susceptibility)),
        "binder_cumulant": float(binder),
        "acceptance_rate": float(np.mean(acceptance)),
        "tau_energy": tau_E,
        "tau_abs_magnetization": tau_M,
        "energies": E,
        "magnetizations": M,
        "final_spins": spins.copy(),
    }
