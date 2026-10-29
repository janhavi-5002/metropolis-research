
import unittest
import numpy as np

from src.ising import (
    initialize_lattice,
    total_energy,
    magnetization,
    local_delta_energy,
    metropolis_step,
)


class TestIsingModel(unittest.TestCase):

    def test_random_lattice_values(self):
        spins = initialize_lattice(8, seed=1)
        self.assertEqual(spins.shape, (8, 8))
        self.assertTrue(np.all(np.isin(spins, [-1, 1])))

    def test_reproducible_seed(self):
        a = initialize_lattice(8, seed=123)
        b = initialize_lattice(8, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_ordered_state_energy(self):
        L = 8
        J = 1.0
        spins = initialize_lattice(L, ordered=True)
        self.assertAlmostEqual(total_energy(spins, J), -2 * J * L**2)

    def test_zero_magnetization(self):
        spins = np.array([[1, -1], [-1, 1]], dtype=np.int8)
        self.assertEqual(magnetization(spins), 0)

    def test_local_energy_difference(self):
        spins = initialize_lattice(8, seed=20)
        before = total_energy(spins)
        predicted = local_delta_energy(spins, 3, 4)

        flipped = spins.copy()
        flipped[3, 4] *= -1
        actual = total_energy(flipped) - before

        self.assertAlmostEqual(predicted, actual)

    def test_metropolis_keeps_valid_spins(self):
        spins = initialize_lattice(8, seed=2)
        rng = np.random.default_rng(5)

        metropolis_step(spins, T=2.0, rng=rng)

        self.assertTrue(np.all(np.isin(spins, [-1, 1])))

    def test_zero_temperature_rejected(self):
        spins = initialize_lattice(4, seed=1)
        rng = np.random.default_rng(2)

        with self.assertRaises(ValueError):
            metropolis_step(spins, T=0, rng=rng)


if __name__ == "__main__":
    unittest.main()
