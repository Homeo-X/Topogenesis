import unittest

import jax.numpy as jnp
import numpy as np

from topogenesis.engine import (
    SigmaFieldGeometric,
    pad_agent_arrays,
    stable_step_field_pde,
)


def _make_field(size=16):
    field = jnp.zeros((size, size, size, 3), dtype=jnp.float32)
    return field.at[..., 2].set(1.0)


def _make_agents(n, size=16, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(
        rng.uniform(1.0, size - 2.0, size=(n, 3)), dtype=jnp.float32)
    energies = jnp.asarray(rng.uniform(0.2, 1.0, size=(n,)), dtype=jnp.float32)
    return positions, energies


class PadAgentArraysTests(unittest.TestCase):
    def test_pads_to_power_of_two_capacity(self):
        for n, expected_cap in [(1, 8), (3, 8), (9, 16), (16, 16), (17, 32)]:
            positions, energies = _make_agents(n)
            pos_pad, eng_pad = pad_agent_arrays(positions, energies)
            self.assertEqual(pos_pad.shape, (expected_cap, 3))
            self.assertEqual(eng_pad.shape, (expected_cap,))
            self.assertTrue(bool(jnp.all(eng_pad[n:] == 0.0)))

    def test_power_of_two_input_is_returned_unchanged(self):
        positions, energies = _make_agents(8)
        pos_pad, eng_pad = pad_agent_arrays(positions, energies)
        self.assertIs(pos_pad, positions)
        self.assertIs(eng_pad, energies)

    def test_empty_population_pads_to_min_capacity(self):
        positions = jnp.zeros((0, 3), dtype=jnp.float32)
        energies = jnp.zeros((0,), dtype=jnp.float32)
        pos_pad, eng_pad = pad_agent_arrays(positions, energies)
        self.assertEqual(pos_pad.shape, (8, 3))
        self.assertEqual(eng_pad.shape, (8,))


class PaddedKernelEquivalenceTests(unittest.TestCase):
    def test_padded_step_matches_unpadded_step(self):
        field = _make_field()
        for n in (1, 3, 5, 7):
            positions, energies = _make_agents(n, seed=n)
            direct = stable_step_field_pde(field, positions, energies)
            padded = stable_step_field_pde(
                field, *pad_agent_arrays(positions, energies))
            self.assertTrue(
                bool(jnp.allclose(direct, padded, atol=0.0, rtol=0.0)),
                f"padded result diverged for n={n}")

    def test_varying_population_uses_bounded_compile_cache(self):
        sigma = SigmaFieldGeometric((16, 16, 16))
        if hasattr(stable_step_field_pde, "_clear_cache"):
            stable_step_field_pde._clear_cache()
        for n in range(1, 21):
            positions, energies = _make_agents(n, seed=n)
            sigma.step(positions, energies)
            self.assertTrue(bool(jnp.all(jnp.isfinite(sigma.phi))))
        if hasattr(stable_step_field_pde, "_cache_size"):
            # Populations 1..20 pad to capacities {8, 16, 32}.
            self.assertLessEqual(stable_step_field_pde._cache_size(), 3)

    def test_extinct_population_field_step_is_finite(self):
        sigma = SigmaFieldGeometric((16, 16, 16))
        sigma.step(jnp.zeros((0, 3), dtype=jnp.float32),
                   jnp.zeros((0,), dtype=jnp.float32))
        self.assertTrue(bool(jnp.all(jnp.isfinite(sigma.phi))))


if __name__ == "__main__":
    unittest.main()
