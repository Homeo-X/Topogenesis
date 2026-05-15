import unittest

import jax
import jax.numpy as jnp

from topogenesis.engine import stable_step_field_pde


class JitRuntimeTests(unittest.TestCase):
    def test_field_kernel_runs_with_jit_enabled(self):
        self.assertFalse(bool(jax.config.jax_disable_jit))

        field = jnp.zeros((4, 4, 4, 3), dtype=jnp.float32)
        field = field.at[..., 2].set(1.0)
        positions = jnp.asarray([[1.0, 1.0, 1.0]], dtype=jnp.float32)
        energies = jnp.asarray([0.5], dtype=jnp.float32)

        step_kernel = jax.jit(stable_step_field_pde)
        evolved = step_kernel(field, positions, energies)

        self.assertEqual(evolved.shape, field.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(evolved))))


if __name__ == "__main__":
    unittest.main()
