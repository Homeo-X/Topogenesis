import unittest

import jax.numpy as jnp


class ModuleLayoutTests(unittest.TestCase):
    """Each extracted subsystem module is importable and usable directly."""

    def test_constants_module(self):
        from topogenesis import constants
        self.assertGreater(constants.BODY_VEC_LEN, 0)
        self.assertGreaterEqual(constants.FIELD_OBS_DIM, constants.FIELD_PATCH_DIM)

    def test_config_module(self):
        from topogenesis.config import TopogenesisConfig, apply_ablations
        config = apply_ablations(TopogenesisConfig(), ["affect"])
        self.assertFalse(config.use_affect)

    def test_fields_module(self):
        from topogenesis.fields.sigma import SigmaFieldGeometric
        sigma = SigmaFieldGeometric((8, 8, 8))
        sigma.step(jnp.zeros((1, 3), dtype=jnp.float32),
                   jnp.ones((1,), dtype=jnp.float32))
        self.assertTrue(bool(jnp.all(jnp.isfinite(sigma.phi))))

    def test_body_module(self):
        from topogenesis.body.body_state import AgentBodyPhys, build_rich_body
        from topogenesis.constants import MAX_JOINTS
        body = AgentBodyPhys(start_pos=(2.0, 2.0, 1.0))
        rich = build_rich_body(body)  # padded to the fixed maximum morphology
        self.assertEqual(len(rich.joint_angles), MAX_JOINTS)

    def test_world_module(self):
        from topogenesis.body.body_state import AgentBodyPhys
        from topogenesis.world.world3d import World3D
        world = World3D(size=(8, 8, 8))
        body = AgentBodyPhys(start_pos=(4.0, 4.0, 1.0))
        obs = world.obs_dict(body)
        self.assertIn("energy", obs)
        world.advance_field([body])
        world.advance_field([])  # extinct population must not crash

    def test_evolution_module(self):
        import numpy as np
        from topogenesis.evolution.genome import Genome
        genome = Genome(np.random.default_rng(0))
        clone = Genome.from_dict(genome.to_dict())
        self.assertEqual(genome.n_motors, clone.n_motors)

    def test_agent_module(self):
        import jax
        from topogenesis.cognition.agent import TopogenesisAgent
        from topogenesis.config import TopogenesisConfig
        agent = TopogenesisAgent(TopogenesisConfig(), jax.random.PRNGKey(0),
                                 num_agents=1, self_idx=0)
        self.assertIsNotNone(agent.genome)

    def test_engine_facade_reexports_everything(self):
        import topogenesis.engine as engine
        for name in engine.__all__:
            self.assertTrue(hasattr(engine, name), name)


if __name__ == "__main__":
    unittest.main()
