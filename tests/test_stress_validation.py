import math
import unittest

import jax
import jax.numpy as jnp

from topogenesis.engine import stable_step_field_pde
from topogenesis.npc import (
    AffectField,
    CommunicationIntent,
    NeedPressure,
    OtherMindModel,
    SocialMemory,
    ViabilityState,
    interpret_intent,
    simulate_future,
)


class StressValidationTests(unittest.TestCase):
    def test_npc_cognition_survives_repeated_updates(self):
        affect = AffectField()
        listener = OtherMindModel(
            agent_id="guard",
            trust=0.55,
            respect=0.45,
            uncertainty=0.35,
        )
        memory = SocialMemory(max_events=128)

        for step in range(5_000):
            phase = (step % 100) / 100.0
            viability = ViabilityState(
                energy=1.0 - 0.75 * phase,
                bodily_integrity=0.95 - 0.20 * phase,
                memory_integrity=0.9,
                prediction_coherence=0.85 - 0.60 * phase,
                social_stability=0.5 + 0.3 * (1.0 - phase),
                attachment_integrity=0.65,
                environmental_safety=0.4 + 0.5 * (1.0 - phase),
            )
            needs = NeedPressure.from_viability(viability)
            affect.update(
                prediction_error=needs.epistemic,
                uncertainty=max(needs.epistemic, needs.safety),
                threat=needs.safety,
                social_support=1.0 - needs.social,
                control_feedback=1.0 - needs.total,
            )
            intent = CommunicationIntent(
                target_agent="guard",
                intended_effect="coordinate",
                belief_to_modify=f"resource_route_{step % 7}",
                confidence=0.65,
                emotional_weight=affect.threat_salience,
                social_risk=needs.safety,
                urgency=needs.metabolic,
            )
            interpretation = interpret_intent(intent, listener, affect)
            listener.update_belief(
                intent.belief_to_modify,
                interpretation.accepted_confidence,
                source_trust=listener.trust,
            )
            memory.remember(
                agent_id="guard",
                kind="coordination",
                salience=interpretation.accepted_confidence,
                valence=1.0 - interpretation.suspicion_delta,
                claim=intent.belief_to_modify,
            )
            outcome = simulate_future(
                action="seek_food" if needs.metabolic > needs.epistemic else "verify",
                needs=needs,
                affect=affect,
                listener=listener,
                intent=intent,
            )

            self.assertTrue(all(math.isfinite(value) for value in affect.as_vector()))
            self.assertTrue(all(0.0 <= value <= 1.0 for value in affect.as_vector()))
            self.assertTrue(math.isfinite(outcome.value))
            self.assertLessEqual(len(memory.events), memory.max_events)

    def test_sigma_field_jit_stress_remains_finite_and_normalized(self):
        field = jnp.zeros((8, 8, 8, 3), dtype=jnp.float32)
        field = field.at[..., 2].set(1.0)
        positions = jnp.asarray([
            [-100.0, -100.0, -100.0],
            [0.0, 0.0, 0.0],
            [7.0, 7.0, 7.0],
            [100.0, 100.0, 100.0],
        ], dtype=jnp.float32)
        energies = jnp.asarray([20.0, 1.0, 0.5, 20.0], dtype=jnp.float32)
        for _ in range(256):
            field = stable_step_field_pde(
                field,
                positions,
                energies,
                dt=0.5,
                D=1.5,
                decay=0.0,
                pump_gain=2.0,
            )

        norms = jnp.linalg.norm(field, axis=-1)
        self.assertTrue(bool(jnp.all(jnp.isfinite(field))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(norms))))
        self.assertLess(float(jnp.max(jnp.abs(norms - 1.0))), 2e-3)


if __name__ == "__main__":
    unittest.main()
