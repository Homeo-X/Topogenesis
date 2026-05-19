import unittest

from topogenesis.npc import (
    AffectField,
    Attachment,
    CommunicationIntent,
    NeedPressure,
    OtherMindModel,
    SocialMemory,
    ViabilityState,
    interpret_intent,
    simulate_future,
)


class NpcCognitionTests(unittest.TestCase):
    def test_need_pressure_emerges_from_viability_deficits(self):
        viability = ViabilityState(
            energy=0.2,
            bodily_integrity=0.9,
            memory_integrity=0.8,
            prediction_coherence=0.4,
            social_stability=0.7,
            attachment_integrity=0.5,
            environmental_safety=0.9,
        )
        pressure = NeedPressure.from_viability(viability)

        self.assertEqual(pressure.dominant, "metabolic")
        self.assertGreater(pressure.total, 0.25)
        self.assertLess(pressure.cognitive_modulators()["planning_depth"], 6.0)

    def test_affect_is_continuous_and_modulates_rumor_susceptibility(self):
        affect = AffectField()
        base = affect.rumor_susceptibility

        affect.update(prediction_error=0.9, uncertainty=0.8, threat=0.7)

        self.assertLess(affect.stability, 0.75)
        self.assertGreater(affect.threat_salience, 0.1)
        self.assertGreater(affect.rumor_susceptibility, base)

    def test_attachment_rupture_creates_loss_pressure(self):
        attachment = Attachment(
            entity_id="village_elder",
            importance=0.9,
            dependency=0.8,
            trust=0.7,
        )
        affect = AffectField()
        rupture = attachment.rupture_strength(observed_absence=1.0)

        affect.update(
            prediction_error=0.4,
            uncertainty=0.6,
            attachment_delta=-rupture,
        )

        self.assertGreater(affect.attachment_loss, 0.5)
        self.assertLess(affect.attachment_security, 0.6)

    def test_communication_is_social_intervention_not_text_response(self):
        listener = OtherMindModel(
            agent_id="guard",
            trust=0.8,
            respect=0.6,
            fear=0.1,
            uncertainty=0.2,
        )
        intent = CommunicationIntent(
            target_agent="guard",
            intended_effect="increase_trust",
            belief_to_modify="player_helped_village",
            confidence=0.9,
            emotional_weight=0.3,
            social_risk=0.1,
            urgency=0.2,
        )

        self.assertTrue(intent.should_speak(listener))
        interpretation = interpret_intent(intent, listener, AffectField())
        self.assertGreater(interpretation.accepted_confidence, 0.4)
        self.assertLess(interpretation.suspicion_delta, 0.5)

    def test_social_memory_tracks_reputation(self):
        memory = SocialMemory()
        memory.remember(agent_id="player", kind="aid", salience=0.9, valence=0.8)
        memory.remember(agent_id="player", kind="insult", salience=0.2, valence=-0.6)

        self.assertGreater(memory.reputation("player"), 0.4)

    def test_imagination_compares_action_futures(self):
        needs = NeedPressure.from_viability(ViabilityState(
            energy=0.15,
            bodily_integrity=0.75,
            prediction_coherence=0.7,
        ))
        affect = AffectField(cognitive_load=0.2, prediction_coherence=0.8)

        seek_food = simulate_future(action="seek_food", needs=needs, affect=affect)
        investigate = simulate_future(action="investigate", needs=needs, affect=affect)

        self.assertGreater(seek_food.stability_delta, investigate.stability_delta)


if __name__ == "__main__":
    unittest.main()
