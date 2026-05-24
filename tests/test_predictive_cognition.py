import unittest

from topogenesis.research import (
    SensoryConfirmationLoop,
    SensoryObservation,
    compare_prediction,
)


class PredictiveCognitionTests(unittest.TestCase):
    def test_first_sensory_input_creates_persistent_world_entity(self):
        loop = SensoryConfirmationLoop()

        signal = loop.observe(SensoryObservation(
            entity_id="door",
            features={"open": 0.0, "distance": 0.4},
            tick=1,
            confidence=0.9,
            salience=0.8,
        ))

        self.assertEqual(signal.novelty, 1.0)
        self.assertIn("door", loop.world_model.entities)
        prediction = loop.predict("door", tick=2)
        self.assertIn("open", prediction.expected_features)
        self.assertAlmostEqual(prediction.expected_features["distance"], 0.4)

    def test_repeated_confirmation_strengthens_model_and_memory(self):
        loop = SensoryConfirmationLoop()
        first = SensoryObservation(
            entity_id="lamp",
            features={"lit": 1.0, "warmth": 0.7},
            tick=1,
            confidence=0.95,
            salience=1.0,
        )
        loop.observe(first)
        before = loop.world_model.entities["lamp"].confidence

        signal = loop.observe(SensoryObservation(
            entity_id="lamp",
            features={"lit": 1.0, "warmth": 0.72},
            tick=2,
            confidence=0.95,
            salience=1.0,
        ))
        after = loop.world_model.entities["lamp"].confidence

        self.assertGreater(signal.confirmation, signal.contradiction)
        self.assertGreater(after, before)
        self.assertGreater(loop.memory.semantic_confidence("lamp:lit"), 0.0)

    def test_contradictory_input_updates_belief_without_unbounded_memory(self):
        loop = SensoryConfirmationLoop()
        loop.memory.max_episodes = 2
        loop.observe(SensoryObservation(
            entity_id="path",
            features={"blocked": 0.0},
            tick=1,
            confidence=1.0,
            salience=1.0,
        ))

        signal = loop.observe(SensoryObservation(
            entity_id="path",
            features={"blocked": 1.0},
            tick=2,
            confidence=1.0,
            salience=1.0,
        ))
        loop.observe(SensoryObservation(
            entity_id="path",
            features={"blocked": 1.0},
            tick=3,
            confidence=1.0,
            salience=0.5,
        ))

        self.assertGreater(signal.contradiction, signal.confirmation)
        self.assertGreater(loop.world_model.entities["path"].beliefs["blocked"], 0.5)
        self.assertLessEqual(len(loop.memory.episodes), 2)

    def test_prediction_comparison_has_bounded_error_terms(self):
        loop = SensoryConfirmationLoop()
        loop.observe(SensoryObservation(
            entity_id="cup",
            features={"x": 0.2, "y": 0.3},
            tick=1,
            confidence=0.8,
        ))

        prediction = loop.predict("cup", tick=2)
        signal = compare_prediction(prediction, SensoryObservation(
            entity_id="cup",
            features={"x": 0.25, "y": 0.35},
            tick=2,
            confidence=0.8,
        ))

        self.assertTrue(0.0 <= signal.prediction_error <= 1.0)
        self.assertTrue(0.0 <= signal.update_weight <= 1.0)


if __name__ == "__main__":
    unittest.main()
