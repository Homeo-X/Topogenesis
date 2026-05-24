import unittest

from topogenesis.research import (
    default_evidence_gates,
    default_functionalist_ladder,
    default_scaling_gates,
    incomplete_contracts,
)


class AgiResearchContractTests(unittest.TestCase):
    def test_functionalist_ladder_contracts_are_complete(self):
        contracts = default_functionalist_ladder()

        self.assertGreaterEqual(len(contracts), 5)
        self.assertEqual(incomplete_contracts(contracts), ())
        self.assertEqual(contracts[0].name, "viability")

    def test_evidence_gates_are_testable_and_include_baselines(self):
        gates = default_evidence_gates()

        self.assertGreaterEqual(len(gates), 3)
        for gate in gates:
            self.assertTrue(gate.is_testable(), gate.claim)
            self.assertGreaterEqual(len(gate.failure_modes), 1)

    def test_scaling_gates_are_actionable_and_ordered_by_load(self):
        gates = default_scaling_gates()

        self.assertGreaterEqual(len(gates), 3)
        self.assertEqual(gates[0].target_agent_count, 1)
        self.assertGreater(gates[-1].target_agent_count, gates[0].target_agent_count)
        for gate in gates:
            self.assertTrue(gate.is_actionable(), gate.name)
            self.assertIn("memory", " ".join(gate.failure_modes + gate.required_checks))


if __name__ == "__main__":
    unittest.main()
