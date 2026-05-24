import unittest

from topogenesis.research import (
    default_evidence_gates,
    default_functionalist_ladder,
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


if __name__ == "__main__":
    unittest.main()
