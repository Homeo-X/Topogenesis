import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from topogenesis.game_bridge.server import BridgeHandler
from topogenesis.game_bridge.state import GameBridgeState


class GameBridgeTests(unittest.TestCase):
    def test_bridge_state_steps_and_returns_bounded_snapshot(self):
        state = GameBridgeState()
        snapshot = state.step({
            "delta": 0.25,
            "pressures": {
                "npc_mara": {
                    "hazard": 0.8,
                    "resource": 0.1,
                    "player_help": 0.4,
                    "player_threat": 0.0,
                }
            },
        })

        self.assertEqual(snapshot["version"], 1)
        self.assertIn("npc_mara", snapshot["npcs"])
        npc = snapshot["npcs"]["npc_mara"]
        self.assertTrue(0.0 <= npc["need_total"] <= 1.0)
        self.assertTrue(0.0 <= npc["affect_stability"] <= 1.0)
        self.assertTrue(1 <= npc["future_depth"] <= 3)
        self.assertTrue(0.0 <= npc["imagination_fatigue"] <= 1.0)
        self.assertIn("semantic_memory", npc)
        self.assertIn(npc["dominant_need"], {
            "metabolic", "repair", "mnemonic", "epistemic",
            "social", "attachment", "safety",
        })
        world = state.world_snapshot()
        self.assertIn("locations", world)
        self.assertIn("npcs", world)
        self.assertIn("core_engine", world)
        self.assertFalse(world["core_engine"]["enabled"])
        self.assertGreaterEqual(world["world"]["carrying_capacity"], 800)

    def test_bridge_can_advertise_full_core_engine_layer(self):
        state = GameBridgeState(full_engine_enabled=True, full_engine_interval=999)
        world = state.world_snapshot()
        self.assertTrue(world["core_engine"]["enabled"])
        self.assertEqual(world["core_engine"]["mode"], "full_topogenesis_agent_self_maintain")
        self.assertFalse(world["core_engine"]["initialized"])

    def test_bridge_http_step_endpoint(self):
        class TestHandler(BridgeHandler):
            state = GameBridgeState()

        server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            body = json.dumps({
                "delta": 0.1,
                "pressures": {"npc_ovan": {"hazard": 0.2, "resource": 0.7}},
            })
            conn.request(
                "POST",
                "/step",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertIn("npc_ovan", payload["npcs"])
        finally:
            server.shutdown()
            server.server_close()

    def test_bridge_http_world_snapshot_endpoint(self):
        class TestHandler(BridgeHandler):
            state = GameBridgeState()

        server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            conn.request("GET", "/world_snapshot")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            conn.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(payload["mode"], "offline_world")
            self.assertGreater(len(payload["locations"]), 10)
            self.assertGreater(len(payload["npcs"]), 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_bridge_director_snapshot_exposes_game_signals(self):
        state = GameBridgeState()
        state.step({
            "delta": 0.1,
            "pressures": {
                "npc_ovan": {"hazard": 0.8, "resource": 0.1},
                "npc_sera": {"hazard": 0.2, "resource": 0.6},
            },
        })
        snapshot = state.director_snapshot()

        self.assertEqual(snapshot["version"], 1)
        self.assertIn(snapshot["tone"], {"stable", "uneasy", "crisis", "collapse"})
        self.assertTrue(0.0 <= snapshot["pressure_score"] <= 1.0)
        self.assertIn("objective", snapshot)
        self.assertIn("suggested_ui", snapshot)
        self.assertIn("focus_need", snapshot["suggested_ui"])


if __name__ == "__main__":
    unittest.main()
