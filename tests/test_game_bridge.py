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
        self.assertIn(npc["dominant_need"], {
            "metabolic", "repair", "mnemonic", "epistemic",
            "social", "attachment", "safety",
        })

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


if __name__ == "__main__":
    unittest.main()
