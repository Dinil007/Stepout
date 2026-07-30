import unittest

from fastapi.testclient import TestClient

from app.api.main import app


class TestAIAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_ai_routes_are_registered_in_openapi(self):
        schema = self.client.get("/openapi.json").json()
        for path in [
            "/ai/match-summary",
            "/ai/team-analysis",
            "/ai/player/{player_id}",
            "/ai/coach-report",
            "/ai/query",
            "/ai/recommendations",
        ]:
            self.assertIn(path, schema["paths"])

    def test_ai_query_uses_structured_context(self):
        response = self.client.post(
            "/ai/query",
            json={"question": "Who should be Player of the Match?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("answer", payload)
        self.assertIn("structured analytics only", payload["answer"])


if __name__ == "__main__":
    unittest.main()
