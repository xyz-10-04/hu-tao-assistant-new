"""HTTP 接口契约测试。

这里只测"不需要调用大模型"的接口，所以不消耗 API 额度，可以在 CI 里跑。
注意：import main_api 会创建 Agent（不联网），需要 .env 里有 DEEPSEEK_API_KEY 才能构造。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

HAS_KEY = bool(os.getenv("DEEPSEEK_API_KEY"))


@unittest.skipUnless(HAS_KEY, "缺少 DEEPSEEK_API_KEY，跳过接口测试")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import main_api

        cls.client = TestClient(main_api.app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_roll_in_range(self):
        for _ in range(20):
            value = self.client.get("/roll").json()["result"]
            self.assertGreaterEqual(value, 1)
            self.assertLessEqual(value, 6)

    def test_time_format(self):
        text = self.client.get("/time").json()["time"]
        self.assertIn("年", text)
        self.assertIn(":", text)

    def test_calc_post_handles_plus(self):
        """回归： '+' 放进 query string 会被解码成空格，所以推荐 POST body。"""
        r = self.client.post("/calc", json={"expr": "3+8"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["result"], "11")

    def test_calc_get_with_encoded_plus(self):
        r = self.client.get("/calc?expr=3%2B8")
        self.assertEqual(r.json()["result"], "11")

    def test_calc_division_by_zero_is_400(self):
        r = self.client.post("/calc", json={"expr": "1/0"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("detail", r.json())

    def test_calc_rejects_injection(self):
        r = self.client.post("/calc", json={"expr": "__import__('os').system('calc')"})
        self.assertEqual(r.status_code, 400)

    def test_chat_rejects_empty_message(self):
        r = self.client.post("/chat", json={"message": ""})
        self.assertEqual(r.status_code, 422)

    def test_chat_rejects_too_long_message(self):
        r = self.client.post("/chat", json={"message": "x" * 501})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
