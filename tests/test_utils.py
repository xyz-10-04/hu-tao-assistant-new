"""utils.safe_eval 的安全性测试。

用标准库 unittest 写，不额外装依赖：python -m unittest discover -s tests -t .
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import safe_eval  # noqa: E402
from tools_api import calculate  # noqa: E402


class TestSafeEval(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(safe_eval("1+2*3"), 7)
        self.assertEqual(safe_eval("(8+2)*3"), 30)
        self.assertEqual(safe_eval("10%3"), 1)
        self.assertEqual(safe_eval("7//2"), 3)

    def test_unary_minus(self):
        self.assertEqual(safe_eval("-5+2"), -3)

    def test_division_by_zero(self):
        with self.assertRaises(ValueError):
            safe_eval("1/0")

    def test_rejects_power_operator(self):
        # ** 不在白名单里，9**9**9 这种 DoS 输入必须拦住
        with self.assertRaises(ValueError):
            safe_eval("9**9**9")

    def test_rejects_code_injection(self):
        for payload in [
            "__import__('os').system('echo hacked')",
            "open('x','w')",
            "lambda: 1",
            "[i for i in range(10)]",
        ]:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    safe_eval(payload)

    def test_rejects_too_long_expression(self):
        with self.assertRaises(ValueError):
            safe_eval("1+" * 60 + "1")

    def test_calculate_formats_integral_float_as_int(self):
        self.assertEqual(calculate("4/2"), "2")
        self.assertEqual(calculate("1/3")[:4], "0.33")


if __name__ == "__main__":
    unittest.main()
