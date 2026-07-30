"""
Unit tests for ConfigManager
"""

import unittest
from app.core.config import get_config


class TestConfig(unittest.TestCase):

    def test_config_loader(self):
        cfg = get_config()
        self.assertIsNotNone(cfg.raw)
        self.assertIn(cfg.device, ["cuda", "cpu"])
        self.assertEqual(cfg.pitch_length_m, 105.0)
        self.assertEqual(cfg.pitch_width_m, 68.0)
        self.assertTrue(cfg.output_dir.name == "outputs")


if __name__ == "__main__":
    unittest.main()
