import unittest
import tempfile
import os

import historical_registry as registry


class RegistryTests(unittest.TestCase):

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            original = registry.REGISTRY_FILE

            try:
                registry.REGISTRY_FILE = os.path.join(
                    folder, "records.json"
                )

                record = {
                    "record_id": "TEST-001",
                    "description": "Integrity test"
                }

                success, message = registry.add_record(record)

                self.assertTrue(success)

                loaded = registry.get_records()

                self.assertEqual(len(loaded), 1)
                self.assertEqual(
                    loaded[0]["record_id"],
                    "TEST-001"
                )

            finally:
                registry.REGISTRY_FILE = original

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            original = registry.REGISTRY_FILE

            try:
                registry.REGISTRY_FILE = os.path.join(
                    folder, "records.json"
                )

                record = {
                    "record_id": "TEST-002",
                    "description": "Duplicate test"
                }

                first, _ = registry.add_record(record)
                second, _ = registry.add_record(record)

                self.assertTrue(first)
                self.assertFalse(second)

            finally:
                registry.REGISTRY_FILE = original


if __name__ == "__main__":
    unittest.main()
