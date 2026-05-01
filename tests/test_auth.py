from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from enterprise_auth import create_access_token, decode_token
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    create_access_token = decode_token = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class AuthTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        if IMPORT_ERROR is not None:
            self.skipTest(f"Auth dependencies unavailable in the current runtime: {IMPORT_ERROR}")

    def test_token_roundtrip_preserves_core_claims(self) -> None:
        user = SimpleNamespace(id=7, email="manager@example.com", role="manager")

        token = create_access_token(user)
        payload = decode_token(token)

        self.assertEqual(payload["sub"], "7")
        self.assertEqual(payload["email"], "manager@example.com")
        self.assertEqual(payload["role"], "manager")
        self.assertIn("exp", payload)


if __name__ == "__main__":
    unittest.main()
