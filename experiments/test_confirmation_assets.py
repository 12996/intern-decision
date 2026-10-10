"""Binary preservation over bounded ASCII framing, including Windows CRLF."""

import base64
import io
import unittest

from confirmation_assets import Base64Reader


class TransportTests(unittest.TestCase):
    def test_every_byte_roundtrips_with_crlf(self):
        original = bytes(range(256)) * 1000
        encoded = base64.b64encode(original)
        lines = b"\r\n".join(encoded[i:i+65536] for i in range(0, len(encoded), 65536)) + b"\r\n"
        reader = Base64Reader(io.BytesIO(lines))
        pieces = []
        while piece := reader.read(10240):
            pieces.append(piece)
        self.assertEqual(b"".join(pieces), original)

    def test_invalid_or_unbounded_encoding_rejected(self):
        for value in (b"not base64!\n", b"A" * 65539):
            with self.assertRaises(ValueError):
                Base64Reader(io.BytesIO(value)).read(10240)


if __name__ == "__main__":
    unittest.main()
