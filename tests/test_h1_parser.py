# tests/test_h1_parser.py
from app.tools_nmrbank import _parse_h1_ppm

def test_parse_h1_ignores_hz_and_counts():
    s = "δ 7.26 (d, J=8.6 Hz, 2H), 6.90 (dd, J=8.6, 2.5 Hz, 1H), 3.78 (s, 3H)"
    out = _parse_h1_ppm(s)
    assert all(0.0 <= x <= 12.5 for x in out)
    # should NOT include 8.6, 2.5, 2, 1, or 3
    for bad in (8.6, 2.5, 2.0, 1.0, 3.0):
        assert bad not in out

