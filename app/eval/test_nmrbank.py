#!/usr/bin/env python3
"""
Test suite for NMRBank reference lookup functionality.

Tests:
- SMILES canonicalization
- H1/C13 NMR parsing accuracy
- Database coverage for common compounds
- Lookup performance

Usage:
    pytest app/eval/test_nmrbank.py -v
    python -m app.eval.test_nmrbank  # standalone
"""

import os
import sys
import time
import pytest
from typing import List, Optional, Tuple

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.tools_nmrbank import (
    get_reference_by_smiles,
    _parse_h1_ppm,
    _parse_h1_with_integrals,
    _parse_c13_ppm,
    _smiles_key,
    warm_cache,
    clear_cache,
)


# ============================================================================
# Test Data
# ============================================================================

# Common compounds that should be in NMRBank
COMMON_COMPOUNDS = [
    ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("chloroform", "ClC(Cl)Cl"),
    ("acetone", "CC(=O)C"),
    ("methanol", "CO"),
    ("ethanol", "CCO"),
    ("acetic acid", "CC(=O)O"),
    ("anisole", "COc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("pyridine", "c1ccncc1"),
]

# H1 NMR parsing test cases: (input_string, expected_ppm_values, forbidden_values)
H1_PARSING_CASES = [
    # Basic single peak
    ("7.26 (s, 1H)", [7.26], []),

    # Multiple peaks
    ("7.26 (d, 2H), 6.90 (d, 2H), 3.78 (s, 3H)", [7.26, 6.90, 3.78], []),

    # With coupling constants - should NOT include J values
    ("δ 7.26 (d, J=8.6 Hz, 2H), 6.90 (dd, J=8.6, 2.5 Hz, 1H), 3.78 (s, 3H)",
     [7.26, 6.90, 3.78], [8.6, 2.5]),

    # Complex coupling pattern
    ("7.40 (ddd, J=8.2, 7.1, 1.2 Hz, 1H)",
     [7.40], [8.2, 7.1, 1.2]),

    # Should exclude proton counts
    ("2.15 (s, 3H)", [2.15], [3.0]),

    # Edge case: very low ppm
    ("0.88 (t, 3H), 1.26 (m, 18H)", [0.88, 1.26], [18.0]),

    # With delta symbol
    ("δ 3.75 (s, 6H)", [3.75], []),

    # Semicolon separated
    ("7.50 (d, 2H); 7.20 (d, 2H); 2.35 (s, 3H)", [7.50, 7.20, 2.35], []),
]

# C13 NMR parsing test cases
C13_PARSING_CASES = [
    ("128.5, 126.2, 125.8", [128.5, 126.2, 125.8]),
    ("δ 170.5, 128.3, 21.0", [170.5, 128.3, 21.0]),
    ("13C: 155.2 (C-1), 114.5 (C-2)", [155.2, 114.5]),
]

# SMILES canonicalization tests: (input, expected_canonical)
SMILES_TESTS = [
    ("c1ccccc1", "c1ccccc1"),
    ("C1=CC=CC=C1", "c1ccccc1"),  # Kekulé to aromatic
    ("Cc1ccccc1", "Cc1ccccc1"),
    ("c1ccc(C)cc1", "Cc1ccccc1"),  # Reorder
    ("OCC", "CCO"),  # Reorder
]


# ============================================================================
# Unit Tests - H1 Parsing
# ============================================================================

class TestH1Parsing:
    """Tests for 1H NMR spectrum parsing."""

    @pytest.mark.parametrize("input_str,expected,forbidden", H1_PARSING_CASES)
    def test_h1_parsing(self, input_str: str, expected: List[float], forbidden: List[float]):
        """Test that H1 parser extracts correct ppm values."""
        result = _parse_h1_ppm(input_str)

        # All expected values should be present (within tolerance)
        for exp in expected:
            found = any(abs(r - exp) < 0.05 for r in result)
            assert found, f"Expected {exp} not found in {result}"

        # Forbidden values should NOT be present
        for bad in forbidden:
            found = any(abs(r - bad) < 0.05 for r in result)
            assert not found, f"Forbidden value {bad} found in {result}"

        # All values should be in valid H1 range
        for val in result:
            assert 0.0 <= val <= 12.5, f"Value {val} outside H1 range"

    def test_h1_with_integrals(self):
        """Test that integral extraction works."""
        ppm, intensity = _parse_h1_with_integrals("7.26 (d, 2H), 3.78 (s, 3H)")

        assert len(ppm) == len(intensity)
        assert len(ppm) >= 2

        # Check that intensities reflect proton counts
        # (normalized to max=1, so ratio should be preserved)
        if len(ppm) == 2:
            ratio = min(intensity) / max(intensity)
            assert 0.5 <= ratio <= 0.8  # 2H/3H ≈ 0.67

    def test_empty_input(self):
        """Test handling of empty/None input."""
        assert _parse_h1_ppm("") == []
        assert _parse_h1_ppm(None) == []

    def test_invalid_input(self):
        """Test handling of non-NMR text."""
        assert _parse_h1_ppm("not a spectrum") == []
        assert _parse_h1_ppm("1234567890") == []  # Out of range


# ============================================================================
# Unit Tests - C13 Parsing
# ============================================================================

class TestC13Parsing:
    """Tests for 13C NMR spectrum parsing."""

    @pytest.mark.parametrize("input_str,expected", C13_PARSING_CASES)
    def test_c13_parsing(self, input_str: str, expected: List[float]):
        """Test that C13 parser extracts correct ppm values."""
        result = _parse_c13_ppm(input_str)

        for exp in expected:
            found = any(abs(r - exp) < 0.5 for r in result)
            assert found, f"Expected {exp} not found in {result}"

        # All values should be in valid C13 range
        for val in result:
            assert 0.0 <= val <= 220.0, f"Value {val} outside C13 range"


# ============================================================================
# Unit Tests - SMILES Handling
# ============================================================================

class TestSMILESHandling:
    """Tests for SMILES canonicalization."""

    @pytest.mark.parametrize("input_smiles,expected", SMILES_TESTS)
    def test_canonicalization(self, input_smiles: str, expected: str):
        """Test SMILES canonicalization."""
        result = _smiles_key(input_smiles)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        assert _smiles_key("not_a_smiles") is None
        assert _smiles_key("") is None
        assert _smiles_key("C1CC") is None  # Unclosed ring


# ============================================================================
# Integration Tests - Database Lookup
# ============================================================================

class TestDatabaseLookup:
    """Tests for NMRBank database operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Warm cache before tests if not in test mode."""
        if os.getenv("NMRBANK_SKIP_LOAD_FOR_TESTS") != "1":
            try:
                warm_cache()
            except Exception:
                pytest.skip("NMRBank CSV not available")

    @pytest.mark.parametrize("name,smiles", COMMON_COMPOUNDS)
    def test_common_compound_lookup(self, name: str, smiles: str):
        """Test that common compounds can be found."""
        if os.getenv("NMRBANK_SKIP_LOAD_FOR_TESTS") == "1":
            pytest.skip("NMRBank not loaded")

        ref = get_reference_by_smiles(smiles)

        # Should find the compound
        assert ref is not None, f"Could not find {name} ({smiles})"

        # Should have spectrum data
        assert "ppm" in ref or "ppm_h1" in ref or "ppm_c13" in ref
        assert "smiles" in ref

    def test_lookup_returns_spectrum_data(self):
        """Test that lookup returns proper spectrum format."""
        if os.getenv("NMRBANK_SKIP_LOAD_FOR_TESTS") == "1":
            pytest.skip("NMRBank not loaded")

        ref = get_reference_by_smiles("c1ccccc1")  # benzene
        if ref is None:
            pytest.skip("Benzene not in database")

        # Check structure
        assert isinstance(ref, dict)
        assert "name" in ref
        assert "smiles" in ref

        # Should have ppm/intensity pairs
        if "ppm_h1" in ref:
            assert "intensity_h1" in ref
            assert len(ref["ppm_h1"]) == len(ref["intensity_h1"])

    def test_nonexistent_compound(self):
        """Test behavior for compounds not in database."""
        # Very unlikely to be in database
        ref = get_reference_by_smiles("C1C2C3C4C5C6C7C8C1C2C3C4C5C6C7C8")
        # Should return None, not raise
        assert ref is None

    def test_lookup_performance(self):
        """Test that lookups are fast after cache warm."""
        if os.getenv("NMRBANK_SKIP_LOAD_FOR_TESTS") == "1":
            pytest.skip("NMRBank not loaded")

        # Time 100 lookups
        start = time.time()
        for _ in range(100):
            get_reference_by_smiles("c1ccccc1")
        elapsed = time.time() - start

        # Should be very fast (< 100ms for 100 lookups)
        assert elapsed < 0.1, f"Lookups too slow: {elapsed:.3f}s for 100"


# ============================================================================
# Coverage Analysis
# ============================================================================

def analyze_coverage():
    """Analyze database coverage for common compounds."""
    print("=" * 60)
    print("NMRBank Coverage Analysis")
    print("=" * 60)

    if os.getenv("NMRBANK_SKIP_LOAD_FOR_TESTS") == "1":
        print("Skipping - NMRBANK_SKIP_LOAD_FOR_TESTS=1")
        return

    try:
        n_compounds = warm_cache()
        print(f"\nTotal compounds in database: {n_compounds}")
    except Exception as e:
        print(f"Could not load database: {e}")
        return

    print("\nCommon compound coverage:")
    found = 0
    for name, smiles in COMMON_COMPOUNDS:
        ref = get_reference_by_smiles(smiles)
        status = "FOUND" if ref else "MISSING"
        n_peaks = len(ref.get("ppm", [])) if ref else 0
        print(f"  {name:15} ({smiles:15}): {status} ({n_peaks} peaks)")
        if ref:
            found += 1

    print(f"\nCoverage: {found}/{len(COMMON_COMPOUNDS)} = {100*found/len(COMMON_COMPOUNDS):.0f}%")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run coverage analysis when executed directly
    analyze_coverage()

    print("\n" + "=" * 60)
    print("Running pytest...")
    print("=" * 60)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
