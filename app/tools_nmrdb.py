"""
NMRDB.org API integration for fetching and parsing NMR spectra data.

This module provides functionality to:
1. Fetch NMR spectra predictions from NMRDB.org
2. Parse HTML responses to extract peak data
3. Format data for Magnetstein compatibility
"""

import requests
import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import json
import time


def fetch_nmr_spectrum(smiles: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch NMR spectrum prediction from NMRDB.org for a given SMILES string.

    Args:
        smiles: SMILES string of the compound
        timeout: Request timeout in seconds

    Returns:
        HTML response content or None if failed
    """
    url = f"https://www.nmrdb.org/service.php?name=nmr-1h-prediction&smiles={smiles}"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching NMR data for {smiles}: {e}")
        return None


def parse_nmr_html(html_content: str) -> Dict[str, List[float]]:
    """
    Parse HTML content from NMRDB.org to extract NMR peak data.
    Uses multiple parsing strategies to handle different response formats.

    Args:
        html_content: HTML content from NMRDB.org

    Returns:
        Dictionary with 'ppm' and 'intensity' lists
    """
    soup = BeautifulSoup(html_content, "html.parser")

    ppm_values = []
    intensity_values = []

    # Method 1: Look for JavaScript data or JSON embedded in the HTML
    scripts = soup.find_all("script")

    for script in scripts:
        if script.string:
            script_content = script.string

            # Try to find JSON data containing NMR peaks
            json_patterns = [
                r"var\s+\w*[Nn]mr\w*\s*=\s*(\[.*?\]);",
                r"var\s+\w*[Ss]pectrum\w*\s*=\s*(\[.*?\]);",
                r"var\s+\w*[Pp]eaks\w*\s*=\s*(\[.*?\]);",
                r"var\s+\w*[Dd]ata\w*\s*=\s*(\[.*?\]);",
                r"nmrData\s*=\s*(\[.*?\]);",
                r"spectrumData\s*=\s*(\[.*?\]);",
            ]

            for pattern in json_patterns:
                json_match = re.search(pattern, script_content, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        if isinstance(data, list) and len(data) > 0:
                            # Extract ppm and intensity values
                            for item in data:
                                if isinstance(item, dict):
                                    if "ppm" in item and "intensity" in item:
                                        ppm_values.append(float(item["ppm"]))
                                        intensity_values.append(
                                            float(item["intensity"])
                                        )
                                    elif "shift" in item and "intensity" in item:
                                        ppm_values.append(float(item["shift"]))
                                        intensity_values.append(
                                            float(item["intensity"])
                                        )
                                    elif "x" in item and "y" in item:
                                        ppm_values.append(float(item["x"]))
                                        intensity_values.append(float(item["y"]))
                                elif isinstance(item, list) and len(item) >= 2:
                                    # Assume first element is ppm, second is intensity
                                    ppm_values.append(float(item[0]))
                                    intensity_values.append(float(item[1]))
                            break  # Found data, stop looking
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        print(f"Error parsing JSON data: {e}")
                        continue

    # Method 2: Look for tables in the main HTML
    if not ppm_values:
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:  # Need at least header + data row
                continue

            # Check if this looks like NMR data table
            header_row = rows[0]
            header_text = header_row.get_text().lower()
            if any(
                keyword in header_text
                for keyword in ["ppm", "shift", "chemical", "intensity", "signal"]
            ):
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        try:
                            ppm_val = float(cells[0].get_text().strip())
                            intensity_val = float(cells[1].get_text().strip())
                            if 0 <= ppm_val <= 20:  # Reasonable NMR range
                                ppm_values.append(ppm_val)
                                intensity_values.append(intensity_val)
                        except ValueError:
                            continue

    # Method 3: Look for text patterns in the content (similar to tools_nmrbank.py)
    if not ppm_values:
        text_content = soup.get_text()

        # Look for patterns like "7.2 (s, 1H)" or "7.2 ppm" or "7.2, 6.8, 2.1"
        # This uses the same pattern as _parse_nmr_shifts in tools_nmrbank.py
        numbers = re.findall(r"-?\d+\.?\d*", text_content)
        try:
            for num in numbers:
                ppm_val = float(num)
                if 0 <= ppm_val <= 20:  # Reasonable NMR range
                    ppm_values.append(ppm_val)
                    intensity_values.append(1.0)  # Default intensity
        except (ValueError, TypeError):
            pass

    # Method 4: Look for specific NMRDB.org patterns
    if not ppm_values:
        text_content = soup.get_text()

        # Look for patterns like "Chemical shifts: 7.2, 6.8, 2.1"
        shift_pattern = r"(?:chemical\s+shifts?|shifts?|peaks?)[:\s]*([0-9.,\s]+)"
        shift_matches = re.findall(shift_pattern, text_content, re.IGNORECASE)

        for match in shift_matches:
            # Extract numbers from the match
            numbers = re.findall(r"\d+\.?\d*", match)
            for num in numbers:
                try:
                    ppm_val = float(num)
                    if 0 <= ppm_val <= 20:  # Reasonable NMR range
                        ppm_values.append(ppm_val)
                        intensity_values.append(1.0)  # Default intensity
                except ValueError:
                    continue

    # Remove duplicates and sort by ppm
    if ppm_values:
        # Create list of tuples, remove duplicates, sort by ppm
        unique_data = list(set(zip(ppm_values, intensity_values)))
        unique_data.sort(key=lambda x: x[0])

        ppm_values = [item[0] for item in unique_data]
        intensity_values = [item[1] for item in unique_data]

    return {"ppm": ppm_values, "intensity": intensity_values}


def get_nmr_spectrum_for_smiles(
    smiles: str, delay: float = 1.0
) -> Dict[str, List[float]]:
    """
    Complete workflow to fetch and parse NMR spectrum for a SMILES string.

    Args:
        smiles: SMILES string of the compound
        delay: Delay between requests to be respectful to the API

    Returns:
        Dictionary with 'ppm' and 'intensity' lists compatible with Magnetstein
    """
    # Add delay to be respectful to the API
    time.sleep(delay)

    html_content = fetch_nmr_spectrum(smiles)
    if not html_content:
        return {"ppm": [], "intensity": []}

    spectrum_data = parse_nmr_html(html_content)

    # Filter out invalid data (Magnetstein requires non-negative ppm values)
    valid_ppm = []
    valid_intensity = []

    for ppm, intensity in zip(spectrum_data["ppm"], spectrum_data["intensity"]):
        if 0 <= ppm <= 20 and intensity >= 0:  # Reasonable NMR range
            valid_ppm.append(ppm)
            valid_intensity.append(intensity)

    return {"ppm": valid_ppm, "intensity": valid_intensity}


def get_reference_by_smiles_nmrdb(smiles: str, delay: float = 1.0) -> Optional[dict]:
    """
    Get NMR reference data from NMRDB.org for a SMILES string.
    Compatible with the existing app flow - returns same format as tools_nmrbank.py

    Args:
        smiles: SMILES string of the compound
        delay: Delay between requests to be respectful to the API

    Returns:
        Dictionary with 'name', 'smiles', 'ppm', 'intensity' keys, or None if not found
    """
    try:
        from rdkit import Chem

        # Normalize SMILES (same as tools_nmrbank.py)
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Fetch NMR data
        spectrum_data = get_nmr_spectrum_for_smiles(canonical_smiles, delay)

        if not spectrum_data["ppm"]:
            return None

        # Return in the same format as tools_nmrbank.py
        return {
            "name": f"NMRDB_{canonical_smiles[:10]}...",  # Truncated name
            "smiles": canonical_smiles,
            "ppm": spectrum_data["ppm"],
            "intensity": spectrum_data["intensity"],
        }

    except Exception as e:
        print(f"Error fetching NMRDB data for {smiles}: {e}")
        return None


def format_for_magnetstein(
    spectrum_data: Dict[str, List[float]], name: str = "Unknown"
) -> Dict[str, List[float]]:
    """
    Format NMR spectrum data for Magnetstein compatibility.

    Args:
        spectrum_data: Dictionary with 'ppm' and 'intensity' lists
        name: Name of the compound

    Returns:
        Dictionary formatted for Magnetstein input
    """
    return {
        "name": name,
        "ppm": spectrum_data["ppm"],
        "intensity": spectrum_data["intensity"],
    }


def batch_fetch_nmr_spectra(
    smiles_list: List[str], delay: float = 2.0
) -> List[Dict[str, List[float]]]:
    """
    Fetch NMR spectra for multiple SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        delay: Delay between requests

    Returns:
        List of formatted spectrum data dictionaries
    """
    results = []

    for i, smiles in enumerate(smiles_list):
        print(f"Fetching NMR data for compound {i + 1}/{len(smiles_list)}: {smiles}")

        spectrum_data = get_nmr_spectrum_for_smiles(smiles, delay)
        formatted_data = format_for_magnetstein(spectrum_data, f"compound_{i + 1}")

        results.append(formatted_data)

        # Add extra delay every 10 requests
        if (i + 1) % 10 == 0:
            time.sleep(delay * 2)

    return results


def create_magnetstein_library_from_smiles(
    smiles_list: List[str], delay: float = 2.0
) -> List[Dict[str, List[float]]]:
    """
    Create a Magnetstein-compatible library from a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        delay: Delay between requests

    Returns:
        List of dictionaries formatted for Magnetstein library input
    """
    return batch_fetch_nmr_spectra(smiles_list, delay)


# Test function
def test_nmrdb_integration():
    """Test the NMRDB integration with a simple compound."""
    test_smiles = "c1ccccc1CC"  # Ethylbenzene

    print(f"Testing NMRDB integration with SMILES: {test_smiles}")

    spectrum_data = get_nmr_spectrum_for_smiles(test_smiles)

    print(f"Retrieved {len(spectrum_data['ppm'])} peaks")
    print("Peak data:")
    for ppm, intensity in zip(spectrum_data["ppm"], spectrum_data["intensity"]):
        print(f"  {ppm:.2f} ppm, intensity: {intensity:.2f}")

    return spectrum_data


def test_magnetstein_integration():
    """Test integration with Magnetstein tools."""
    try:
        from .tools_magnetstein import quantify_single

        print("Testing Magnetstein integration...")

        # Get test spectrum
        test_smiles = "c1ccccc1CC"  # Ethylbenzene
        spectrum_data = get_nmr_spectrum_for_smiles(test_smiles)

        if not spectrum_data["ppm"]:
            print("No spectrum data retrieved, cannot test Magnetstein integration")
            return False

        # Create a simple library with the same compound
        library = [format_for_magnetstein(spectrum_data, "ethylbenzene")]

        # Create a mock mixture spectrum (same as reference for testing)
        mixture_ppm = spectrum_data["ppm"]
        mixture_intensity = spectrum_data["intensity"]

        # Test Magnetstein quantification
        result = quantify_single(mixture_ppm, mixture_intensity, library)

        print("Magnetstein integration test successful!")
        print(f"Concentrations: {result['concentrations']}")
        return True

    except ImportError as e:
        print(f"Magnetstein not available: {e}")
        return False
    except Exception as e:
        print(f"Magnetstein integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_nmrdb_integration()
    test_magnetstein_integration()
