# Description: Test cases for shadow_check_location.py
#              Tests the shadow calculation for known locations and times.
#
# Usage: pytest test_shadow_check_location.py -v
#        or: python test_shadow_check_location.py
#
# Copyright (c) 2025
# MIT License

import pytest
import subprocess
import sys
import os

# Path to Python executable in virtual environment
PYTHON_EXE = os.path.join(os.path.dirname(__file__), "..", "..", "horayzon_env", "python.exe")
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "shadow_check_location.py")


def run_shadow_check(date, time, east, north):
    """Run shadow_check_location.py and return the output."""
    cmd = [
        PYTHON_EXE,
        SCRIPT_PATH,
        "-d", date,
        "-t", time,
        "-e", str(east),
        "-n", str(north)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    return result.stdout, result.stderr, result.returncode


class TestShadowCheckLocation:
    """Test cases for shadow_check_location.py"""

    def test_nodata_detection(self):
        """
        Test Case 0: NoData handling
        Location: E=2693542.0, N=1233645.0 (Zürichsee area - water body)
        Date/Time: 24.10.2025 10:21:00 UTC
        Expected: NODATA (elevation = 0.0)
        """
        stdout, stderr, returncode = run_shadow_check(
            date="24.10.2025",
            time="10:21:00",
            east=2693542.0,
            north=1233645.0
        )

        assert returncode == 0, f"Script failed with error: {stderr}"
        assert "NODATA" in stdout, f"Expected NODATA but got: {stdout}"
        assert "Standort LV95:       E=2693542.00 / N=1233645.00" in stdout
        assert "Hoehe:               0.0 m" in stdout

    def test_bern_default_case(self):
        """
        Test Case 1: Bern (Default)
        Location: E=2600000.0, N=1200000.0 (Bern area)
        Date/Time: 13.12.2025 12:22:00 UTC
        Expected: BELEUCHTET (illuminated)
        """
        stdout, stderr, returncode = run_shadow_check(
            date="13.12.2025",
            time="12:22:00",
            east=2600000.0,
            north=1200000.0
        )

        assert returncode == 0, f"Script failed with error: {stderr}"
        assert "BELEUCHTET" in stdout, f"Expected BELEUCHTET but got: {stdout}"
        assert "Standort LV95:       E=2600000.00 / N=1200000.00" in stdout
        assert "Hoehe:               569.1 m" in stdout

        # Check sun position (approximate values)
        assert "Sonnen-Elevation:    18.67 Grad" in stdout
        assert "Sonnen-Azimut:       193.94 Grad" in stdout

    def test_zurich_area_october(self):
        """
        Test Case 2: Zurich area
        Location: E=2643244.0, N=1265255.0
        Date/Time: 24.10.2025 10:21:00 UTC
        Expected: BELEUCHTET (illuminated)
        """
        stdout, stderr, returncode = run_shadow_check(
            date="24.10.2025",
            time="10:21:00",
            east=2643244.0,
            north=1265255.0
        )

        assert returncode == 0, f"Script failed with error: {stderr}"
        assert "BELEUCHTET" in stdout, f"Expected BELEUCHTET but got: {stdout}"
        assert "Standort LV95:       E=2643244.00 / N=1265255.00" in stdout
        assert "Hoehe:               419.0 m" in stdout

        # Check sun position (approximate values)
        assert "Sonnen-Elevation:    29.46 Grad" in stdout
        assert "Sonnen-Azimut:       165.60 Grad" in stdout

    def test_perpendicular_sun_steep_slope(self):
        """
        Test Case 3: Steep slope with near-perpendicular sun incidence
        Location: E=2578545.0, N=1161265.0 (Wallis, steep south-facing slope)
        Date/Time: 13.12.2025 12:22:00 UTC
        Expected: BELEUCHTET (illuminated)

        This location has a ~68 degree slope facing SSW (~200 degrees),
        resulting in near-perpendicular solar incidence angle (~6.8 degrees)
        when the sun is at elevation ~19 degrees and azimuth ~194 degrees.
        """
        stdout, stderr, returncode = run_shadow_check(
            date="13.12.2025",
            time="12:22:00",
            east=2578545.0,
            north=1161265.0
        )

        assert returncode == 0, f"Script failed with error: {stderr}"
        assert "BELEUCHTET" in stdout, f"Expected BELEUCHTET but got: {stdout}"
        assert "Standort LV95:       E=2578545.00 / N=1161265.00" in stdout
        assert "Hoehe:               1194.5 m" in stdout

        # Check slope and aspect (steep south-facing slope)
        assert "Hangneigung:         67.9 Grad" in stdout
        assert "Exposition:          199.8 Grad" in stdout

        # Check sun position
        assert "Sonnen-Elevation:    19.06 Grad" in stdout
        assert "Sonnen-Azimut:       193.70 Grad" in stdout


def run_tests_standalone():
    """Run tests without pytest (standalone mode)."""
    print("=" * 70)
    print("SHADOW CHECK LOCATION - TEST SUITE")
    print("=" * 70)

    tests = [
        ("Bern Default Case", "13.12.2025", "12:22:00", 2600000.0, 1200000.0, "BELEUCHTET"),
        ("Zurich Area October", "24.10.2025", "10:21:00", 2643244.0, 1265255.0, "BELEUCHTET"),
        ("NoData Zuerichsee", "24.10.2025", "10:21:00", 2693542.0, 1233645.0, "NODATA"),
        ("Perpendicular Sun Wallis", "13.12.2025", "12:22:00", 2578545.0, 1161265.0, "BELEUCHTET"),
    ]

    passed = 0
    failed = 0

    for name, date, time, east, north, expected_status in tests:
        print(f"\nRunning: {name}")
        print(f"  Parameters: {date} {time} E={east} N={north}")
        print("-" * 70)

        try:
            stdout, stderr, returncode = run_shadow_check(date, time, east, north)

            if returncode != 0:
                print(f"  FAILED: Script returned error code {returncode}")
                print(f"  Error: {stderr}")
                failed += 1
                continue

            if expected_status in stdout:
                print(f"  PASSED: Found expected status '{expected_status}'")
                passed += 1
            else:
                print(f"  FAILED: Expected '{expected_status}' not found in output")
                failed += 1

        except Exception as e:
            print(f"  FAILED: Exception occurred: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest
        # Run with pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        # Run standalone
        success = run_tests_standalone()
        sys.exit(0 if success else 1)
