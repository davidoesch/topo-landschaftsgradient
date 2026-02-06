# Description: Verify shadow calculation using pvlib as alternative to horayzon
#              Compares sun position and calculates horizon angle from DEM
#
# Usage: python shadow_check_pvlib.py
#
# Required: pip install pvlib pandas rasterio pyproj numpy
#
# Copyright (c) 2025
# MIT License

import os
import sys
import numpy as np
import rasterio
from datetime import datetime, timezone
from pyproj import CRS, Transformer

# Try to import pvlib, fall back to pysolar
try:
    import pvlib
    from pvlib import solarposition
    import pandas as pd
    USE_PVLIB = True
except ImportError:
    USE_PVLIB = False
    try:
        import pysolar.solar as solar
        USE_PYSOLAR = True
    except ImportError:
        USE_PYSOLAR = False
        print("FEHLER: Bitte installiere pvlib oder pysolar:")
        print("  pip install pvlib pandas")
        print("  oder: pip install pysolar")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DOM = r"D:\temp\DOM\Thinout_highest_object_10m_LV95_LHN95.tif"

# Test cases: (name, date, time_utc, east_lv95, north_lv95, expected_result)
TEST_CASES = [
    ("Bern Default", "13.12.2025", "12:22:00", 2600000.0, 1200000.0, "BELEUCHTET"),
    ("Zurich Area October", "24.10.2025", "10:21:00", 2643244.0, 1265255.0, "BELEUCHTET"),
    ("NoData Zuerichsee", "24.10.2025", "10:21:00", 2693542.0, 1233645.0, "NODATA"),
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def lv95_to_wgs84(e_lv95, n_lv95):
    """Convert LV95 (EPSG:2056) to WGS84 (EPSG:4326) coordinates."""
    crs_lv95 = CRS.from_epsg(2056)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_lv95, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(e_lv95, n_lv95)
    return lon, lat


def parse_datetime(date_str, time_str):
    """Parse date and time strings to datetime object (UTC)."""
    dt_str = f"{date_str} {time_str}"
    for fmt in ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"]:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date/time: {dt_str}")


def get_sun_position_pvlib(dt_utc, lat, lon):
    """Calculate sun position using pvlib."""
    times = pd.DatetimeIndex([dt_utc])
    pos = solarposition.get_solarposition(times, lat, lon)
    elevation = pos['apparent_elevation'].values[0]
    azimuth = pos['azimuth'].values[0]
    return elevation, azimuth


def get_sun_position_pysolar(dt_utc, lat, lon):
    """Calculate sun position using pysolar."""
    elevation = solar.get_altitude(lat, lon, dt_utc)
    azimuth = solar.get_azimuth(lat, lon, dt_utc)
    return elevation, azimuth


def get_sun_position(dt_utc, lat, lon):
    """Get sun position using available library."""
    if USE_PVLIB:
        return get_sun_position_pvlib(dt_utc, lat, lon)
    else:
        return get_sun_position_pysolar(dt_utc, lat, lon)


def calculate_horizon_angle(dem_data, transform, nodata, x_loc, y_loc, azimuth_deg,
                            max_distance_m=20000, step_m=10):
    """
    Calculate horizon angle from a point in direction of sun azimuth.

    Returns the maximum elevation angle to the horizon in the given direction.
    """
    # Get location elevation
    col_loc = int((x_loc - transform[2]) / transform[0])
    row_loc = int((y_loc - transform[5]) / transform[4])

    nrows, ncols = dem_data.shape

    if col_loc < 0 or col_loc >= ncols or row_loc < 0 or row_loc >= nrows:
        return None, None

    elev_loc = dem_data[row_loc, col_loc]

    if nodata is not None and elev_loc == nodata:
        return None, None
    if elev_loc == 0.0 or np.isnan(elev_loc):
        return None, None

    # Convert azimuth to radians (from North, clockwise)
    az_rad = np.radians(azimuth_deg)

    # Direction vector (in LV95: x=East, y=North)
    dx = np.sin(az_rad)  # East component
    dy = np.cos(az_rad)  # North component

    max_horizon_angle = -90.0
    max_horizon_distance = 0.0

    # Sample along the ray
    distances = np.arange(step_m, max_distance_m, step_m)

    for dist in distances:
        # Calculate sample point
        x_sample = x_loc + dx * dist
        y_sample = y_loc + dy * dist

        # Convert to pixel coordinates
        col = int((x_sample - transform[2]) / transform[0])
        row = int((y_sample - transform[5]) / transform[4])

        # Check bounds
        if col < 0 or col >= ncols or row < 0 or row >= nrows:
            continue

        elev_sample = dem_data[row, col]

        # Skip nodata
        if nodata is not None and elev_sample == nodata:
            continue
        if np.isnan(elev_sample):
            continue

        # Calculate elevation angle to this point
        height_diff = elev_sample - elev_loc
        angle = np.degrees(np.arctan2(height_diff, dist))

        if angle > max_horizon_angle:
            max_horizon_angle = angle
            max_horizon_distance = dist

    return max_horizon_angle, max_horizon_distance


def check_self_shading(dem_data, transform, x_loc, y_loc, sun_azimuth, sun_elevation):
    """
    Check if the surface is self-shaded (facing away from sun).

    Returns True if self-shaded, False otherwise.
    """
    # Get pixel coordinates
    col = int((x_loc - transform[2]) / transform[0])
    row = int((y_loc - transform[5]) / transform[4])

    nrows, ncols = dem_data.shape

    # Need at least one pixel border for gradient
    if col < 1 or col >= ncols - 1 or row < 1 or row >= nrows - 1:
        return False

    # Calculate slope using central differences
    dx = transform[0]  # pixel size x
    dy = abs(transform[4])  # pixel size y (negative in transform)

    # Gradient in x direction (dz/dx)
    dzdx = (dem_data[row, col + 1] - dem_data[row, col - 1]) / (2 * dx)

    # Gradient in y direction (dz/dy) - note: row increases southward
    dzdy = (dem_data[row - 1, col] - dem_data[row + 1, col]) / (2 * dy)

    # Surface normal vector (pointing up)
    # n = (-dzdx, -dzdy, 1) normalized
    normal = np.array([-dzdx, -dzdy, 1.0])
    normal = normal / np.linalg.norm(normal)

    # Sun direction vector
    sun_elev_rad = np.radians(sun_elevation)
    sun_az_rad = np.radians(sun_azimuth)

    sun_dir = np.array([
        np.cos(sun_elev_rad) * np.sin(sun_az_rad),  # East
        np.cos(sun_elev_rad) * np.cos(sun_az_rad),  # North
        np.sin(sun_elev_rad)                         # Up
    ])

    # Dot product: positive = sun illuminates surface, negative = self-shaded
    cos_incidence = np.dot(normal, sun_dir)

    return cos_incidence < 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SCHATTEN-CHECK MIT PVLIB/PYSOLAR")
    print("(Alternative Verifikation zu horayzon)")
    print("=" * 70)

    lib_name = "pvlib" if USE_PVLIB else "pysolar"
    print(f"Verwendete Bibliothek: {lib_name}")
    print(f"DOM-Datei: {DEFAULT_DOM}")
    print("=" * 70)

    if not os.path.isfile(DEFAULT_DOM):
        print(f"\nFEHLER: DOM nicht gefunden: {DEFAULT_DOM}")
        sys.exit(1)

    # Load DEM once
    print("\nLade DOM-Daten...")
    with rasterio.open(DEFAULT_DOM) as src:
        dem_data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        print(f"  Groesse: {dem_data.shape}")
        print(f"  NoData-Wert: {nodata}")

    print("\n" + "-" * 70)
    print("TEST CASES")
    print("-" * 70)

    results = []

    for name, date_str, time_str, east, north, expected in TEST_CASES:
        print(f"\n[{name}]")
        print(f"  Standort LV95: E={east:.2f} / N={north:.2f}")
        print(f"  Datum/Zeit (UTC): {date_str} {time_str}")

        # Parse datetime
        dt_utc = parse_datetime(date_str, time_str)

        # Convert coordinates
        lon, lat = lv95_to_wgs84(east, north)
        print(f"  WGS84: Lat={lat:.5f} / Lon={lon:.5f}")

        # Get elevation at location
        col = int((east - transform[2]) / transform[0])
        row = int((north - transform[5]) / transform[4])

        if 0 <= col < dem_data.shape[1] and 0 <= row < dem_data.shape[0]:
            elev_loc = dem_data[row, col]
        else:
            elev_loc = None

        # Check NoData
        if elev_loc is None or (nodata is not None and elev_loc == nodata) or elev_loc == 0.0:
            print(f"  Hoehe: NoData oder 0")
            pvlib_result = "NODATA"
            sun_elev = None
            sun_az = None
            horizon_angle = None
        else:
            print(f"  Hoehe: {elev_loc:.1f} m")

            # Calculate sun position
            sun_elev, sun_az = get_sun_position(dt_utc, lat, lon)
            print(f"  Sonnen-Elevation ({lib_name}): {sun_elev:.2f} Grad")
            print(f"  Sonnen-Azimut ({lib_name}): {sun_az:.2f} Grad")

            if sun_elev < 0:
                pvlib_result = "NACHT"
                horizon_angle = None
                print(f"  -> Sonne unter Horizont (Nacht)")
            else:
                # Calculate horizon angle in sun direction
                horizon_angle, horizon_dist = calculate_horizon_angle(
                    dem_data, transform, nodata, east, north, sun_az
                )

                if horizon_angle is not None:
                    print(f"  Horizont-Winkel (Richtung Sonne): {horizon_angle:.2f} Grad")
                    print(f"  Horizont-Distanz: {horizon_dist:.0f} m")

                    # Check self-shading first
                    is_self_shaded = check_self_shading(
                        dem_data, transform, east, north, sun_az, sun_elev
                    )

                    if is_self_shaded:
                        pvlib_result = "SELBSTBESCHATTET"
                        print(f"  -> Hang zeigt von Sonne weg")
                    elif sun_elev > horizon_angle:
                        pvlib_result = "BELEUCHTET"
                        print(f"  -> Sonne ({sun_elev:.1f}) > Horizont ({horizon_angle:.1f})")
                    else:
                        pvlib_result = "SCHATTEN"
                        print(f"  -> Sonne ({sun_elev:.1f}) < Horizont ({horizon_angle:.1f})")
                else:
                    pvlib_result = "FEHLER"
                    print(f"  -> Konnte Horizont nicht berechnen")

        # Compare with expected
        # Map results for comparison
        result_map = {
            "BELEUCHTET": "BELEUCHTET",
            "SCHATTEN": "SCHATTEN",
            "SELBSTBESCHATTET": "SCHATTEN",
            "NODATA": "NODATA",
            "NACHT": "NACHT",
            "FEHLER": "FEHLER"
        }

        pvlib_mapped = result_map.get(pvlib_result, pvlib_result)
        match = pvlib_mapped == expected or pvlib_result == expected

        print(f"  {lib_name} Ergebnis: {pvlib_result}")
        print(f"  Erwartet (horayzon): {expected}")
        print(f"  Vergleich: {'OK' if match else 'UNTERSCHIED'}")

        results.append({
            "name": name,
            "pvlib_result": pvlib_result,
            "expected": expected,
            "match": match,
            "sun_elevation": sun_elev,
            "sun_azimuth": sun_az,
            "horizon_angle": horizon_angle
        })

    # Summary
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    matches = sum(1 for r in results if r["match"])
    print(f"\n{matches}/{len(results)} uebereinstimmend\n")

    print(f"{'Test':<25} {lib_name:<18} {'horayzon':<15} {'Status':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['pvlib_result']:<18} {r['expected']:<15} {'OK' if r['match'] else 'DIFF':<10}")

    print("\n" + "=" * 70)
    print("SONNEN-POSITIONEN (zur manuellen Verifikation)")
    print("=" * 70)
    print(f"\n{'Test':<25} {'Elevation':<12} {'Azimut':<12} {'Horizont':<12}")
    print("-" * 70)
    for r in results:
        elev = f"{r['sun_elevation']:.2f}" if r['sun_elevation'] is not None else "N/A"
        az = f"{r['sun_azimuth']:.2f}" if r['sun_azimuth'] is not None else "N/A"
        hor = f"{r['horizon_angle']:.2f}" if r['horizon_angle'] is not None else "N/A"
        print(f"{r['name']:<25} {elev:<12} {az:<12} {hor:<12}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
