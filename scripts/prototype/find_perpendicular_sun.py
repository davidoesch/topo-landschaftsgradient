# Description: Find locations in the DOM where the sun is perpendicular
#              to the surface (incidence angle = 0 degrees)
#
# Usage: python find_perpendicular_sun.py -d 13.12.2025 -t 12:22:00
#
# Copyright (c) 2025
# MIT License

import os
import sys
import argparse
import numpy as np
import rasterio
import pandas as pd
from datetime import datetime, timezone
from pyproj import CRS, Transformer
from pvlib import solarposition


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DOM = r"C:\temp\DOM\Thinout_highest_object_10m_LV95_LHN95.tif"


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


def wgs84_to_lv95(lon, lat):
    """Convert WGS84 (EPSG:4326) to LV95 (EPSG:2056) coordinates."""
    crs_lv95 = CRS.from_epsg(2056)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, crs_lv95, always_xy=True)
    e, n = transformer.transform(lon, lat)
    return e, n


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


def get_sun_pos(dt_utc, lat, lon):
    """Calculate sun position using pvlib."""
    times = pd.DatetimeIndex([dt_utc])
    pos = solarposition.get_solarposition(times, lat, lon)
    elevation = pos["apparent_elevation"].values[0]
    azimuth = pos["azimuth"].values[0]
    return elevation, azimuth


def calculate_slope_aspect(dem_data, transform, nodata):
    """
    Calculate slope and aspect for entire DEM.

    Returns:
        slope: Slope angle in degrees (0 = flat, 90 = vertical)
        aspect: Aspect in degrees from North,
        clockwise (0=N, 90=E, 180=S, 270=W)
    """
    nrows, ncols = dem_data.shape
    dx = transform[0]  # pixel size x
    dy = abs(transform[4])  # pixel size y

    # Create output arrays
    slope = np.full((nrows, ncols), np.nan, dtype=np.float32)
    aspect = np.full((nrows, ncols), np.nan, dtype=np.float32)

    # Calculate gradients using numpy (faster than loop)
    # Pad array for edge handling
    dem_padded = np.pad(dem_data, 1, mode="edge")

    # Handle nodata
    if nodata is not None:
        dem_padded = np.where(dem_padded == nodata, np.nan, dem_padded)

    # Central differences
    dzdx = (dem_padded[1:-1, 2:] - dem_padded[1:-1, :-2]) / (2 * dx)
    dzdy = (dem_padded[:-2, 1:-1] - dem_padded[2:, 1:-1]) / (2 * dy)

    # Slope = arctan(sqrt(dzdx^2 + dzdy^2))
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Aspect = direction the slope faces (downslope direction)
    # arctan2(dzdx, dzdy) gives the direction of steepest ascent
    # We need the direction the slope faces (opposite = downslope)
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    return slope, aspect


def calc_incidence_angle(slope, aspect, sun_elevation, sun_azimuth):
    """
    Calculate the solar incidence angle for each pixel.

    Incidence angle = 0 means sun is perpendicular to the surface.

    cos(incidence) = cos(slope) * sin(sun_elev) +
                     sin(slope) * cos(sun_elev) * cos(sun_az - aspect)
    """
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    sun_elev_rad = np.radians(sun_elevation)
    sun_az_rad = np.radians(sun_azimuth)

    cos_incidence = np.cos(slope_rad) * np.sin(sun_elev_rad) + np.sin(
        slope_rad
    ) * np.cos(sun_elev_rad) * np.cos(sun_az_rad - aspect_rad)

    # Clamp to valid range
    cos_incidence = np.clip(cos_incidence, -1.0, 1.0)

    incidence_angle = np.degrees(np.arccos(cos_incidence))

    return incidence_angle


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Find locations where sun is perpendicular to terrain"
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        default="13.12.2025",
        help="Date (DD.MM.YYYY), default: 13.12.2025",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="12:22:00",
        help="Time UTC (HH:MM:SS), default: 12:22:00",
    )
    parser.add_argument(
        "-i",
        "--input-dom",
        type=str,
        default=DEFAULT_DOM,
        help=f"Path to DOM GeoTIFF (default: {DEFAULT_DOM})",
    )
    parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=10,
        help="Number of best locations to show (default: 10)",
    )
    parser.add_argument(
        "--max-incidence",
        type=float,
        default=5.0,
        help="Maximum incidence angle to consider\
                            (default: 5.0 degrees)",
    )
    parser.add_argument(
        "--min-slope",
        type=float,
        default=30.0,
        help="Minimum slope to consider\
                            (default: 30.0 degrees)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SUCHE STANDORTE MIT SENKRECHTEM SONNENEINFALL")
    print("=" * 70)
    print(f"Datum/Zeit (UTC): {args.date} {args.time}")
    print(f"DOM-Datei: {args.input_dom}")
    print(f"Min. Hangneigung: {args.min_slope} Grad")
    print(f"Max. Einfallswinkel: {args.max_incidence} Grad")
    print("=" * 70)

    if not os.path.isfile(args.input_dom):
        print(f"\nFEHLER: DOM nicht gefunden: {args.input_dom}")
        sys.exit(1)

    # Parse datetime
    dt_utc = parse_datetime(args.date, args.time)

    # Load DEM
    print("\nLade DOM-Daten...")
    with rasterio.open(args.input_dom) as src:
        dem_data = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        bounds = src.bounds

        print(f"  Groesse: {dem_data.shape}")
        print(f"  Bounds: {bounds}")
        print(f"  NoData: {nodata}")

    # Calculate center of DEM for sun position
    center_e = (bounds.left + bounds.right) / 2
    center_n = (bounds.top + bounds.bottom) / 2
    center_lon, center_lat = lv95_to_wgs84(center_e, center_n)

    # Get sun position
    print("\nBerechne Sonnenposition...")
    sun_elevation, sun_azimuth = get_sun_pos(dt_utc, center_lat, center_lon)
    print(f"  Sonnen-Elevation: {sun_elevation:.2f} Grad")
    print(f"  Sonnen-Azimut: {sun_azimuth:.2f} Grad")

    if sun_elevation < 0:
        print("\nFEHLER: Sonne ist unter dem Horizont!")
        sys.exit(1)

    # Calculate required slope for perpendicular incidence
    required_slope = 90.0 - sun_elevation
    print("\nFuer senkrechten Einfall benoetigt:")
    print(f"  Hangneigung: {required_slope:.1f} Grad")
    print(f"  Exposition: {sun_azimuth:.1f} Grad (Richtung Sonne)")

    # Calculate slope and aspect
    print("\nBerechne Hangneigung und Exposition...")
    slope, aspect = calculate_slope_aspect(dem_data, transform, nodata)
    print(
        f"  Hangneigung: {np.nanmin(slope):.1f} -\
              {np.nanmax(slope):.1f} Grad"
    )

    # Calculate incidence angle
    print("\nBerechne Einfallswinkel...")
    incidence = calc_incidence_angle(slope, aspect, sun_elevation, sun_azimuth)

    # Create mask for valid pixels
    valid_mask = (
        ~np.isnan(slope)
        & ~np.isnan(aspect)
        & ~np.isnan(incidence)
        & (dem_data != nodata if nodata is not None else True)
        & (dem_data > 0)  # Exclude water/nodata areas
        & (slope >= args.min_slope)  # Only consider steep slopes
        & (incidence <= args.max_incidence)  # Only near-perpendicular
    )

    num_candidates = np.sum(valid_mask)
    print(
        f"\nGefundene Kandidaten\
           (Einfallswinkel < {args.max_incidence} Grad): {num_candidates}"
    )

    if num_candidates == 0:
        print("\nKeine Standorte gefunden mit den gegebenen Kriterien.")
        print("Versuche:")
        print(f"  --max-incidence {args.max_incidence * 2}")
        print(f"  --min-slope {args.min_slope / 2}")
        sys.exit(0)

    # Get coordinates of valid pixels
    rows, cols = np.where(valid_mask)
    incidence_values = incidence[valid_mask]
    slope_values = slope[valid_mask]
    aspect_values = aspect[valid_mask]
    elevation_values = dem_data[valid_mask]

    # Sort by incidence angle (lowest first = most perpendicular)
    sort_idx = np.argsort(incidence_values)

    # Take top N results
    n_results = min(args.num_results, len(sort_idx))

    print("\n" + "=" * 70)
    print(f"TOP {n_results} STANDORTE MIT SENKRECHTEM SONNENEINFALL")
    print("=" * 70)

    results = []

    for i in range(n_results):
        idx = sort_idx[i]
        row = rows[idx]
        col = cols[idx]

        # Calculate LV95 coordinates
        x_lv95 = transform[2] + transform[0] * (col + 0.5)
        y_lv95 = transform[5] + transform[4] * (row + 0.5)

        # Get values
        inc = incidence_values[idx]
        slp = slope_values[idx]
        asp = aspect_values[idx]
        elev = elevation_values[idx]

        print(f"\n[{i+1}] Einfallswinkel: {inc:.2f} Grad")
        print(f"    LV95: E={x_lv95:.2f} / N={y_lv95:.2f}")
        print(f"    Hoehe: {elev:.1f} m")
        print(f"    Hangneigung: {slp:.1f} Grad")
        print(f"    Exposition: {asp:.1f} Grad")
        print(f"    Pixel: Row={row}, Col={col}")

        results.append(
            {
                "rank": i + 1,
                "incidence_deg": inc,
                "east_lv95": x_lv95,
                "north_lv95": y_lv95,
                "elevation_m": elev,
                "slope_deg": slp,
                "aspect_deg": asp,
            }
        )

    # Print best result for easy copy-paste
    best = results[0]
    print("\n" + "=" * 70)
    print("BESTER STANDORT")
    print("=" * 70)
    print("\nKoordinaten (LV95):")
    print(f"  East:  {best['east_lv95']:.2f}")
    print(f"  North: {best['north_lv95']:.2f}")
    print("\nZum Testen mit shadow_check_location.py:")
    print(
        f"  python shadow_check_location.py -d {args.date} -t {args.time} "
        f"-e {best['east_lv95']:.0f} -n {best['north_lv95']:.0f}"
    )
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
