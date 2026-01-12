# Description: Calculate incidence angle for a grid (configurable resolution)
#              Outputs a single-band float32 GeoTIFF with incidence angles
#
# Usage: python incidence_angle_grid.py --x 2600000 --y 1200000
#                                       --date "13.12.2025" --time "12:22:00"
#
# Required: pip install rasterio pyproj numpy skyfield horayzon
#
# Copyright (c) 2025
# MIT License

import argparse
import logging
import math
import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from datetime import datetime, timezone
from pyproj import Transformer
import horayzon as hray
from skyfield.api import load, wgs84

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DOM = r"C:\temp\DOM\Thinout_highest_object_10m_LV95_LHN95.tif"
GRID_SIZE = 1000  # meters
GRID_STEP = 10   # meters (100m / 10m = 10 points per axis = 100 total)
NUM_POINTS = GRID_SIZE // GRID_STEP  # 10 x 10 = 100 points


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def lv95_to_wgs84(e_lv95, n_lv95):
    """Convert LV95 (EPSG:2056) to WGS84 (EPSG:4326) coordinates."""
    transformer = Transformer.from_crs(
        "EPSG:2056", "EPSG:4326", always_xy=True
        )
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


def get_sun_position(dt_utc, lat, lon, ephemeris_dir=None):
    """Calculate sun position using skyfield
    (same as shadow_check_location.py)."""
    if ephemeris_dir:
        load.directory = ephemeris_dir

    planets = load("de421.bsp")
    sun = planets["sun"]
    earth = planets["earth"]
    loc = earth + wgs84.latlon(lat, lon)

    ts = load.timescale()
    t = ts.from_datetime(dt_utc)
    astrometric = loc.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()

    return alt.degrees, az.degrees


def calculate_slope_aspect(dem_data, transform, x_loc, y_loc, x_full, y_full):
    """
    Calculate slope and aspect at a given location from DEM
    using horayzon's slope_plane_meth.

    Returns:
        slope_deg: Slope angle in degrees (0 = flat, 90 = vertical)
        aspect_deg: Aspect angle in degrees from North,
        clockwise (0=N, 90=E, 180=S, 270=W)
        This is the direction the slope faces.
    """
    # Get pixel coordinates using argmin (same as shadow_check_location.py)
    col = np.argmin(np.abs(x_full - x_loc))
    row = np.argmin(np.abs(y_full - y_loc))

    nrows, ncols = dem_data.shape

    # Need at least 2 pixel border for slope_plane_meth
    # (it removes 1 pixel border)
    if col < 2 or col >= ncols - 2 or row < 2 or row >= nrows - 2:
        return None, None

    # Extract 5x5 window around the point
    window_size = 5
    half = window_size // 2

    rows_idx = slice(row - half, row + half + 1)
    cols_idx = slice(col - half, col + half + 1)

    # Get coordinate arrays for the window
    x_window = x_full[cols_idx].astype(np.float32)
    y_window = y_full[rows_idx].astype(np.float32)

    x_2d, y_2d = np.meshgrid(x_window, y_window)
    elevation_window = dem_data[rows_idx, cols_idx].astype(np.float32)

    # Use horayzon's slope_plane_meth
    vec_tilt = hray.topo_param.slope_plane_meth(x_2d, y_2d, elevation_window)

    # Get the center pixel
    # (slope_plane_meth returns array with 1 pixel border removed)
    # So a 5x5 input gives 3x3 output, center is at [1,1]
    center_idx = vec_tilt.shape[0] // 2
    vec_center = vec_tilt[center_idx, center_idx]

    # Calculate slope from tilt vector
    # (vec_tilt[2] is the z-component of the normal)
    slope_deg = np.rad2deg(np.arccos(vec_center[2]))

    # Calculate aspect from tilt vector (x and y components)
    aspect_vec = vec_center[:2]
    aspect_deg = np.rad2deg(np.arctan2(aspect_vec[0], aspect_vec[1]))
    if aspect_deg < 0:
        aspect_deg += 360

    return slope_deg, aspect_deg


def sun_height_over_slope(slope_deg, aspect_deg, sun_elev_deg, sun_az_deg):
    """
    Calculate incidence angle (theta) for sun on a tilted surface.

    Args:
        slope_deg: Slope angle in degrees
        aspect_deg: Aspect angle in degrees (from North, clockwise)
        sun_elev_deg: Sun elevation angle in degrees
        sun_az_deg: Sun azimuth in degrees (from North, clockwise)

    Returns:
        theta: Incidence angle in degrees
        (0 = perpendicular to surface, 90 = parallel)
    """
    beta = math.radians(slope_deg)
    gamma = math.radians(aspect_deg)
    alpha = math.radians(sun_elev_deg)
    A = math.radians(sun_az_deg)

    cos_theta = math.sin(alpha) * math.cos(beta)
    cos_theta += math.cos(alpha) * math.sin(beta) * math.cos(A - gamma)

    # Clamp to valid range
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))

    return theta


def calculate_incidence_grid(
        origin_x, origin_y, dt_utc, dem_path, output_dir=None
        ):
    """
    Calculate incidence angle for a grid.

    Args:
        origin_x: X coordinate of lower-left corner (LV95)
        origin_y: Y coordinate of lower-left corner (LV95)
        dt_utc: Datetime in UTC
        dem_path: Path to DEM file
        output_dir: Directory for ephemeris data (optional)

    Returns:
        grid: numpy array with incidence angles (float32)
        transform: Affine transform for GeoTIFF
        points: List of tuples (x, y, incidence_angle) for CSV export
    """
    # Initialize output grid
    grid = np.full((NUM_POINTS, NUM_POINTS), np.nan, dtype=np.float32)
    points = []  # List to store (x, y, angle) for CSV

    # Load DEM
    logger.info(f"Lade DEM: {dem_path}")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_transf = src.transform
        nodata = src.nodata
        nrows, ncols = dem_data.shape
        logger.info(f"  DEM Groesse: {dem_data.shape}")

        # Create coordinate arrays for slope_plane_meth
        x_full = np.array(
            [
                dem_transf[2] + dem_transf[0] * (i + 0.5) for i in range(nrows)
                ]
            )
        y_full = np.array(
            [
                dem_transf[5] + dem_transf[4] * (i + 0.5) for i in range(nrows)
                ]
            )

    # Calculate sun position at grid center
    center_x = origin_x + GRID_SIZE / 2
    center_y = origin_y + GRID_SIZE / 2
    lon, lat = lv95_to_wgs84(center_x, center_y)
    sun_elev, sun_az = get_sun_position(
        dt_utc, lat, lon, ephemeris_dir=output_dir
        )

    logger.info("Sonnenposition (Zentrum des Gitters):")
    logger.info(f"  Elevation: {sun_elev:.2f} Grad")
    logger.info(f"  Azimut: {sun_az:.2f} Grad")

    if sun_elev < 0:
        logger.warning("Sonne unter Horizont (Nacht)")

    # Calculate incidence angle for each grid point
    logger.info(
        f"Berechne Inzidenzwinkel fuer {NUM_POINTS}x{NUM_POINTS} Punkte..."
        )

    valid_count = 0
    for row in range(NUM_POINTS):
        for col in range(NUM_POINTS):
            # Calculate LV95 coordinates
            # row 0 = top (north), row 9 = bottom (south)
            # col 0 = left (west), col 9 = right (east)
            # Flip for raster convention
            x = origin_x + col * GRID_STEP
            y = origin_y + (NUM_POINTS - 1 - row) * GRID_STEP

            # Get slope and aspect from DEM using horayzon's slope_plane_meth
            slope, aspect = calculate_slope_aspect(
                dem_data, dem_transf, x, y, x_full, y_full
                )

            if slope is None or aspect is None:
                points.append((x, y, np.nan))
                continue

            # Check for NoData in DEM (use argmin for consistency)
            dem_col = np.argmin(np.abs(x_full - x))
            dem_row = np.argmin(np.abs(y_full - y))

            if (
                    0 <= dem_row < dem_data.shape[0] and
                    0 <= dem_col < dem_data.shape[1]
            ):
                elev = dem_data[dem_row, dem_col]
                if nodata is not None and elev == nodata:
                    points.append((x, y, np.nan))
                    continue
                if np.isnan(elev) or elev == 0:
                    points.append((x, y, np.nan))
                    continue
            else:
                points.append((x, y, np.nan))
                continue

            # Calculate incidence angle
            theta = sun_height_over_slope(slope, aspect, sun_elev, sun_az)
            grid[row, col] = theta
            points.append((x, y, theta))
            logger.debug(f"E={x} / N={y} / Theta {theta:0.2f}")
            valid_count += 1

    logger.info(
        f"  {valid_count} von {NUM_POINTS * NUM_POINTS} Punkten berechnet"
        )

    # Create transform for output GeoTIFF
    # Origin is upper-left corner in raster convention
    upper_left_x = origin_x
    upper_left_y = origin_y + GRID_SIZE

    out_transform = from_bounds(
        upper_left_x,                    # west
        origin_y,                        # south
        origin_x + GRID_SIZE,            # east
        upper_left_y,                    # north
        NUM_POINTS,                      # width
        NUM_POINTS                       # height
    )

    return grid, out_transform, points


def write_csv(points, output_path):
    """Write points to CSV with columns X, Y, Winkel."""
    logger.info(f"Schreibe CSV: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("X,Y,Winkel\n")
        for x, y, angle in points:
            if np.isnan(angle):
                f.write(f"{x:.2f},{y:.2f},\n")
            else:
                f.write(f"{x:.2f},{y:.2f},{angle:.2f}\n")

    logger.info(f"  {len(points)} Punkte geschrieben")


def write_geotiff(grid, transform, output_path):
    """Write grid to GeoTIFF with float32 data type."""
    logger.info(f"Schreibe GeoTIFF: {output_path}")

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=np.float32,
        crs=CRS.from_epsg(2056),  # LV95
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(grid, 1)

    logger.info(f"  Groesse: {grid.shape[0]} x {grid.shape[1]} Pixel")
    logger.info("  Datentyp: float32")
    logger.info("  CRS: EPSG:2056 (LV95)")


def setup_logging(verbose=False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress verbose output from third-party libraries
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('pyproj').setLevel(logging.WARNING)
    logging.getLogger('skyfield').setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Berechne Inzidenzwinkel fuer ein Raster'
    )
    parser.add_argument('--x', type=float, default=2600000.0,
                        help="Easting in LV95 [m], default: 2600000.0 (Bern)")
    parser.add_argument('--y', type=float, default=1200000.0,
                        help="Northing in LV95 [m], default: 1200000.0 (Bern)")
    parser.add_argument('--date', type=str, default="13.12.2025",
                        help='Datum (Format: DD.MM.YYYY)')
    parser.add_argument('--time', type=str, default="12:22:00",
                        help='Zeit UTC (Format: HH:MM:SS oder HH:MM)')
    parser.add_argument('--dem', type=str, default=DEFAULT_DOM,
                        help=f'Pfad zum DEM (default: {DEFAULT_DOM})')
    parser.add_argument('--output', type=str, default="C:/temp/",
                        help='Ausgabe-Pfad fuer GeoTIFF\
                            (default: incidence_<x>_<y>.tif)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output (DEBUG level)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("INZIDENZWINKEL-RASTER BERECHNUNG")
    logger.info("=" * 70)
    logger.info(f"Linke untere Ecke: E={args.x:.2f} / N={args.y:.2f}")
    logger.info(f"Rastergroesse: {GRID_SIZE}m x {GRID_SIZE}m")
    logger.info(
        f"Aufloesung: {GRID_STEP}m\
            ({NUM_POINTS}x{NUM_POINTS} = {NUM_POINTS*NUM_POINTS} Punkte)"
        )
    logger.info(f"Datum/Zeit (UTC): {args.date} {args.time}")
    logger.info("=" * 70)

    # Check DEM exists
    if not os.path.isfile(args.dem):
        logger.error(f"DEM nicht gefunden: {args.dem}")
        sys.exit(1)

    # Parse datetime
    dt_utc = parse_datetime(args.date, args.time)

    # Determine output directory (needed for ephemeris data)
    basename = f"incidence_{int(args.x)}_{int(args.y)}"
    filename_tif = f"{basename}.tif"
    filename_csv = f"{basename}.csv"

    if args.output:
        # Check if output is a directory
        if os.path.isdir(args.output) or args.output.endswith(('/', '\\')):
            # Ensure directory exists
            os.makedirs(args.output, exist_ok=True)
            output_dir = args.output
            output_path_tif = os.path.join(args.output, filename_tif)
            output_path_csv = os.path.join(args.output, filename_csv)
        else:
            # Output is a file path - use directory for both files
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = "."
            output_path_tif = args.output
            output_path_csv = os.path.splitext(args.output)[0] + ".csv"
    else:
        output_dir = "."
        output_path_tif = filename_tif
        output_path_csv = filename_csv

    # Calculate grid
    grid, transform, points = calculate_incidence_grid(
        args.x, args.y, dt_utc, args.dem, output_dir
        )

    # Write outputs
    write_geotiff(grid, transform, output_path_tif)
    write_csv(points, output_path_csv)

    # Log statistics
    valid_values = grid[~np.isnan(grid)]
    if len(valid_values) > 0:
        logger.info("Statistik:")
        logger.info(f"  Min Inzidenzwinkel: {np.min(valid_values):.2f} Grad")
        logger.info(f"  Max Inzidenzwinkel: {np.max(valid_values):.2f} Grad")
        logger.info(f"  Mittelwert: {np.mean(valid_values):.2f} Grad")
    else:
        logger.warning("Keine gueltigen Werte berechnet!")

    logger.info("=" * 70)
    logger.info("FERTIG")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
