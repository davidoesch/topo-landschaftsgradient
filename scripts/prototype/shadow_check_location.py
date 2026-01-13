# Description: Check if a specific location (in LV95 coordinates) is in shadow
#              or sunlit at a given date/time. Uses a DOM (GeoTIFF) in LV95 and
#              distinguishes between self-shading and terrain-shading.
#              Ignore Earth's surface curvature (valid for Switzerland).


# Load modules
import logging
import os
import sys
import argparse
import numpy as np
import rasterio
from datetime import datetime, timezone
from skyfield.api import load, wgs84
from pyproj import CRS, Transformer
import horayzon as hray

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Functions
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
    # Try different formats
    for fmt in [
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]:
        try:
            dt_local = datetime.strptime(dt_str, fmt)
            # Assume input is local time (CET/CEST) - convert to UTC
            # For simplicity, we assume the user provides UTC time
            # or we could add timezone handling
            return dt_local.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date/time: {dt_str}")


def get_shadow_description(shadow_value):
    """Get human-readable description of shadow value."""
    descriptions = {
        0: "BELEUCHTET (illuminated) - Der Ort liegt in der Sonne",
        1: "SELBSTBESCHATTET (self-shaded) - Der Hang zeigt von der Sonne weg",
        2: "GELAENDEBESCHATTET (terrain-shaded) - Schatten umliegende Berge",
        3: "NICHT BERUECKSICHTIGT (not considered) - Ausserhalb der Maske",
        -1: "NODATA - Keine gueltige Hoehe am Standort",
    }
    return descriptions.get(shadow_value, f"Unbekannter Wert: {shadow_value}")


def calculate_incidence_angle(slope_deg, aspect_deg, sun_elev_deg, sun_az_deg):
    """
    Calculate incidence angle (theta) for sun on a tilted surface.

    The incidence angle is the angle between the sun ray and the surface
    normal.
    - 0° = sun perpendicular to surface (maximum irradiance)
    - 90° = sun parallel to surface (no direct irradiance)
    - >90° = sun behind the surface (self-shading)

    Args:
        slope_deg: Slope angle in degrees (0 = flat, 90 = vertical)
        aspect_deg: Aspect angle in degrees
            (from North, clockwise: 0=N, 90=E,180=S, 270=W)
        sun_elev_deg: Sun elevation angle in degrees above horizon
        sun_az_deg: Sun azimuth in degrees (from North, clockwise)

    Returns:
        theta: Incidence angle in degrees
    """
    beta = np.radians(slope_deg)
    gamma = np.radians(aspect_deg)
    alpha = np.radians(sun_elev_deg)
    A = np.radians(sun_az_deg)

    cos_theta = np.sin(alpha) * np.cos(beta)
    cos_theta += np.cos(alpha) * np.sin(beta) * np.cos(A - gamma)

    # Clamp to valid range
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    return theta


def setup_logging(verbose=False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose output from third-party libraries
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("pyproj").setLevel(logging.WARNING)
    logging.getLogger("skyfield").setLevel(logging.WARNING)
    logging.getLogger("horayzon").setLevel(
        logging.WARNING if not verbose else logging.DEBUG
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    # -------------------------------------------------------------------------
    # Parse arguments or use defaults
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Check if a location is in shadow at a specific time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shadow_check_location.py --date 13.12.2025
                                  --time 12:22:00
                                  --east 2600000
                                  --north 1200000
  python shadow_check_location.py -d 21.06.2025
                                  -t 08:00:00
                                  -e 2683000
                                  -n 1248000
        """,
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
        "-e",
        "--east",
        type=float,
        default=2600000.0,
        help="Easting in LV95 [m], default: 2600000.0 (Bern)",
    )
    parser.add_argument(
        "-n",
        "--north",
        type=float,
        default=1200000.0,
        help="Northing in LV95 [m], default: 1200000.0 (Bern)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="C:/temp/shadow_check/",
        help="Output directory, default: C:/temp/shadow_check/",
    )
    parser.add_argument(
        "-s",
        "--search-dist",
        type=float,
        default=20.0,
        help="Search distance for terrain shading [km], default: 20.0",
    )
    parser.add_argument(
        "-i",
        "--input-dom",
        type=str,
        default="C:/temp/DOM/Thinout_highest_object_10m_LV95_LHN95.tif",
        help="Path to DOM GeoTIFF in LV95",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    # Location (LV95 - used directly)
    e_lv95, n_lv95 = args.east, args.north

    # Convert to WGS84 for sun position calculation
    lon_loc, lat_loc = lv95_to_wgs84(e_lv95, n_lv95)

    # Parse date and time
    dt_utc = parse_datetime(args.date, args.time)

    # Domain around location (in LV95 coordinates)
    dist_search = args.search_dist  # [km]
    buffer = dist_search * 1000.0  # [m]
    domain = {
        "x_min": e_lv95 - buffer,
        "x_max": e_lv95 + buffer,
        "y_min": n_lv95 - buffer,
        "y_max": n_lv95 + buffer,
    }

    # Paths
    path_out = args.output
    file_dom = args.input_dom

    # -------------------------------------------------------------------------
    # Log settings
    # -------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("SCHATTEN-CHECK FUER EINZELNEN STANDORT")
    logger.info("=" * 60)
    logger.info(f"Datum/Zeit (UTC): {dt_utc.strftime('%d.%m.%Y %H:%M:%S')}")
    logger.info(f"Standort LV95:    E={e_lv95:.2f} / N={n_lv95:.2f}")
    logger.info(f"Standort WGS84:   Lon={lon_loc:.5f} / Lat={lat_loc:.5f}")
    logger.info(f"Suchradius:       {dist_search} km")
    logger.info(f"DOM-Datei:        {file_dom}")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------------------------------

    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    # -------------------------------------------------------------------------
    # Check if DOM file exists
    # -------------------------------------------------------------------------

    if not os.path.isfile(file_dom):
        logger.error(f"DOM-Datei nicht gefunden: {file_dom}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load DOM data (GeoTIFF in LV95)
    # -------------------------------------------------------------------------

    logger.info("Lade DOM-Daten...")
    with rasterio.open(file_dom) as src:
        # Read elevation data
        elev_full_raw = src.read(1)
        transform = src.transform
        nodata = src.nodata

        # Get coordinate arrays
        nrows, ncols = elev_full_raw.shape
        x_full = np.array(
            [transform[2] + transform[0] * (i + 0.5) for i in range(ncols)]
        )
        y_full = np.array(
            [transform[5] + transform[4] * (i + 0.5) for i in range(nrows)]
        )

    # Create a mask for nodata values (before replacing them)
    nodata_mask = np.zeros_like(elev_full_raw, dtype=bool)
    if nodata is not None:
        nodata_mask = elev_full_raw == nodata

    # Also mark NaN/inf values as nodata
    nodata_mask = (
        nodata_mask | np.isnan(elev_full_raw) | np.isinf(elev_full_raw)
    )

    # Handle nodata values - replace with 0 (sea level) for terrain processing
    elev_full = elev_full_raw.copy()
    if nodata is not None:
        elev_full = np.where(elev_full == nodata, 0.0, elev_full)

    # Replace any remaining NaN/inf values
    elev_full = np.nan_to_num(elev_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract domain with buffer for outer terrain
    dom_outer = hray.domain.planar_grid(domain, dist_search)

    # Find indices for clipping to dom_outer
    x_mask = (x_full >= dom_outer["x_min"]) & (x_full <= dom_outer["x_max"])
    y_mask = (y_full >= dom_outer["y_min"]) & (y_full <= dom_outer["y_max"])

    if not np.any(x_mask) or not np.any(y_mask):
        logger.error("Der Standort liegt ausserhalb des DOM!")
        sys.exit(1)

    # Convert to float32 as required by horayzon
    x = x_full[x_mask].astype(np.float32)
    y = y_full[y_mask].astype(np.float32)
    elevation = elev_full[np.ix_(y_mask, x_mask)].astype(np.float32)
    nodata_mask_clipped = nodata_mask[np.ix_(y_mask, x_mask)]

    # Ensure y is in descending order (north to south)
    if y[0] < y[-1]:
        y = y[::-1]
        elevation = elevation[::-1, :]
        nodata_mask_clipped = nodata_mask_clipped[::-1, :]

    logger.info(f"DEM Groesse: {elevation.shape}")
    logger.info(f"Z-Bereich: {elevation.min():.1f} - {elevation.max():.1f} m")

    # -------------------------------------------------------------------------
    # Compute indices of inner domain
    # -------------------------------------------------------------------------

    slice_in = (
        slice(
            np.where(y >= domain["y_max"])[0][-1],
            np.where(y <= domain["y_min"])[0][0] + 1,
        ),
        slice(
            np.where(x <= domain["x_min"])[0][-1],
            np.where(x >= domain["x_max"])[0][0] + 1,
        ),
    )
    offset_0 = slice_in[0].start
    offset_1 = slice_in[1].start

    logger.info(f"Innere Domaene: {elevation[slice_in].shape}")
    elevation_in = np.ascontiguousarray(elevation[slice_in])

    # Find index of location in grid
    idx_x = np.argmin(np.abs(x[slice_in[1]] - e_lv95))
    idx_y = np.argmin(np.abs(y[slice_in[0]] - n_lv95))
    elev_loc = elevation_in[idx_y, idx_x]

    # Check if location has NoData in the clipped mask
    nodata_mask_in = nodata_mask_clipped[slice_in]
    is_nodata_loc = nodata_mask_in[idx_y, idx_x]

    logger.debug(f"Standort im Gitter: idx_x={idx_x}, idx_y={idx_y}")
    logger.info(f"Hoehe am Standort: {elev_loc:.1f} m")

    # Check for NoData or zero elevation at location
    date_str = f"Datum/Zeit (UTC):    {dt_utc.strftime('%d.%m.%Y %H:%M:%S')}"
    if is_nodata_loc or elev_loc == 0.0:
        logger.info("=" * 60)
        logger.info("ERGEBNIS")
        logger.info("=" * 60)
        logger.info(date_str)
        logger.info(f"Standort LV95:       E={e_lv95:.2f} / N={n_lv95:.2f}")
        logger.info(f"Hoehe:               {elev_loc:.1f} m")
        logger.info("-" * 60)
        logger.info(f"SCHATTEN-STATUS:     {get_shadow_description(-1)}")
        logger.info("=" * 60)
        return {
            "datetime_utc": dt_utc,
            "location_lv95": (e_lv95, n_lv95),
            "location_wgs84": (lon_loc, lat_loc),
            "elevation_m": elev_loc,
            "slope_deg": None,
            "aspect_deg": None,
            "sun_elevation_deg": None,
            "sun_azimuth_deg": None,
            "incidence_angle_deg": None,
            "shadow_value": -1,
            "shadow_description": get_shadow_description(-1),
        }

    # -------------------------------------------------------------------------
    # Prepare terrain data
    # -------------------------------------------------------------------------

    logger.info("Bereite Gelaendedaten vor...")

    # Create directional unit vectors (up) for inner domain
    dem_dim_0, dem_dim_1 = elevation.shape
    inner_dim_0, inner_dim_1 = elevation_in.shape
    vec_norm = np.zeros((inner_dim_0, inner_dim_1, 3), dtype=np.float32)
    vec_norm[:, :, 2] = 1.0

    # Merge vertex coordinates and pad geometry buffer
    v_grid = hray.auxiliary.rearrange_pad_buffer(*np.meshgrid(x, y), elevation)

    # Compute slope
    x_2d, y_2d = np.meshgrid(x, y)
    slice_in_a1 = (
        slice(slice_in[0].start - 1, slice_in[0].stop + 1),
        slice(slice_in[1].start - 1, slice_in[1].stop + 1),
    )
    vec_tilt = np.ascontiguousarray(
        hray.topo_param.slope_plane_meth(
            x_2d[slice_in_a1], y_2d[slice_in_a1], elevation[slice_in_a1]
        )[1:-1, 1:-1]
    )

    # Compute slope angle and aspect at location
    slope_loc = np.rad2deg(np.arccos(vec_tilt[idx_y, idx_x, 2]))
    aspect_vec = vec_tilt[idx_y, idx_x, :2]
    aspect_loc = np.rad2deg(np.arctan2(aspect_vec[0], aspect_vec[1]))
    if aspect_loc < 0:
        aspect_loc += 360.0

    logger.info(f"Hangneigung am Standort: {slope_loc:.1f} Grad")
    logger.info(f"Exposition am Standort: {aspect_loc:.1f} Grad (von Nord)")

    # Compute surface enlargement factor
    surf_enl_fac = 1.0 / (vec_norm * vec_tilt).sum(axis=2)

    # -------------------------------------------------------------------------
    # Initialize terrain
    # -------------------------------------------------------------------------

    logger.info("Initialisiere Terrain...")
    mask = np.ones(vec_tilt.shape[:2], dtype=np.uint8)
    terrain = hray.shadow.Terrain()
    terrain.initialise(
        v_grid,
        dem_dim_0,
        dem_dim_1,
        offset_0,
        offset_1,
        vec_tilt,
        vec_norm,
        surf_enl_fac,
        mask=mask,
        elevation=elevation_in,
        refrac_cor=False,
    )

    # -------------------------------------------------------------------------
    # Calculate sun position
    # -------------------------------------------------------------------------

    logger.info("Berechne Sonnenposition...")
    load.directory = path_out
    planets = load("de421.bsp")
    sun = planets["sun"]
    earth = planets["earth"]
    loc_or = earth + wgs84.latlon(lat_loc, lon_loc)

    ts = load.timescale()
    t = ts.from_datetime(dt_utc)
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()

    sun_elevation = alt.degrees
    sun_azimuth = az.degrees

    logger.info(f"Sonnen-Elevation: {sun_elevation:.2f} Grad")
    logger.info(f"Sonnen-Azimut: {sun_azimuth:.2f} Grad (von Nord)")

    # Calculate incidence angle (sun zenith on tilted surface)
    incidence_angle = calculate_incidence_angle(
        slope_loc, aspect_loc, sun_elevation, sun_azimuth
    )
    logger.info(
        f"Inzidenzwinkel: {incidence_angle:.2f} \
        Grad (Zenitwinkel auf geneigter Flaeche)"
    )

    # Check if sun is below horizon
    if sun_elevation < 0:
        logger.info("=" * 60)
        logger.warning("ERGEBNIS: Die Sonne ist unter dem Horizont!")
        logger.info(f"Sonnen-Elevation: {sun_elevation:.2f} Grad")
        logger.info("Der gesamte Ort liegt in der Nacht.")
        logger.info("=" * 60)
        return

    # -------------------------------------------------------------------------
    # Compute shadow
    # -------------------------------------------------------------------------

    logger.info("Berechne Schatten...")

    # Sun position in ENU coordinates
    x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
    y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
    z_sun = d.m * np.sin(alt.radians)
    sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

    # Compute shadow map
    shadow_buffer = np.zeros(vec_tilt.shape[:2], dtype=np.uint8)
    terrain.shadow(sun_position, shadow_buffer)

    # Get shadow value at location
    shadow_value = shadow_buffer[idx_y, idx_x]

    # -------------------------------------------------------------------------
    # Output result
    # -------------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("ERGEBNIS")
    logger.info("=" * 60)
    logger.info(f"Datum/Zeit (UTC):    {dt_utc.strftime('%d.%m.%Y %H:%M:%S')}")
    logger.info(f"Standort LV95:       E={e_lv95:.2f} / N={n_lv95:.2f}")
    logger.info(f"Hoehe:               {elev_loc:.1f} m")
    logger.info(f"Hangneigung:         {slope_loc:.1f} Grad")
    logger.info(f"Exposition:          {aspect_loc:.1f} Grad")
    logger.info(f"Sonnen-Elevation:    {sun_elevation:.2f} Grad")
    logger.info(f"Sonnen-Azimut:       {sun_azimuth:.2f} Grad")
    logger.info(f"Inzidenzwinkel:      {incidence_angle:.2f} Grad")
    logger.info("-" * 60)
    logger.info(f"SCHATTEN-STATUS:     {get_shadow_description(shadow_value)}")
    logger.info("=" * 60)

    # Return values for programmatic use
    return {
        "datetime_utc": dt_utc,
        "location_lv95": (e_lv95, n_lv95),
        "location_wgs84": (lon_loc, lat_loc),
        "elevation_m": elev_loc,
        "slope_deg": slope_loc,
        "aspect_deg": aspect_loc,
        "sun_elevation_deg": sun_elevation,
        "sun_azimuth_deg": sun_azimuth,
        "incidence_angle_deg": incidence_angle,
        "shadow_value": shadow_value,
        "shadow_description": get_shadow_description(shadow_value),
    }


if __name__ == "__main__":
    result = main()
