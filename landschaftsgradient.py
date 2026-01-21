import logging
import os
import rasterio
import numpy as np
import horayzon as hray
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
from datetime import datetime, timezone
from skyfield.api import load, wgs84


def lv95_to_wgs84(e_lv95, n_lv95):
    """Convert LV95 (EPSG:2056) to WGS84 (EPSG:4326) coordinates."""
    crs_lv95 = CRS.from_epsg(2056)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_lv95, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(e_lv95, n_lv95)
    return lon, lat


def parse_datetime(dateoi, timeoi):
    """Parse date and time strings to datetime object (UTC)."""
    dt_str = f"{dateoi} {timeoi}"
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


def calc_sunpos(lat_loc, lon_loc, dt_utc, output_path, planets_file):
    logging.info("Berechne Sonnenposition...")
    load.directory = output_path
    planets = load(planets_file)
    sun = planets["sun"]
    earth = planets["earth"]
    loc_or = earth + wgs84.latlon(lat_loc, lon_loc)
    ts = load.timescale()
    t = ts.from_datetime(dt_utc)
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    return alt.degrees, az.degrees


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


class DOM:
    def __init__(self, dom):
        self.__dom = dom
        self.__checkinput()
        self.__x_full, self.__y_full, self.__elev_full, self.__nodata_mask = (
            self.__loadingDOM()
        )

    @property
    def x_full(self):
        return self.__x_full

    @property
    def y_full(self):
        return self.__y_full

    @property
    def elev_full(self):
        return self.__elev_full

    @property
    def nodata_mask(self):
        return self.__nodata_mask

    def __checkinput(self):
        if os.path.isfile(self.__dom):
            logging.debug(f"DOM {self.__dom} found")
        else:
            raise AttributeError(f"DOM {self.__dom} not found")

    def __loadingDOM(self):
        logging.info("Lade DOM-Daten...")
        with rasterio.open(self.__dom) as src:
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
        logging.info("Nodata has been masked")

        # Also mark NaN/inf values as nodata
        nodata_mask = nodata_mask | np.isnan(elev_full_raw) | np.isinf(elev_full_raw)
        logging.info("NaN/inf values as nodata")

        # Handle nodata values - replace with 0 (sea level) for terrain processing
        elev_full = elev_full_raw.copy()
        if nodata is not None:
            elev_full = np.where(elev_full == nodata, 0.0, elev_full)

        # Replace any remaining NaN/inf values
        elev_full = np.nan_to_num(elev_full, nan=0.0, posinf=0.0, neginf=0.0)

        return x_full, y_full, elev_full, nodata_mask


class SonnenWinkel:

    def __init__(self, dom, planets, output_path):
        self.__dom = DOM(dom)
        self.__planets = planets
        self.__output_path = output_path
        self.__checkinput()

    def getdat_point(self, e_lv95, n_lv95, dateoi, timeoi, search_dist):
        domain = self.__get_domain(e_lv95, n_lv95, search_dist)
        dom_outer = hray.domain.planar_grid(domain, search_dist)
        dt_utc = parse_datetime(dateoi, timeoi)
        lon_loc, lat_loc = lv95_to_wgs84(e_lv95, n_lv95)
        x_mask = (self.__dom.x_full >= dom_outer["x_min"]) & (
            self.__dom.x_full <= dom_outer["x_max"]
        )
        y_mask = (self.__dom.y_full >= dom_outer["y_min"]) & (
            self.__dom.y_full <= dom_outer["y_max"]
        )
        if np.any(x_mask) and np.any(y_mask):
            # Convert to float32 as required by horayzon
            x = self.__dom.x_full[x_mask].astype(np.float32)
            y = self.__dom.y_full[y_mask].astype(np.float32)
            elevation = self.__dom.elev_full[np.ix_(y_mask, x_mask)].astype(np.float32)
            nodata_mask_clipped = self.__dom.nodata_mask[np.ix_(y_mask, x_mask)]
            # Ensure y is in descending order (north to south)
            if y[0] < y[-1]:
                y = y[::-1]
                elevation = elevation[::-1, :]
                nodata_mask_clipped = nodata_mask_clipped[::-1, :]
            logging.info(f"DEM Groesse: {elevation.shape}")
            logging.info(f"Z-Bereich: {elevation.min():.1f} - {elevation.max():.1f} m")

            slice_in = self.__get_slice(x, y, domain)

            logging.info(f"Innere Domaene: {elevation[slice_in].shape}")
            elevation_in = np.ascontiguousarray(elevation[slice_in])

            # Find index of location in grid
            idx_x = np.argmin(np.abs(x[slice_in[1]] - e_lv95))
            idx_y = np.argmin(np.abs(y[slice_in[0]] - n_lv95))
            elev_loc = elevation_in[idx_y, idx_x]

            # Check if location has NoData in the clipped mask
            nodata_mask_in = nodata_mask_clipped[slice_in]
            is_nodata_loc = nodata_mask_in[idx_y, idx_x]
            logging.info(f"Standort im Gitter: idx_x={idx_x}, idx_y={idx_y}")
            logging.info(f"Hoehe am Standort: {elev_loc:.1f} m")

            if not is_nodata_loc and elev_loc > 0.0:
                logging.info("Bereite Gelaendedaten vor...")
                # Create directional unit vectors (up) for inner domain
                slope_loc, aspect_loc, vec_tilt = self.__calc_slope_angles(
                    elevation, elevation_in, slice_in, idx_y, idx_x, x, y
                )
                logging.info(f"Hangneigung am Standort: {slope_loc:.1f} Grad")
                logging.info(
                    f"Exposition am Standort: {aspect_loc:.1f} Grad (von Nord)"
                )
                sun_elevation, sun_azimuth = calc_sunpos(
                    lat_loc, lon_loc, dt_utc, self.__output_path, self.__planets
                )
                logging.info(f"Sonnen-Elevation: {sun_elevation:.2f} Grad")
                logging.info(f"Sonnen-Azimut: {sun_azimuth:.2f} Grad (von Nord)")
                # Calculate incidence angle (sun zenith on tilted surface)
                incidence_angle = calculate_incidence_angle(
                    slope_loc, aspect_loc, sun_elevation, sun_azimuth
                )

                logging.info(
                    f"Inzidenzwinkel: {incidence_angle:.2f} Grad (Zenitwinkel auf geneigter Flaeche)"
                )
                if sun_elevation >= 0:
                    # Get shadow value at location
                    shadow_value = np.zeros(vec_tilt.shape[:2], dtype=np.uint8)[
                        idx_y, idx_x
                    ]
                    self.__write_results(
                        dt_utc,
                        e_lv95,
                        n_lv95,
                        elev_loc,
                        slope_loc,
                        aspect_loc,
                        sun_elevation,
                        sun_azimuth,
                        incidence_angle,
                        shadow_value,
                    )
                else:
                    logging.info("=" * 60)
                    logging.warning("ERGEBNIS: Die Sonne ist unter dem Horizont!")
                    logging.info(f"Sonnen-Elevation: {sun_elevation:.2f} Grad")
                    logging.info("Der gesamte Ort liegt in der Nacht.")
                    logging.info("=" * 60)
            else:
                logging.info("=" * 60)
                logging.info("ERGEBNIS")
                logging.info("=" * 60)
                logging.info(
                    f"Datum/Zeit (UTC):    {dt_utc.strftime('%d.%m.%Y %H:%M:%S')}"
                )
                logging.info(f"Standort LV95:       E={e_lv95:.2f} / N={n_lv95:.2f}")
                logging.info(f"Hoehe:               {elev_loc:.1f} m")
                logging.info("-" * 60)
                logging.info(
                    f"SCHATTEN-STATUS:     {self.__get_shadow_description(-1)}"
                )
                logging.info("=" * 60)
        else:
            logging.error("Point is outside DOM")

    def __checkinput(self):
        if os.path.isdir(self.__output_path):
            logging.debug(f"Outpath {self.__output_path} found")
        else:
            raise AttributeError(f"Outpath {self.__output_path} not found")

    def __calc_slope_angles(
        self, elevation, elevation_in, slice_in, idx_y, idx_x, x, y
    ):
        inner_dim_0, inner_dim_1 = elevation_in.shape
        vec_norm = np.zeros((inner_dim_0, inner_dim_1, 3), dtype=np.float32)
        vec_norm[:, :, 2] = 1.0
        x_2d, y_2d = np.meshgrid(x, y)
        slice_in_a1 = self.__get_slice_a1(slice_in)
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
        return slope_loc, aspect_loc, vec_tilt

    def __write_results(
        self,
        dt_utc,
        e_lv95,
        n_lv95,
        elev_loc,
        slope_loc,
        aspect_loc,
        sun_elevation,
        sun_azimuth,
        incidence_angle,
        shadow_value,
    ):
        logging.info("=" * 60)
        logging.info("ERGEBNIS")
        logging.info("=" * 60)
        logging.info(f"Datum/Zeit (UTC):    {dt_utc.strftime('%d.%m.%Y %H:%M:%S')}")
        logging.info(f"Standort LV95:       E={e_lv95:.2f} / N={n_lv95:.2f}")
        logging.info(f"Hoehe:               {elev_loc:.1f} m")
        logging.info(f"Hangneigung:         {slope_loc:.1f} Grad")
        logging.info(f"Exposition:          {aspect_loc:.1f} Grad")
        logging.info(f"Sonnen-Elevation:    {sun_elevation:.2f} Grad")
        logging.info(f"Sonnen-Azimut:       {sun_azimuth:.2f} Grad")
        logging.info(f"Inzidenzwinkel:      {incidence_angle:.2f} Grad")
        logging.info("-" * 60)
        logging.info(
            f"SCHATTEN-STATUS:     {self.__get_shadow_description(shadow_value)}"
        )
        logging.info("=" * 60)

    @staticmethod
    def __get_slice(x, y, domain):
        return (
            slice(
                np.where(y >= domain["y_max"])[0][-1],
                np.where(y <= domain["y_min"])[0][0] + 1,
            ),
            slice(
                np.where(x <= domain["x_min"])[0][-1],
                np.where(x >= domain["x_max"])[0][0] + 1,
            ),
        )

    @staticmethod
    def __get_slice_a1(slice_in):
        return (
            slice(slice_in[0].start - 1, slice_in[0].stop + 1),
            slice(slice_in[1].start - 1, slice_in[1].stop + 1),
        )

    @staticmethod
    def __get_shadow_description(shadow_value):
        """Get human-readable description of shadow value."""
        descriptions = {
            0: "BELEUCHTET (illuminated) - Der Ort liegt in der Sonne",
            1: "SELBSTBESCHATTET (self-shaded) - Der Hang zeigt von der Sonne weg",
            2: "GELAENDEBESCHATTET (terrain-shaded) - Schatten umliegende Berge",
            3: "NICHT BERUECKSICHTIGT (not considered) - Ausserhalb der Maske",
            -1: "NODATA - Keine gueltige Hoehe am Standort",
        }
        return descriptions.get(shadow_value, f"Unbekannter Wert: {shadow_value}")

    @staticmethod
    def __get_domain(e_lv95, n_lv95, search_dist):
        buffer = search_dist * 1000
        return {
            "x_min": e_lv95 - buffer,
            "x_max": e_lv95 + buffer,
            "y_min": n_lv95 - buffer,
            "y_max": n_lv95 + buffer,
        }


class InzidenWinkel:

    def __init__(self, dom, planets, output_path):
        self.__dom = DOM(dom)
        self.__planets = planets
        self.__output_path = output_path
        self.__checkinput()

    def calc_incidence_grid(self, e_lv95, dateoi, timeoi, n_lv95, grid_size, grid_step):
        """
        Calculate incidence angle for a grid.

        Returns:
            grid: numpy array with incidence angles (float32)
            transform: Affine transform for GeoTIFF
            points: List of tuples (x, y, incidence_angle) for CSV export
        """
        # Initialize output grid
        dt_utc = parse_datetime(dateoi, timeoi)
        num_points = grid_size // grid_step
        grid = np.full((num_points, num_points), np.nan, dtype=np.float32)
        points = []  # List to store (x, y, angle) for CSV

        # Calculate sun position at grid center
        center_x = e_lv95 + grid_size / 2
        center_y = n_lv95 + grid_size / 2
        lon, lat = lv95_to_wgs84(center_x, center_y)
        sun_elev, sun_az = calc_sunpos(
            lat, lon, dt_utc, self.__output_path, self.__planets
        )

        logging.info("Sonnenposition (Zentrum des Gitters):")
        logging.info(f"  Elevation: {sun_elev:.2f} Grad")
        logging.info(f"  Azimut: {sun_az:.2f} Grad")
        logging.info(f"Linke untere Ecke: E={e_lv95:.2f} / N={n_lv95:.2f}")
        logging.info(
            f"Aufloesung: {grid_step}m\
                    ({num_points}x{num_points} = {num_points*num_points} Punkte)"
        )
        if sun_elev < 0:
            logging.warning("Sonne unter Horizont (Nacht)")

        # Calculate incidence angle for each grid point
        logging.info(f"Berechne Inzidenzwinkel fuer {num_points}x{num_points} Pkt")

        valid_count = 0
        for row in range(num_points):
            for col in range(num_points):
                # Calculate LV95 coordinates
                # row 0 = top (north), row 9 = bottom (south)
                # col 0 = left (west), col 9 = right (east)
                # Flip for raster convention
                x = e_lv95 + col * grid_step
                y = n_lv95 + (num_points - 1 - row) * grid_step

                # Get slope and aspect from DEM using horayzon's slope_plane_meth
                slope, aspect = self.__calculate_slope_aspect(x, y)

                if slope is None or aspect is None:
                    points.append((x, y, np.nan))
                    continue

                # Calculate incidence angle
                theta = calculate_incidence_angle(slope, aspect, sun_elev, sun_az)
                grid[row, col] = theta
                points.append((x, y, theta))
                logging.debug(f"E={x} / N={y} / Theta {theta:0.2f}")
                valid_count += 1

        logging.info(f"  {valid_count} von {num_points * num_points} Pkt berechnet")

        # Create transform for output GeoTIFF
        # Origin is upper-left corner in raster convention
        upper_left_x = e_lv95
        upper_left_y = n_lv95 + grid_size

        out_transform = from_bounds(
            upper_left_x,  # west
            n_lv95,  # south
            e_lv95 + grid_size,  # east
            upper_left_y,  # north
            num_points,  # width
            num_points,  # height
        )
        return grid, out_transform, points

    def __checkinput(self):
        if os.path.isdir(self.__output_path):
            logging.debug(f"Outpath {self.__output_path} found")
        else:
            raise AttributeError(f"Outpath {self.__output_path} not found")

    def __calculate_slope_aspect(self, x, y):
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
        col = np.argmin(np.abs(self.__dom.x_full - x))
        row = np.argmin(np.abs(self.__dom.y_full - y))

        nrows, ncols = self.__dom.elev_full.shape

        # Need at least 2 pixel border for slope_plane_meth
        # (it removes 1 pixel border)
        if col < 2 or col >= ncols - 2 or row < 2 or row >= nrows - 2:
            return None, None

        # Extract 5x5 window around the point
        window_size = 5
        half = window_size// 2

        rows_idx = slice(row - half, row + half + 1)
        cols_idx = slice(col - half, col + half + 1)

        # Get coordinate arrays for the window
        x_window = self.__dom.x_full[cols_idx].astype(np.float32)
        y_window = self.__dom.y_full[rows_idx].astype(np.float32)

        x_2d, y_2d = np.meshgrid(x_window, y_window)
        elevation_window = self.__dom.elev_full[rows_idx, cols_idx].astype(np.float32)

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

    def write_csv(self, points, output_path):
        """Write points to CSV with columns X, Y, Winkel."""
        logging.info(f"Schreibe CSV: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("X,Y,Winkel\n")
            for x, y, angle in points:
                if np.isnan(angle):
                    f.write(f"{x:.2f},{y:.2f},\n")
                else:
                    f.write(f"{x:.2f},{y:.2f},{angle:.2f}\n")

        logging.info(f"  {len(points)} Punkte geschrieben")

    def write_geotiff(self, grid, transform, output_path):
        """Write grid to GeoTIFF with float32 data type."""
        logging.info(f"Schreibe GeoTIFF: {output_path}")

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=grid.shape[0],
            width=grid.shape[1],
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(2056),  # LV95
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(grid, 1)

        logging.info(f"  Groesse: {grid.shape[0]} x {grid.shape[1]} Pixel")
        logging.info("  Datentyp: float32")
        logging.info("  CRS: EPSG:2056 (LV95)")