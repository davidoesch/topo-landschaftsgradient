"""Contains 4 classes: HelperFunctions, DOM, SonnenWinkel (used for Shadow Check), and
InzidenWinkel (used to get incidence angle)"""

import logging
import os
import rasterio
import csv
import numpy as np
import horayzon as hray
from rasterio.windows import Window
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from affine import Affine
from pyproj import CRS, Transformer
from datetime import datetime, timezone
from skyfield.api import load, wgs84


class HelperFunctions:
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

    def calc_sunpos(lat_loc, lon_loc, dt_utc, planets_path, planets_file):
        logging.info(f"{os.getpid()} Berechne Sonnenposition...")
        load.directory = planets_path
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


class DOM_iw:
    def __init__(self, dom_iw):
        self.__dom_iw = dom_iw
        self.__checkinput()
        self.__loadingDOM()

    def __checkinput(self):
        if os.path.isfile(self.__dom):
            logging.debug(f"DOM {self.__dom} found")
        else:
            raise AttributeError(f"DOM {self.__dom} not found")

    def __loadingDOM(self):
        logging.info(f"{os.getpid()} Lade DOM-Daten...")

        # keep DOM open 
        self._src = rasterio.open(self.__dom)
        self._src_path = self.__dom

        self._transform = self._src.transform
        self._width = self._src.width
        self._height = self._src.height

        self._nodata = self._src.nodata
        self._dx = self._transform.a
        self._dy = self._transform.e

        self._x0 = self._transform.c + self._dx / 2
        self._y0 = self._transform.f + self._dy / 2

        logging.info(f"{os.getpid()} Loaded DOM metadata: {self._width} × {self._height} px")
        
    def close(self):
        if self._src:
            self._src.close()
            logging.debug(f"{os.getpid()} DOM dataset closed.")

