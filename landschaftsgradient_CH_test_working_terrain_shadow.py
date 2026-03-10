"""Contains 4 classes: DOM, SonnenWinkel (used in main_shadow_check),
InzidenWinkel (used in main_incidence_angle) and ImgCheck (used in main_image_check)"""

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


class DOM:
    def __init__(self, dom):
        self.__dom = dom
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
        

class SonnenWinkel:

    def __init__(self, dom, planets, output_path):
        self.__dom = DOM(dom)
        self.__planets = planets
        self.__output_path = output_path
        self.__checkinput()

    
    def __checkinput(self):
        bsp_path = os.path.join(self.__planets["path"], self.__planets["bsp_file"])
        if os.path.isfile(bsp_path):
            logging.debug(f".bsp file {bsp_path} found")
        else:
            raise AttributeError(f".bsp file {bsp_path} not found")


    # Grid part 
    def calc_shadow_grid(self, e_lv95, dateoi, timeoi, n_lv95, grid_size, grid_step, search_dist):
        """
        Compute a grid of shadow values over a rectangular area and write to GeoTIFF.

        Shadow values:
            0 = BELEUCHTET (illuminated)
            1 = SELBSTBESCHATTET (self-shaded)
            2 = GELAENDEBESCHATTET (terrain-shaded)  [future extension]
            3 = NICHT BERUECKSICHTIGT (sun below horizon)
            225 = NODATA (invalid elevation)
        """
        logging.info("Berechne Shadow Grid...")
        dt_utc = HelperFunctions.parse_datetime(dateoi, timeoi)

        # compute bounds of grid
        xmin, xmax = e_lv95, e_lv95 + grid_size
        ymin, ymax = n_lv95, n_lv95 + grid_size

        # Read DEM tile using rasterio window
        src = self.__dom._src
        window = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
        window = window.round_offsets().round_lengths()

        elev_tile = src.read(1, window=window)
        nodata = self.__dom._nodata

        # Replace nodata with NaN
        if nodata is not None:
            elev_tile = np.where(elev_tile == nodata, np.nan, elev_tile)

        height, width = elev_tile.shape
        logging.info(f"Tile reading: {width} x {height} px")

        # Compute slope/aspect using horayzon
        x_window = self.__dom._x0 + np.arange(width) * self.__dom._dx
        y_window = self.__dom._y0 + np.arange(height) * self.__dom._dy
        x_full_2d, y_full_2d = np.meshgrid(x_window, y_window)

        vec_tilt = hray.topo_param.slope_plane_meth(
            x_full_2d.astype(np.float32),
            y_full_2d.astype(np.float32),
            np.nan_to_num(elev_tile, nan=0).astype(np.float32)
        )
        
        vec_tilt = vec_tilt[1:-1, 1:-1]
        elev_inner = elev_tile[1:-1, 1:-1]

        # Extract slope + aspect in DEM resolution
        slope = np.rad2deg(np.arccos(vec_tilt[..., 2]))
        aspect_vec = vec_tilt[..., :2]
        aspect = np.rad2deg(np.arctan2(aspect_vec[..., 0], aspect_vec[..., 1]))
        aspect[aspect < 0] += 360

        # Sun position from center of tile
        center_x = xmin + grid_size / 2
        center_y = ymin + grid_size / 2

        lon, lat = HelperFunctions.lv95_to_wgs84(center_x, center_y)
        sun_elevation, sun_azimuth = HelperFunctions.calc_sunpos(
            lat,
            lon,
            dt_utc,
            self.__planets["path"],
            self.__planets["bsp_file"],
        )

        # Incidence angle on all pixels of DEM tile
        theta = HelperFunctions.calculate_incidence_angle(
            slope, aspect, sun_elevation, sun_azimuth
        )

        # cast/terrain shadow
        # x must be ascending, y must be descending (north→south)
        x_inner = x_window[1:-1].astype(np.float32)
        y_inner = y_window[1:-1].astype(np.float32)

        # HORAYZON expects y descending (top = largest N value)
        if y_inner[0] < y_inner[-1]:
            y_inner = y_inner[::-1]
            elev_inner = elev_inner[::-1, :]
            slope     = slope[::-1, :]
            aspect    = aspect[::-1, :]
            theta     = theta[::-1, :]

        # Number of azimuth sectors for horizon search
        # 360 gives ~1° resolution; use 72 (5°) for speed
        num_sectors = 360

        logging.info("Computing HORAYZON horizon angles (this may take a moment)...")
        import inspect
        logging.info(f"horizon_gridded signature: {inspect.signature(hray.horizon.horizon_gridded)}")
        # hray.horizon.horizon_gridded returns shape (rows, cols, num_sectors)
        # Each value is the horizon elevation angle [radians] in that azimuth sector
        horizon_elev = hray.horizon.horizon_gridded(
            x_inner,
            y_inner,
            np.nan_to_num(elev_inner, nan=0.0).astype(np.float32),
            num_sectors,
        )   # shape: (rows, cols, num_sectors)

        # Map sun azimuth to sector index
        # HORAYZON sectors start at North (0°) and go clockwise
        sector_width = 360.0 / num_sectors
        sun_az_sector = int((sun_azimuth % 360.0) / sector_width)

        # Horizon elevation in sun direction [radians] → degrees
        horizon_in_sun_dir = np.rad2deg(horizon_elev[:, :, sun_az_sector])
        
        # Classify pixels
        # Prepare shadow grid (same resolution as vec_tilt)
        nodata_mask = np.isnan(elev_inner)
        shadow_grid = np.zeros(elev_inner.shape[:2], dtype=np.uint8)
        
        # -1 = nodata
        shadow_grid[nodata_mask] = 255

        # 1 = self-shaded 
        shadow_grid[theta >= 90] = 1

        # 2 = terrain / cast shadow (horizon blocks the sun)
        terrain_shadow = (sun_elevation < horizon_in_sun_dir) & (shadow_grid == 0)
        shadow_grid[terrain_shadow] = 2

        # 3 = sun below horizon
        if sun_elevation < 0:
            shadow_grid[~nodata_mask] = 3

        # Raster transform for output
        out_transform = rasterio.windows.transform(window, src.transform)

        logging.info("Shadow Grid berechnet.")

        return shadow_grid, out_transform
    
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
            dtype=np.uint8,
            crs=CRS.from_epsg(2056),  # LV95
            transform=transform,
            nodata=255,
        ) as dst:
            dst.write(grid, 1)

        logging.info(f"  Groesse: {grid.shape[0]} x {grid.shape[1]} Pixel")
        logging.info("  Datentyp: float32")
        logging.info("  CRS: EPSG:2056 (LV95)")

        self.__dom.close()


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
        """
        dt_utc = HelperFunctions.parse_datetime(dateoi, timeoi)
        num_points = grid_size // grid_step

        # compute bounds of grid
        xmin, xmax = e_lv95, e_lv95 + grid_size
        ymin, ymax = n_lv95, n_lv95 + grid_size

        # Initialize slope/aspect arrays
        slope = np.full((num_points, num_points), np.nan, dtype=np.float32)
        aspect = np.full((num_points, num_points), np.nan, dtype=np.float32)
        logging.info(f"{os.getpid()} Computing slope/aspect grid (vectorized)...")

        # Read DOM window once
        src = self.__dom._src
        window = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
        window = window.round_offsets().round_lengths()

        elev_tile = src.read(1, window=window)
        height, width = elev_tile.shape 
        nodata = self.__dom._nodata

        if nodata is not None:
            elev_tile = np.where(elev_tile == nodata, np.nan, elev_tile)

        height, width = elev_tile.shape

        # Compute slope/aspect using horayzon
        x_window = self.__dom._x0 + np.arange(int(width)) * self.__dom._dx
        y_window = self.__dom._y0 + np.arange(int(height)) * self.__dom._dy
        x_full_2d, y_full_2d = np.meshgrid(x_window, y_window)

        # # Read dom_ext
        # col_start = int((xmin - self.__dom._xmin) / self.__dom._dx)
        # col_end   = int((xmax - self.__dom._xmin) / self.__dom._dx)
        # row_start = int((self.__dom._ymax - ymax) / abs(self.__dom._dy))  # row index increases downward
        # row_end   = int((self.__dom._ymax - ymin) / abs(self.__dom._dy))

        # # Extract elevation tile
        # elev_tile = self.__dom._dom_ext[row_start:row_end, col_start:col_end]
        # nodata = self.__dom._nodata

        # if nodata is not None:
        #     elev_tile = np.where(elev_tile == nodata, np.nan, elev_tile)

        # height, width = elev_tile.shape

        # # Compute slope/aspect using horayzon
        # x_window = self.__dom._xmin + (col_start + np.arange(width) + 0.5) * self.__dom._dx
        # y_window = self.__dom._ymax + (-row_start - np.arange(height) - 0.5) * self.__dom._dy
        # x_full_2d, y_full_2d = np.meshgrid(x_window, y_window)

        vec_tilt = hray.topo_param.slope_plane_meth(
            x_full_2d.astype(np.float32),
            y_full_2d.astype(np.float32),
            np.nan_to_num(elev_tile).astype(np.float32)
        )

        slope_tile = np.rad2deg(np.arccos(vec_tilt[1:-1, 1:-1, 2]))
        aspect_vec = vec_tilt[1:-1, 1:-1, :2]
        aspect_tile = np.rad2deg(np.arctan2(aspect_vec[..., 0], aspect_vec[..., 1]))
        aspect_tile[aspect_tile < 0] += 360

        # Crop extra pixels if needed
        slope_tile = slope_tile[:num_points, :num_points]
        aspect_tile = aspect_tile[:num_points, :num_points]
        rows = min(num_points, slope_tile.shape[0])
        cols = min(num_points, slope_tile.shape[1])

        slope[:rows, :cols] = slope_tile[:rows, :cols]
        aspect[:rows, :cols] = aspect_tile[:rows, :cols]

        # Sun position once per tile 
        center_x = e_lv95 + grid_size / 2
        center_y = n_lv95 + grid_size / 2

        lon, lat = HelperFunctions.lv95_to_wgs84(center_x, center_y)
        sun_elev, sun_az = HelperFunctions.calc_sunpos(
            lat, lon, dt_utc,
            self.__planets["path"],
            self.__planets["bsp_file"]
        )

        logging.info(f"{os.getpid()} Computing incidence angle")
        # incidence calculation
        theta = HelperFunctions.calculate_incidence_angle(slope, aspect, sun_elev, sun_az)
        grid = theta.astype(np.float32)
        logging.info(f"{os.getpid()} Center{center_x}/{center_y}")

        # Build output transform based on actual read
        out_transform = rasterio.windows.transform(window, src.transform)

        # out_transform = Affine(
        #     self.__dom._dx, 0, xmin,
        #     0, self.__dom._dy, ymax
        # )
        logging.info(f"{os.getpid()} Transform of dom ext: {out_transform}")
        logging.info(f"{os.getpid()} Tile finished.")

        return grid, out_transform

    def __checkinput(self):
        if os.path.isdir(self.__output_path):
            logging.debug(f"{os.getpid()} Outpath {self.__output_path} found")
        else:
            raise AttributeError(f"{os.getpid()} Outpath {self.__output_path} not found")
        
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
    def close(self):
        self.__dom._src.close()


class ImgChecker2:

    def __init__(self, test_raster, cfg=None):
        self.__testraster = test_raster
        self.__cfg = cfg
        self.__reffilepattern = cfg["incident_ref_file_pattern"]
        fnarr = os.path.basename(test_raster).split("_")
        self.__datum = fnarr[3]
        self.__zeit = fnarr[4]
        self.__diff = None

    def comparer(self, base_path=None, reffilepattern=None, write_csv=True):
        if not base_path:
            if "refdata" in self.__cfg.keys():
                base_path = os.path.join(
                    self.__cfg["refdata"]["path"], self.__cfg["refdata"]["dom_source"]
                )
            else:
                raise ValueError("Path to reference files not defined")
        if not reffilepattern:
            if "incident_ref_file_pattern" in self.__cfg.keys():
                reffilepattern = self.__cfg["incident_ref_file_pattern"]
            else:
                raise ValueError("Path to reference files not defined")
        subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
        for subfolder in subfolders:
            algo = os.path.basename(subfolder)
            reftif_name = self.__reffilepattern.format(
                algo=algo, datum=self.__datum, zeit=self.__zeit
            )
            reftif = os.path.join(subfolder, reftif_name)
            if os.path.isfile(reftif):
                logging.info(f"File {reftif_name} found")
                self.__rasdiff(reftif)
                if write_csv:
                    prefix = os.path.basename(self.__testraster).rsplit(".", 1)[0]
                    self.__write_csv(f"{prefix}_{algo}.csv")
            else:
                logging.info(f"File {reftif_name} not found")

    def compare(self, reftif, write_csv=True):
        if os.path.isfile(reftif):
            logging.info(f"File {os.path.basename(reftif)} found")
            self.__rasdiff(reftif)
            if write_csv:
                filename = os.path.basename(self.__testraster).rsplit(".", 1)[0]
                self.__write_csv(f"{filename}.csv")
        else:
            logging.info(f"File {os.path.basename(reftif)} not found")

    def __rasdiff(self, refraster):
        # Les limites de raster A pour couper et avoir raster B 1km x 1km pixel
        testraster_data, testraster_bounds, testraster_crs = self.__read_testraster()
        with rasterio.open(refraster) as refraster_data:
            left, bottom, right, top = self.__get_bbox(
                testraster_crs, testraster_bounds, refraster_data
            )
            window_ref = window_from_bounds(
                left, bottom, right, top, refraster_data.transform
            )
            window_ref = window_ref.round_offsets().round_lengths()
            raster_ref = refraster_data.read(1, window=window_ref)
        if testraster_data.shape != raster_ref.shape:
            raise ValueError("Test raster and Reference raster shapes do not match")
        self.__diff = testraster_data - raster_ref
        stats = {
            "mean": f"{float(np.nanmean(self.__diff)):.1f}",
            "max": f"{float(np.nanmax(self.__diff)):.1f}",
            "min": f"{float(np.nanmin(self.__diff)):.1f}",
            "std": f"{float(np.nanstd(self.__diff)):.1f}",
        }
        logging.info(f"Stats for file - {os.path.basename(refraster)}: {stats}")

    def __write_csv(self, csv_filename):
        """
        Write counts of diff values in intervals to CSV.

        diff_array: numpy array of differences (raster_test - raster_ref)
        output_path: path to save the CSV
        """
        # Définir les intervalles symétriques de 5
        output_path = os.path.join(os.path.dirname(self.__testraster), csv_filename)
        max_range = int(np.nanmax(np.abs(self.__diff))) + 5
        bins = np.arange(0, max_range + 5, 5)
        counts = []
        for i in range(1, len(bins)):
            lower = bins[i - 1]
            upper = bins[i]
            # Les intervales comme [-upper, -lower[ U ]lower, upper]
            mask = ((self.__diff >= lower) & (self.__diff < upper)) | (
                (self.__diff <= -lower) & (self.__diff > -upper)
            )
            counts.append(np.sum(mask))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            headers = [
                f"[{-bins[i]}, {-bins[i-1]}] & [{bins[i-1]}, {bins[i]}]"
                for i in range(1, len(bins))
            ]
            writer.writerow(headers)
            writer.writerow(counts)

    def __get_bbox(self, testraster_crs, testraster_bounds, refraster_data):
        refdata_crs = refraster_data.crs
        if testraster_crs != refdata_crs:
            transformer = Transformer.from_crs(
                testraster_crs, refdata_crs, always_xy=True
            )
            left, bottom = transformer.transform(
                testraster_bounds.left, testraster_bounds.bottom
            )
            right, top = transformer.transform(
                testraster_bounds.right, testraster_bounds.top
            )
            return left, bottom, right, top
        else:
            return (
                testraster_bounds.left,
                testraster_bounds.bottom,
                testraster_bounds.right,
                testraster_bounds.top,
            )

    def __read_testraster(self):
        with rasterio.open(self.__testraster) as src_test:
            testraster_data = src_test.read(1)
            testraster_bounds = src_test.bounds
            testraster_crs = src_test.crs
        return testraster_data, testraster_bounds, testraster_crs


class ImgChecker:
    def __init__(self, test_raster_path, ref_raster_path, output_path):
        self.__test_raster_path = test_raster_path
        self.__ref_raster_path = ref_raster_path
        self.__output_path = output_path
        self.__checkinput()

        self.raster_test, self.__bounds_test, self.__crs_test = self.__get_extends()

    def compare(self):
        raster_ref = self.__cut_ref_raster()

        if raster_ref.shape != self.raster_test.shape:
            raise ValueError("Test raster and Reference raster shapes do not match")

        diff = self.raster_test - raster_ref

        stats = {
            "mean": float(np.nanmean(diff)),
            "max": float(np.nanmax(diff)),
            "min": float(np.nanmin(diff)),
            "std": float(np.nanstd(diff)),
        }

        logging.info(f"Stats for file - {self.__ref_raster_path}: {stats}")
        return diff

    def __checkinput(self):
        if os.path.isdir(self.__output_path):
            logging.debug(f"Outpath {self.__output_path} found")
        else:
            raise AttributeError(f"Outpath {self.__output_path} not found")

    def __get_extends(self):
        with rasterio.open(self.__test_raster_path) as src_test:
            raster_test = src_test.read(1)
            bounds_test = src_test.bounds
            crs_test = src_test.crs
            res_test = src_test.res

        logging.info("Test raster loaded")
        logging.info(f"Resolution of test raster: {res_test}")
        logging.info(f"Shape of test raster: {raster_test.shape}")
        return raster_test, bounds_test, crs_test

    def __cut_ref_raster(self):
        # Les limites de raster A pour couper et avoir raster B 1km x 1km pixel
        with rasterio.open(self.__ref_raster_path) as src_ref:
            crs_ref = src_ref.crs
            if self.__crs_test != crs_ref:
                transformer = Transformer.from_crs(
                    self.__crs_test, crs_ref, always_xy=True
                )
                left, bottom = transformer.transform(
                    self.__bounds_test.left, self.__bounds_test.bottom
                )
                right, top = transformer.transform(
                    self.__bounds_test.right, self.__bounds_test.top
                )
            else:
                left = self.__bounds_test.left
                bottom = self.__bounds_test.bottom
                right = self.__bounds_test.right
                top = self.__bounds_test.top

            window_ref = window_from_bounds(left, bottom, right, top, src_ref.transform)
            window_ref = window_ref.round_offsets().round_lengths()
            raster_ref = src_ref.read(1, window=window_ref)
            res_ref = src_ref.res

        logging.info("Reference raster loaded")
        logging.info(f"Resolution of reference raster: {res_ref}")
        logging.info(f"Shape of reference raster: {raster_ref.shape}")
        logging.info(f"Reference raster nodata value: {src_ref.nodata}")
        logging.info(
            f"Number of NaNs in reference raster : {np.isnan(raster_ref).sum()}"
        )
        logging.info(f"Number of non-finite values: {np.isfinite(raster_ref).sum()}")
        return raster_ref

    # .csv file pour que les valeurs de diff soit col1 = [-5°, 5°], col2 = [-10°; -5°] + [5°, 10°], col3 = [-15°, -10°] + [10°, 15°]...
    def write_csv(self, diff: np.ndarray, filename_csv: str):
        """
        Write counts of diff values in intervals to CSV.

        diff_array: numpy array of differences (raster_test - raster_ref)
        output_path: path to save the CSV
        """
        # Définir les intervalles symétriques de 5
        output_path = os.path.join(self.__output_path, filename_csv)
        max_range = int(np.nanmax(np.abs(diff))) + 5
        bins = np.arange(0, max_range + 5, 5)

        counts = []
        for i in range(1, len(bins)):
            lower = bins[i - 1]
            upper = bins[i]
            # Les intervales comme [-upper, -lower[ U ]lower, upper]
            mask = ((diff >= lower) & (diff < upper)) | (
                (diff <= -lower) & (diff > -upper)
            )
            counts.append(np.sum(mask))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            headers = [
                f"[{-bins[i]}, {-bins[i-1]}] & [{bins[i-1]}, {bins[i]}]"
                for i in range(1, len(bins))
            ]
            writer.writerow(headers)
            writer.writerow(counts)

        logging.info(f"CSV écrit : {output_path}, avec {len(counts)} colonnes")
