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
from rasterio.transform import Affine
from osgeo import gdal, osr
from pyproj import CRS, Transformer
from datetime import datetime, timezone
from skyfield.api import load, wgs84
import time



class HelperFunctions:
    def lv95_to_wgs84(e_lv95, n_lv95):
        """Convert LV95 (EPSG:2056) to WGS84 (EPSG:4326) coordinates."""
        crs_lv95 = CRS.from_epsg(2056)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_lv95, crs_wgs84, always_xy=True)
        lon, lat = transformer.transform(e_lv95, n_lv95)
        return lon, lat

    def wgs84_to_lv95(lon, lat):
        """Convert WGS84 (EPSG:4326) to LV95 (EPSG:2056) coordinates."""
        crs_wgs84 = CRS.from_epsg(4326)
        crs_lv95 = CRS.from_epsg(2056)
        transformer = Transformer.from_crs(crs_wgs84, crs_lv95, always_xy=True)
        e, n = transformer.transform(lon, lat)
        return e, n

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


class DOM_sw:
    def __init__(self, dom, search_dist, output_path):
        self.__dom = dom
        self.__search_dist = search_dist
        self.__output_path = output_path
        self.__checkinput()

    def __checkinput(self):
        if os.path.isfile(self.__dom):
            logging.debug(f"DOM {self.__dom} found")
        else:
            raise AttributeError(f"DOM {self.__dom} not found")

    def reproject_dom(self, e_lv95, n_lv95, grid_size, grid_step, search_dist):
        domain_lv95 = {
            "x_min": e_lv95,
            "x_max": e_lv95 + grid_size,
            "y_min": n_lv95,
            "y_max": n_lv95 + grid_size
        }
        ellps = "WGS84"

        # Set up coordinate transformations
        srs_lv95 = osr.SpatialReference()
        srs_lv95.ImportFromEPSG(2056)
        srs_lv95.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs_wgs84 = osr.SpatialReference()
        srs_wgs84.ImportFromEPSG(4326)
        srs_wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(srs_lv95, srs_wgs84)

        # Convert inner domain corners LV95 -> WGS84 to get domain in degrees
        corners_lv95 = [(domain_lv95["x_min"], domain_lv95["y_min"]),
                        (domain_lv95["x_max"], domain_lv95["y_min"]),
                        (domain_lv95["x_max"], domain_lv95["y_max"]),
                        (domain_lv95["x_min"], domain_lv95["y_max"])]
        corners_wgs84 = [ct.TransformPoint(px, py) for px, py in corners_lv95]
        domain = {
                "lon_min": min(c[0] for c in corners_wgs84),
                "lon_max": max(c[0] for c in corners_wgs84),
                "lat_min": min(c[1] for c in corners_wgs84),
                "lat_max": max(c[1] for c in corners_wgs84)
        }

        # Clamp domain to valid geographic range (important for horayzon)
        domain["lat_min"] = max(-89.9, domain["lat_min"])
        domain["lat_max"] = min(89.9, domain["lat_max"])
        domain["lon_min"] = max(-179.9, domain["lon_min"])
        domain["lon_max"] = min(179.9, domain["lon_max"])
        logging.warning(f"DOMAIN (inner): {domain}")

        # Compute outer domain including search buffer in WGS84
        #domain_outer = hray.domain.curved_grid(domain, search_dist, ellps)
        try:
            domain_outer = hray.domain.curved_grid(domain, search_dist, ellps)

            lon_span = domain_outer["lon_max"] - domain_outer["lon_min"]
            lat_span = domain_outer["lat_max"] - domain_outer["lat_min"]
            if lon_span > 1 or lat_span > 1:
                raise ValueError("curved_grid produced unrealistic domain")

        except Exception:
            logging.warning(f"{os.getpid()} curved_grid failed → using SAFE fallback")

            buffer_deg = search_dist / 111320.0  # meters → degrees

            domain_outer = {
                "lon_min": domain["lon_min"] - buffer_deg,
                "lon_max": domain["lon_max"] + buffer_deg,
                "lat_min": domain["lat_min"] - buffer_deg,
                "lat_max": domain["lat_max"] + buffer_deg,
            }
        logging.warning(f"DOMAIN OUTER: {domain_outer}")

        # Target resolution in WGS84 degrees (~10 m at lat 47 N)
        dem_res_deg = grid_step / 111320.0  # 1 degree latitude ~ 111.32 km

        # Warp DOM from LV95 to WGS84 at outer domain extent (in memory)
        ds_src = gdal.Open(self.__dom)
        nodata_src = ds_src.GetRasterBand(1).GetNoDataValue()
        ds_src = None

        ds_warp = gdal.Warp(
            "", self.__dom,
            format="MEM",
            dstSRS="EPSG:4326",
            outputBounds=(domain_outer["lon_min"], domain_outer["lat_min"],
                        domain_outer["lon_max"], domain_outer["lat_max"]),
            xRes=dem_res_deg, yRes=dem_res_deg,
            resampleAlg=gdal.GRA_Bilinear,
            srcNodata=nodata_src,
            dstNodata=-9999.0,
        )
        elevation = ds_warp.GetRasterBand(1).ReadAsArray().astype(np.float32)
        gt = ds_warp.GetGeoTransform()
        nx, ny = ds_warp.RasterXSize, ds_warp.RasterYSize
        ds_warp = None

        lon = np.linspace(gt[0] + gt[1] / 2.0, gt[0] + gt[1] * (nx - 0.5), nx)
        lat = np.linspace(gt[3] + gt[5] / 2.0, gt[3] + gt[5] * (ny - 0.5), ny)
        # lat is decreasing (north to south, gt[5] < 0)

        elevation[elevation == -9999.0] = 0.0
        logging.info("Ilu: DOM reprojected to WGS84")
        logging.info("Ilu: Size of loaded DOM domain: " + str(elevation.shape))
        logging.info("Ilu: Elevation range of DOM: %.1f" % elevation.min()
            + " - %.1f" % elevation.max() + " m")

        return elevation, lon, lat, domain, domain_lv95, srs_wgs84


class SonnenWinkel:

    gdal.UseExceptions()

    def __init__(self, dom, planets, search_dist, output_path):
        self.__dom = DOM_sw(dom, search_dist, output_path)
        self.__planets = planets
        self.__search_dist = search_dist
        self.__output_path = output_path
        self.__checkinput_dom()
        self.__checkinput()

    def __checkinput_dom(self):
        bsp_path = os.path.join(self.__planets["path"], self.__planets["bsp_file"])
        if os.path.isfile(bsp_path):
            logging.debug(f".bsp file {bsp_path} found")
        else:
            raise AttributeError(f".bsp file {bsp_path} not found")

    def __checkinput(self):
        if os.path.isdir(self.__output_path):
            logging.debug(f"{os.getpid()} Outpath {self.__output_path} found")
        else:
            raise AttributeError(f"{os.getpid()} Outpath {self.__output_path} not found")

    def calc_illuminate_grid(self, e_lv95, n_lv95, grid_size, grid_step, timeoi, dateoi):
        (
            self.__elevation,
            self.__lon,
            self.__lat,
            self.__domain,
            self.__domain_lv95,
            self.__srs_wgs84,
         ) = self.__dom.reproject_dom(e_lv95, n_lv95, grid_size, grid_step, self.__search_dist)

        ellps = "WGS84"
        # Compute indices of inner domain
        slice_in = (slice(np.where(self.__lat >= self.__domain["lat_max"])[0][-1],
                        np.where(self.__lat <= self.__domain["lat_min"])[0][0] + 1),
                    slice(np.where(self.__lon <= self.__domain["lon_min"])[0][-1],
                        np.where(self.__lon >= self.__domain["lon_max"])[0][0] + 1))
        offset_0 = slice_in[0].start
        offset_1 = slice_in[1].start
        logging.info(f"{os.getpid()} Inner domain size: " + str(self.__elevation[slice_in].shape))
        elevation_ortho = np.ascontiguousarray(self.__elevation[slice_in])

        # Compute ellipsoidal heights
        self.__elevation += hray.geoid.undulation(self.__lon, self.__lat, geoid="EGM96", path_to_aux_data=r"D:\temp\github\topo-landschaftsgradient\EGM/")  # [m]

        # Compute ECEF coordinates
        x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(self.__lon, self.__lat),
                                                            self.__elevation, ellps=ellps)
        dem_dim_0, dem_dim_1 = self.__elevation.shape

        # Compute ENU coordinates
        trans_ecef2enu = hray.transform.TransformerEcef2enu(
            lon_or=self.__lon[int(len(self.__lon) / 2)],
            lat_or=self.__lat[int(len(self.__lat) / 2)],
            ellps=ellps)
        x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                                    trans_ecef2enu)

        # Compute unit vectors for inner domain
        vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(self.__lon[slice_in[1]],
                                                            self.__lat[slice_in[0]]))
        vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in],
                                                z_ecef[slice_in], vec_norm_ecef,
                                                ellps=ellps)
        del x_ecef, y_ecef, z_ecef
        vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
        vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
        del vec_norm_ecef, vec_north_ecef

        # Merge vertex coordinates and pad geometry buffer
        vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

        # Compute rotation matrix (global ENU -> local ENU)
        rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                                vec_norm_enu)
        del vec_north_enu

        # Compute slope
        slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
                    slice(slice_in[1].start - 1, slice_in[1].stop + 1))
        vec_tilt_enu = np.ascontiguousarray(hray.topo_param.slope_plane_meth(
            x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
            rot_mat=rot_mat_glob2loc, output_rot=False)[1:-1, 1:-1])

        # Compute surface enlargement factor
        surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
        logging.info(f"{os.getpid()} Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
            + ", %.3f" % surf_enl_fac.max())

        # Initialise terrain
        mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
        terrain = hray.shadow.Terrain()
        #dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
        terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                        offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                        surf_enl_fac, mask=mask, elevation=elevation_ortho,
                        refrac_cor=True)

        # Load Skyfield data
        load.directory = self.__planets["path"]
        planets = load(self.__planets["bsp_file"])
        sun = planets["sun"]
        earth = planets["earth"]
        loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)

        # -----------------------------------------------------------------------------
        # Compute shadow map
        # -----------------------------------------------------------------------------

        t_beg = time.time()
        ts = load.timescale()
        dt_utc = HelperFunctions.parse_datetime(dateoi, timeoi)
        t = ts.from_datetime(dt_utc)
        astrometric = loc_or.at(t).observe(sun)
        alt, az, d = astrometric.apparent().altaz()
        x_s = d.m * np.cos(alt.radians) * np.sin(az.radians)
        y_s = d.m * np.cos(alt.radians) * np.cos(az.radians)
        z_s = d.m * np.sin(alt.radians)
        sun_position = np.array([x_s, y_s, z_s], dtype=np.float32)
        logging.info(f"{os.getpid()} Sun altitude: {alt.degrees:.2f}°, azimuth: {az.degrees:.2f}°")

        # buffer shadow
        shadow_buffer = np.zeros(vec_tilt_enu.shape[:2], dtype=np.uint8)
        terrain.shadow(sun_position, shadow_buffer)
        logging.info(f"{os.getpid()} Shadow computation time: %.2f s" % (time.time() - t_beg))

        illuminated = (shadow_buffer == 0).astype(np.uint8)

        # lv95_transform = transform_from_bounds(
        #     self.__domain_lv95["x_min"],
        #     self.__domain_lv95["y_min"],
        #     self.__domain_lv95["x_max"],
        #     self.__domain_lv95["y_max"],
        #     illuminated.shape[1],
        #     illuminated.shape[0],
        # )
        # return illuminated, lv95_transform

        # reprojection WGS84 → LV95
        lon_in  = self.__lon[slice_in[1]]
        lat_in  = self.__lat[slice_in[0]]   # décroissant (N→S)
        lon_res = float(lon_in[1]  - lon_in[0])
        lat_res = float(lat_in[1]  - lat_in[0])

        gt_inner = (
        float(lon_in[0]) - lon_res / 2,  lon_res, 0,
        float(lat_in[0]) - lat_res / 2,  0,       lat_res,
        )

        # Data in memory : WGS84
        NODATA_ILU = 255
        driver_mem = gdal.GetDriverByName("MEM")
        ds_mem = driver_mem.Create("", illuminated.shape[1], illuminated.shape[0],
                                    1, gdal.GDT_Byte)
        ds_mem.SetGeoTransform(gt_inner)
        ds_mem.SetProjection(self.__srs_wgs84.ExportToWkt())
        ds_mem.GetRasterBand(1).WriteArray(illuminated)

        band_mem = ds_mem.GetRasterBand(1)
        band_mem.SetNoDataValue(NODATA_ILU)   # ← déclarer nodata sur la source
        band_mem.WriteArray(illuminated)

        # clip on wanted domain
        ds_lv95 = gdal.Warp(
        "", ds_mem,
        format="MEM",
        dstSRS="EPSG:2056",
        outputBounds=(
            self.__domain_lv95["x_min"], self.__domain_lv95["y_min"],
            self.__domain_lv95["x_max"], self.__domain_lv95["y_max"],
        ),
        xRes=grid_step, yRes=grid_step,
        resampleAlg=gdal.GRA_NearestNeighbour,
        srcNodata=NODATA_ILU,
        dstNodata=NODATA_ILU,  # nodata = 255
        )
        ds_mem = None

        illuminated_lv95 = ds_lv95.GetRasterBand(1).ReadAsArray()
        transform_mem = ds_lv95.GetGeoTransform()
        ds_lv95 = None

        transform_lv95 = Affine(
            transform_mem[1], transform_mem[2], transform_mem[0],
            transform_mem[4], transform_mem[5], transform_mem[3],
        )
        ds_lv95 = None

        return illuminated_lv95, transform_lv95, NODATA_ILU

    def close(self):
        pass


class DOM_iw:
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


class InzidenWinkel:

    def __init__(self, dom, planets, output_path):
        self.__dom = DOM_iw(dom)
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
        NODATA_INC = np.float32(-9999.0)
        theta = HelperFunctions.calculate_incidence_angle(slope, aspect, sun_elev, sun_az)
        grid = theta.astype(np.float32)
        grid[np.isnan(grid)] = NODATA_INC  # nodata = -9999
        logging.info(f"{os.getpid()} Center{center_x}/{center_y}")

        # Build output transform based on actual read
        #out_transform = rasterio.windows.transform(window, src.transform)

        transform_out = Affine(
            self.__dom._dx, 0, xmin,
            0, self.__dom._dy, ymax
        )

        grid_out = np.full((num_points, num_points), NODATA_INC, dtype=np.float32)
        rows = min(num_points, grid.shape[0])
        cols = min(num_points, grid.shape[1])
        grid_out[:rows, :cols] = grid[:rows, :cols]
        logging.info(f"{os.getpid()} Transform of dom ext: {transform_out}")
        logging.info(f"{os.getpid()} Tile finished.")

        return grid_out, transform_out, NODATA_INC

    def __checkinput(self):
        if os.path.isdir(self.__output_path):
            logging.debug(f"{os.getpid()} Outpath {self.__output_path} found")
        else:
            raise AttributeError(f"{os.getpid()} Outpath {self.__output_path} not found")

    def close(self):
        self.__dom._src.close()
