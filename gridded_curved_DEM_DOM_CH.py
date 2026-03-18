# Description: Compute gridded shadow map from DOM (LV95, 10 m) for a region
#              in Switzerland. Earth's surface curvature is considered.
#              Input DEM (LV95) is warped to WGS84 for computation; output
#              GeoTIFFs are reprojected back to LV95 at 10 m resolution.
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import datetime as dt
import time
from scipy import ndimage
from skyfield.api import load, wgs84
from osgeo import gdal, osr
import horayzon as hray

gdal.UseExceptions()


def fill_small_holes(arr, max_pixels):
    """Fill connected regions of 0s completely enclosed by 1s if size <= max_pixels."""
    labeled, num_features = ndimage.label(arr == 0)
    border_labels = set(np.unique(labeled[0, :])) | set(np.unique(labeled[-1, :])) \
                  | set(np.unique(labeled[:, 0])) | set(np.unique(labeled[:, -1]))
    border_labels.discard(0)
    result = arr.copy()
    for lbl in range(1, num_features + 1):
        if lbl not in border_labels and np.sum(labeled == lbl) <= max_pixels:
            result[labeled == lbl] = 1
    return result

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Inner domain boundaries in LV95 (EPSG:2056) [m]
domain_lv95 = {"x_min": 2604000, "x_max": 2624000,
               "y_min": 1160000, "y_max": 1180000}
dist_search = 20.0   # search distance for terrain shading [kilometre]
ellps = "WGS84"      # Earth's surface approximation
dem_res_out = 10.0   # output resolution in LV95 [m]
max_hole_pixels = 1  # max size of 0-holes surrounded by 1s to fill per layer

# Paths and file names
file_dem = r"D:\temp\DOM\DOM_full_CH_nodata.tif"
path_out = r"D:\temp\shadow_check\swissalti3d" + "\\"
path_bsp = r"D:\temp\planets\\"

# Time point for shadow computation
time_dt = dt.datetime(2023, 12, 25, 10, 34, 41, tzinfo=dt.timezone.utc)
# time_dt = dt.datetime(2025, 6, 17, 10, 26, 21, tzinfo=dt.timezone.utc)

# -----------------------------------------------------------------------------
# Reproject DEM from LV95 to WGS84 (in memory)
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    raise FileNotFoundError("Output directory does not exist")

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
domain = {"lon_min": min(c[0] for c in corners_wgs84),
          "lon_max": max(c[0] for c in corners_wgs84),
          "lat_min": min(c[1] for c in corners_wgs84),
          "lat_max": max(c[1] for c in corners_wgs84)}

# Compute outer domain including search buffer in WGS84
domain_outer = hray.domain.curved_grid(domain, dist_search, ellps)

# Target resolution in WGS84 degrees (~10 m at lat 47 N)
dem_res_deg = 10.0 / 111320.0  # 1 degree latitude ~ 111.32 km

# Warp DOM from LV95 to WGS84 at outer domain extent (in memory)
ds_src = gdal.Open(file_dem)
nodata_src = ds_src.GetRasterBand(1).GetNoDataValue()
ds_src = None

ds_warp = gdal.Warp(
    "", file_dem,
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
print("DEM reprojected to WGS84")
print("Size of loaded DEM domain: " + str(elevation.shape))
print("Elevation range of DEM: %.1f" % elevation.min()
      + " - %.1f" % elevation.max() + " m")

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class  (identical to curved script)
# -----------------------------------------------------------------------------

# Compute indices of inner domain
slice_in = (slice(np.where(lat >= domain["lat_max"])[0][-1],
                  np.where(lat <= domain["lat_min"])[0][0] + 1),
            slice(np.where(lon <= domain["lon_min"])[0][-1],
                  np.where(lon >= domain["lon_max"])[0][0] + 1))
offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))
elevation_ortho = np.ascontiguousarray(elevation[slice_in])

# Compute ellipsoidal heights
elevation += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                    elevation, ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape

# Compute ENU coordinates
trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[int(len(lon) / 2)], lat_or=lat[int(len(lat) / 2)], ellps=ellps)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Compute unit vectors for inner domain
vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon[slice_in[1]],
                                                      lat[slice_in[0]]))
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
print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
      + ", %.3f" % surf_enl_fac.max())

# Initialise terrain
mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
terrain = hray.shadow.Terrain()
dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                   offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                   surf_enl_fac, mask=mask, elevation=elevation_ortho,
                   refrac_cor=True)

# Load Skyfield data
load.directory = path_bsp
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)

# -----------------------------------------------------------------------------
# Compute shadow map
# -----------------------------------------------------------------------------

t_beg = time.time()
ts = load.timescale()
t = ts.from_datetime(time_dt)
astrometric = loc_or.at(t).observe(sun)
alt, az, d = astrometric.apparent().altaz()
x_s = d.m * np.cos(alt.radians) * np.sin(az.radians)
y_s = d.m * np.cos(alt.radians) * np.cos(az.radians)
z_s = d.m * np.sin(alt.radians)
sun_position = np.array([x_s, y_s, z_s], dtype=np.float32)
print(f"Sun altitude: {alt.degrees:.2f}°, azimuth: {az.degrees:.2f}°")

shadow_buffer = np.zeros(vec_tilt_enu.shape[:2], dtype=np.uint8)
terrain.shadow(sun_position, shadow_buffer)
print("Shadow computation time: %.2f s" % (time.time() - t_beg))

# -----------------------------------------------------------------------------
# Write output GeoTIFFs in LV95 at 10 m
# -----------------------------------------------------------------------------

# Build WGS84 geotransform for the inner domain
lon_in = lon[slice_in[1]]
lat_in = lat[slice_in[0]]  # decreasing
lon_res = float(lon_in[1] - lon_in[0])
lat_res = float(lat_in[1] - lat_in[0])  # negative
gt_inner = (float(lon_in[0]) - lon_res / 2, lon_res, 0,
            float(lat_in[0]) - lat_res / 2, 0, lat_res)

ts_str = time_dt.strftime("%Y%m%d_%H%M%S")
tiff_layers = {
    "0_illuminated":    (shadow_buffer == 0).astype(np.uint8),
    "1_self_shaded":    (shadow_buffer == 1).astype(np.uint8),
    "2_terrain_shaded": (shadow_buffer == 2).astype(np.uint8),
    "3_not_considered": (shadow_buffer == 3).astype(np.uint8),
}
for name, data in tiff_layers.items():
    # Write shadow layer as WGS84 in-memory raster
    driver_mem = gdal.GetDriverByName("MEM")
    ds_mem = driver_mem.Create("", data.shape[1], data.shape[0],
                                1, gdal.GDT_Byte)
    ds_mem.SetGeoTransform(gt_inner)
    ds_mem.SetProjection(srs_wgs84.ExportToWkt())
    ds_mem.GetRasterBand(1).WriteArray(data)

    # Warp to LV95 at 10 m, clipped to requested domain (in memory)
    ds_lv95 = gdal.Warp(
        "", ds_mem,
        format="MEM",
        dstSRS="EPSG:2056",
        outputBounds=(domain_lv95["x_min"], domain_lv95["y_min"],
                      domain_lv95["x_max"], domain_lv95["y_max"]),
        xRes=dem_res_out, yRes=dem_res_out,
        resampleAlg=gdal.GRA_NearestNeighbour,
    )
    ds_mem = None

    # Fill small enclosed holes (0-pixels surrounded by 1-pixels)
    data_lv95 = ds_lv95.GetRasterBand(1).ReadAsArray()
    data_lv95 = fill_small_holes(data_lv95, max_hole_pixels)

    # Write filled raster to GTiff
    driver_tif = gdal.GetDriverByName("GTiff")
    ds_out = driver_tif.Create(
        path_out + f"{ts_str}_shadow_{name}.tif",
        ds_lv95.RasterXSize, ds_lv95.RasterYSize, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(ds_lv95.GetGeoTransform())
    ds_out.SetProjection(ds_lv95.GetProjection())
    ds_out.GetRasterBand(1).WriteArray(data_lv95)
    ds_out.FlushCache()
    ds_out = None
    ds_lv95 = None

print("Shadow GeoTIFFs written to", path_out)
