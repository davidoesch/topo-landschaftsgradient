"""Analyses SAP Excelexports"""

import datetime
import logging
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.merge import merge
from iluina_module import HelperFunctions, InzidenWinkel, SonnenWinkel
from multiprocessing import Pool
import multiprocessing as mp
import math
import fiona
from pyproj import CRS as PyprojCRS, Transformer



LOGLEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_loglevel(level_str: str) -> int:
    return getattr(logging, level_str.strip().upper())


def parse_args():
    parser = ArgumentParser(description="Add arguments of incidence_angle_grid")
    parser.add_argument(
        "--date",
        "-d",
        default="25.12.2023",
        type=str,
        help="Date (DD.MM.YYYY), default: 13.12.2025",
    )
    parser.add_argument(
        "--time",
        "-t",
        default="10:34:41",
        type=str,
        help="Time UTC (HH:MM:SS), default: 10:00:00 (till 11:02:00 every 2 minutes)",
    )
    parser.add_argument(
        "--east",
        "-e",
        default=2480000,  # Niesen 2604000
        type=float,
        help="Easting in LV95 [m], default: 2480000 (start of CH grid)",
    )
    parser.add_argument(
        "--north",
        "-n",
        default=1060000,  # Niesen 1160000
        type=float,
        help="Northing in LV95 [m], default: 1060000 (start of CH grid)",
    )
    parser.add_argument(
        "--grid_size",
        "-gs",
        default=20000,
        type=int,
        help="meters",
    )
    parser.add_argument(
        "--grid_step",
        "-gstp",
        default=10,
        type=int,
        help="meters (100m / 10m = 10 points per axis = 100 total) = resolution",
    )
    parser.add_argument(
        "--loglevel",
        # type=parse_loglevel,
        choices=LOGLEVELS,
        default="INFO",
        help=f"Logelvel, possible values {LOGLEVELS}, default: INFO ",
    )

    parser.add_argument(
    "--perimeter",
    "-p",
    default="65",
    choices=["CH", "108", "22", "65", "8"],
    help="Perimeter to process: 'CH' for full Switzerland, or orbit ID (108, 22, 65, 8)",
    )

    return vars(parser.parse_args())


def setup_logging(
    level=logging.INFO,
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    logfolder: Path = None,
):
    loghandlers = [logging.StreamHandler()]
    logfile = ""
    if logfolder:
        logfolder.mkdir(parents=True, exist_ok=True)
        log_file = "{}_{}.log".format(
            os.path.splitext(os.path.basename(__file__))[0],
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        logfile = os.path.join(logfolder, log_file)
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(logging.Formatter(fmt))
        loghandlers.append(file_handler)
    logging.basicConfig(level=level, format=fmt, handlers=loghandlers)
    # Suppress verbose output from third-party libraries
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("pyproj").setLevel(logging.WARNING)
    logging.getLogger("skyfield").setLevel(logging.WARNING)


def log_listener_process(log_queue: mp.Queue, log_path: Path, loglvl=logging.INFO) -> None:
    setup_logging(level=loglvl, logfolder=log_path)

    while True:
        record = log_queue.get()
        if record is None:  # Sentinel
            break
        logging.getLogger(record.name).handle(record)


def get_config():
    with open(
        os.path.join(os.path.dirname(__file__), "config", "iluina.json"), "r"
    ) as file:
        data = json.load(file)
    return data


def logparameter(args, cfg, coord_tuple):
    tile_e, tile_n = coord_tuple
    lon_loc, lat_loc = HelperFunctions.lv95_to_wgs84(args["east"], args["north"])
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
    datum = dt_utc.strftime('%d.%m.%Y %H:%M:%S')
    east_lv95 = args["east"]
    north_lv95 = args["north"]
    grid_size = args["grid_size"]
    grid_step = args["grid_step"]
    dom_path = cfg["dom_path"]

    logging.info("=" * 60)
    logging.info(f"{os.getpid()} INZIDENZWINKEL-RASTER BERECHNUNG")
    logging.info(f"{os.getpid()}" + "=" * 60)
    logging.info(f"{os.getpid()} Datum/Zeit (UTC): {datum}")
    logging.info(f"{os.getpid()} Tile LV95:        E={tile_e:.2f} / N={tile_n:.2f}")
    logging.info(f"{os.getpid()} Tile WGS84:       Lon={lon_loc:.5f} / Lat={lat_loc:.5f}")
    logging.info(f"{os.getpid()} Standort LV95:    E={east_lv95:.2f} / N={north_lv95:.2f}")
    logging.info(f"{os.getpid()} Standort WGS84:   Lon={lon_loc:.5f} / Lat={lat_loc:.5f}")
    logging.info(
        f"{os.getpid()} Grid:       Grid size={grid_size:.2f} m / Grid step={grid_step:.2f} m"
    )
    logging.info(f"{os.getpid()} Rastergroesse: {grid_size:.2f}m x {grid_step:.2f}m")
    logging.info(f"{os.getpid()} DOM-Datei:     {dom_path}")
    logging.info(f"{os.getpid()}" + "=" * 60)


def load_perimeter_bbox(gpkg_path):

    with fiona.open(gpkg_path, layer=0) as src:
        bounds = src.bounds  # (minx, miny, maxx, maxy)
        
        # Handle both old fiona (dict CRS) and new fiona (CRS object)
        crs_obj = src.crs
        try:
            # New fiona: CRS object — convert via pyproj
            epsg = PyprojCRS.from_user_input(crs_obj).to_epsg()
            src_epsg = str(epsg) if epsg else "4326"
        except Exception:
            # Old fiona: dict with "init" key
            raw = crs_obj.get("init") if isinstance(crs_obj, dict) else ""
            src_epsg = raw.lower().replace("epsg:", "").strip() if raw else "4326"

    if src_epsg != "2056":
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:2056", always_xy=True)
        e_min, n_min = transformer.transform(bounds[0], bounds[1])
        e_max, n_max = transformer.transform(bounds[2], bounds[3])
    else:
        e_min, n_min, e_max, n_max = bounds

    logging.info(f"Perimeter bbox LV95: E={e_min:.0f}–{e_max:.0f}, N={n_min:.0f}–{n_max:.0f}")
    return e_min, n_min, e_max, n_max
def run(args, cfg, coord_tuple):
    logger = logging.getLogger()
    logger.info(f"Start processing data...{os.getpid()}")
    logparameter(args, cfg, coord_tuple)
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
    try:
        # initialize iw class with static config parameters
        iw = InzidenWinkel(
            dom=cfg["dom_path"],
            planets=cfg["planets"],
            output_path=cfg["output_path_IG"],
        )

        inc_bands = []
        inc_transform_ref = None
        inc_shape_ref = None
        # calculate incidence for every 10x10m cell of 20x20km tile
        inc_tile, inc_transform = iw.calc_incidence_grid(
            e_lv95=coord_tuple[0],
            n_lv95=coord_tuple[1],
            dateoi=args["date"],
            timeoi=args["time"], 
            grid_size=args["grid_size"],
            grid_step=args["grid_step"],
        )

        if inc_transform_ref is None:
            inc_transform_ref = inc_transform
            inc_shape_ref = inc_tile.shape
        else:
            if inc_transform != inc_transform_ref:
                logging.warning("Inc : Transform changed across time slices; using the first transform.")
            if inc_tile.shape != inc_shape_ref:
                raise ValueError(
                    f"Inc : Inconsistent tile shapes across time slices: got {inc_tile.shape}, expected {inc_shape_ref}"
                )

        inc_bands.append(inc_tile)

        if not inc_bands:
            raise RuntimeError("No bands were created for this tile")
        
        inc_stack = np.stack(inc_bands, axis=0)  # [time, H, W]
        inc_crs_value = getattr(iw, "crs", None) or "EPSG:2056"


        # Log statistics
        inc_valid_values = inc_tile[~np.isnan(inc_tile)]
        if len(inc_valid_values) > 0:
            logging.info(f"{os.getpid()} Statistik:")
            logging.info(f"  {os.getpid()} Min Inzidenzwinkel: {np.min(inc_valid_values):.2f} Grad")
            logging.info(f"  {os.getpid()} Max Inzidenzwinkel: {np.max(inc_valid_values):.2f} Grad")
            logging.info(f"  {os.getpid()} Mittelwert: {np.mean(inc_valid_values):.2f} Grad")
        else:
            logging.warning(f"{os.getpid()} Inc: Keine gueltigen Werte berechnet!")
    
        # initialize sw class with static config parameters
        sw = SonnenWinkel(
            dom=cfg["dom_path"],
            planets=cfg["planets"],
            search_dist=cfg["search_dist"],
            output_path=cfg["output_path_SW"],
        )
        
        ilu_bands = []
        ilu_transform_ref = None
        ilu_shape_ref = None

        #for t_str in time_strings:
        # calculate illuminate for every 10x10m cell of 20x20km tile
        ilu_tile, ilu_transform = sw.calc_illuminate_grid(
            e_lv95=coord_tuple[0],
            n_lv95=coord_tuple[1],
            dateoi=args["date"],
            timeoi=args["time"],
            grid_size=args["grid_size"],
            grid_step=args["grid_step"],
        )

        if ilu_transform_ref is None:
            ilu_transform_ref = ilu_transform
            ilu_shape_ref = ilu_tile.shape
        else:
            if ilu_transform_ref != ilu_transform_ref:
                logging.warning("Transform changed across time slices; using the first transform.")
            if ilu_tile.shape != ilu_shape_ref:
                raise ValueError(
                    f"Inconsistent tile shapes across time slices: got {ilu_tile.shape}, expected {ilu_shape_ref}"
                )

        ilu_bands.append(ilu_tile)

        if not ilu_bands:
            raise RuntimeError("No bands were created for this tile")
        
        ilu_stack = np.stack(ilu_bands, axis=0)  # [time, H, W]
        ilu_crs_value = getattr(sw, "crs", None) or "EPSG:2056"
        
        # Log statistics
        total_pixels = ilu_tile.size
        illuminated_pixels = int(np.sum(ilu_tile == 1))  # ← count the 1s
        pct = 100.0 * illuminated_pixels / total_pixels
        logging.info(f"Illuminated: {illuminated_pixels} / {total_pixels} ({pct:.1f} %)")

    except Exception:
        logging.exception(f"{os.getpid()} Inc: Error inside run()")
        raise

    logging.info(f"{os.getpid()}"+"=" * 60)
    logging.info(f"{os.getpid()} End processing data of incidence angle..")
    logging.info(f"{os.getpid()}"+"=" * 60)

    iw.close()
    sw.close()

    return inc_stack, inc_transform_ref, inc_crs_value, ilu_stack, ilu_transform_ref, ilu_crs_value, 


def calc_grid_for_perimeter(args, cfg, perimeter_key):
    """
    Build a list of tile origin coordinates (E, N) in LV95 that cover
    either full Switzerland (perimeter_key == 'CH') or the bbox of the
    chosen GPKG perimeter.

    Tiles always align to multiples of grid_size so the CH grid is a
    strict superset — partial-overlap tiles are included (clipping happens
    at the output stage).
    """
    grid_size = args["grid_size"]

    if perimeter_key == "CH":
        # Original full-Switzerland extents
        e_origin = args["east"]   # 2480000
        n_origin = args["north"]  # 1060000
        n_e = 18
        n_n = 12
        grid = []
        for e in range(n_e):
            for n in range(n_n):
                grid.append((e_origin + e * grid_size, n_origin + n * grid_size))
        return grid

    # GPKG perimeter path
    perimeters = cfg["perimeters"]
    gpkg_path = perimeters[perimeter_key] if perimeter_key in perimeters else None
    if not gpkg_path or not os.path.isfile(gpkg_path):
        raise FileNotFoundError(
            f"GPKG for perimeter '{perimeter_key}' not found: {gpkg_path}"
        )

    e_min, n_min, e_max, n_max = load_perimeter_bbox(gpkg_path)

    # Snap tile origins to the CH grid so tiles are always aligned
    ch_e0 = args["east"]
    ch_n0 = args["north"]

    # First tile origin that starts at or before the perimeter bbox
    start_e = ch_e0 + math.floor((e_min - ch_e0) / grid_size) * grid_size
    start_n = ch_n0 + math.floor((n_min - ch_n0) / grid_size) * grid_size

    grid = []
    e = start_e
    while e < e_max:
        n = start_n
        while n < n_max:
            grid.append((e, n))
            n += grid_size
        e += grid_size

    logging.info(
        f"Perimeter '{perimeter_key}': {len(grid)} candidate tiles (before nodata filter)"
    )
    return grid


def tile_contains_valid_data(dom_path, e, n, grid_size):
    with rasterio.open(dom_path) as src:
        window = window_from_bounds(
            e,
            n,
            e + grid_size,
            n + grid_size,
            src.transform
        )
       # for 0 -size window guard
        if window.width <= 0 or window.height <= 0:
            return False
        data = src.read(1, window=window)
        nodata = src.nodata

        if nodata is not None:
            return not np.all(data == nodata)
        return not np.all(np.isnan(data))


def merge_results(tile_results, args, cfg, output_path, label):
    logging.info(f"{os.getpid()} Merging all tiles into a single Switzerland-wide TIFF...")
    
    src_files_to_mosaic = []
    memfiles = []

    #counter = 0
    for stack, transform, crs in tile_results:
        # Create an in-memory rasterio dataset for merging
        memfile = rasterio.io.MemoryFile()
        memfiles.append(memfile)
        dataset = memfile.open(
            driver='GTiff',
            height=stack.shape[1],
            width=stack.shape[2],
            count=stack.shape[0],
            dtype=stack.dtype,
           transform=transform,
            crs=crs
        ) 
        for i in range(stack.shape[0]):
             dataset.write(stack[i], i+1)

        src_files_to_mosaic.append(dataset)

    mosaic, out_transform = merge(src_files_to_mosaic)

    # Save final TIFF
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
    doy_str = f"{dt_utc.timetuple().tm_yday:03d}"
    datum = dt_utc.strftime('%Y%m%d_%H%M%S')

    output_tif = os.path.join(output_path, f"{label}_DOM_CH_DOY_{doy_str}_{datum}_test.tif")
    if os.path.isfile(output_tif):
        os.remove(output_tif)
    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=mosaic.shape[0],
        dtype=mosaic.dtype,
        crs=crs,
        transform=out_transform,
        tiled=True,
        compress="LZW"
    ) as dst:

        start_sec = 10*3600
        step_sec = 120

        for i in range(mosaic.shape[0]):
            dst.write(mosaic[i], i+1)
            ms_value = (start_sec + i*step_sec) * 1000  # milliseconds for 10:00 + i*step
            dst.set_band_description(i+1, f"shadow_{ms_value}")
            dst.update_tags(i+1, TIMESTAMP_MILLISECONDS=str(ms_value))


    logging.info(f"{os.getpid()} Merged {label} Switzerland-wide TIFF written: {output_tif}")
    
    # close
    for ds in src_files_to_mosaic:
         ds.close()
    for mf in memfiles:
         mf.close()

if __name__ == "__main__":
    """Entrypoint for the application"""
    __args = parse_args()
    __cfg = get_config()
    if os.path.isdir(__cfg["logfolder_path"]):

            try:
                loglvl = getattr(logging, __args["loglevel"].strip().upper())
                ctx = mp.get_context("spawn")  # Windows-safe; funktioniert überall
                log_queue: mp.Queue = ctx.Queue(-1)
                listener = ctx.Process(target=log_listener_process, args=(log_queue, Path(__cfg["logfolder_path"]), loglvl), name="LogListener")
                listener.start()

                setup_logging(level=loglvl, logfolder=Path(__cfg["logfolder_path"]))

                # generate list with all start_e and start_n
                # single process                
                logging.info(f"origin coordinate East {__args['east']} and North {__args['north']}")

                perimeter_key = __args["perimeter"]
                logging.info(f"Selected perimeter: {perimeter_key}")

                grid_ch = calc_grid_for_perimeter(__args, __cfg, perimeter_key)
                valid_tiles = [
                    coord_tuple for coord_tuple in grid_ch
                    if tile_contains_valid_data(
                        __cfg["dom_path"], 
                        coord_tuple[0], 
                        coord_tuple[1],
                        __args["grid_size"])
                ]

                nodata_tiles = len(grid_ch) - len(valid_tiles)

                logging.info(f"Total tiles: {len(grid_ch)}")
                logging.info(f"Valid tiles: {len(valid_tiles)}")
                logging.info(f"Number of nodata tiles: {nodata_tiles}")

                tasks = [(__args, __cfg, coord_tuple) for coord_tuple in valid_tiles]
                
                # multiprocess
                n_proc = 5 #min(len(tasks), os.cpu_count() - 1)
                logging.info(f"Starting processing of {len(tasks)} tiles with {n_proc} workers")

                with ctx.Pool(processes=n_proc, initializer=setup_logging, initargs=(logging.INFO,)) as pool:
                    try:
                        results = pool.starmap(run, tasks)

                        incidence_CH = [(res[0], res[1], res[2]) for res in results]
                        illuminate_CH = [(res[3], res[4], res[5]) for res in results]
                    except KeyboardInterrupt:
                        logging.warning("KeyboardInterrupt received. Terminating workers...")
                        pool.terminate()
                        pool.join()
                        raise
            
                # as soon as all workers have done their job: join merge
                # single process
                merge_results(incidence_CH, __args, __cfg, __cfg["output_path_IG"], f"incidence_{perimeter_key}")
                merge_results(illuminate_CH, __args, __cfg, __cfg["output_path_SW"], f"illuminate_{perimeter_key}")

                # Listener beenden
                log_queue.put_nowait(None)
                listener.join()

            except Exception as exc:
                print(exc)
                try:
                    logging.fatal(exc)
                except Exception:
                    pass
                sys.exit(1)
    else:
        print("Not working as logfolder path not found")



# def calc_grid(args, cfg) -> list[tuple[float, float]]:
#     n_e = 18    # numbe of cells in east direction: 18
#     n_n = 12     # number of cells in north direction: 12
#     grid_CH = []  # list containing coordinate tuples, e.g. [(2600000, 1200000), (2620000, 1220000)]
#     for e in range(n_e):
#         for n in range(n_n):
#             grid_CH.append( (args["east"] + e * args["grid_size"], args["north"] + n * args["grid_size"]) )

#     return grid_CH

