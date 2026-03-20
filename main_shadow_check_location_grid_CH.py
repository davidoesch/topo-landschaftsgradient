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
from iluina_module import SonnenWinkel, HelperFunctions
from multiprocessing import Pool
import multiprocessing as mp


LOGLEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_loglevel(level_str: str) -> int:
    return getattr(logging, level_str.strip().upper())


def parse_args():
    parser = ArgumentParser(description="Add arguments of shadow_check_grid_CH")
    parser.add_argument(
        "--date",
        "-d",
        default="20.06.2025",
        type=str,
        help="Date (DD.MM.YYYY), default: 13.12.2025",
    )
    parser.add_argument(
        "--time",
        "-t",
        default="10:00:00",
        type=str,
        help="Time UTC (HH:MM:SS), default: 10:00:00 (till 11:02:00 every 2 minutes)",
    )
    parser.add_argument(
        "--east",
        "-e",
        default=2480000,
        type=float,
        help="Easting in LV95 [m], default: 2480000 (start of CH grid)",
    )
    parser.add_argument(
        "--north",
        "-n",
        default=1060000,
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
        os.path.join(os.path.dirname(__file__), "config", "config.json"), "r"
    ) as file:
        data = json.load(file)
    return data


def logparameter(args, cfg):
    lon_loc, lat_loc = HelperFunctions.lv95_to_wgs84(args["east"], args["north"])
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
    datum = dt_utc.strftime('%d.%m.%Y %H:%M:%S')
    east_lv95 = args["east"]
    north_lv95 = args["north"]
    grid_size = args["grid_size"]
    grid_step = args["grid_step"]
    dom_path = cfg["dom_path"]

    logging.info("=" * 60)
    logging.info(f"{os.getpid()} SONNENWINKEL-RASTER BERECHNUNG")
    logging.info(f"{os.getpid()}" + "=" * 60)
    logging.info(f"{os.getpid()} Datum/Zeit (UTC): {datum}")
    logging.info(f"{os.getpid()} Standort LV95:    E={east_lv95:.2f} / N={north_lv95:.2f}")
    logging.info(f"{os.getpid()} Standort WGS84:   Lon={lon_loc:.5f} / Lat={lat_loc:.5f}")
    logging.info(
        f"{os.getpid()} Grid:       Grid size={grid_size:.2f} m / Grid step={grid_step:.2f} m"
    )
    logging.info(f"{os.getpid()} Rastergroesse: {grid_size:.2f}m x {grid_step:.2f}m")
    logging.info(f"{os.getpid()} DOM-Datei:     {dom_path}")
    logging.info(f"{os.getpid()}" + "=" * 60)


def run(args, cfg, coord_tuple):
    logger = logging.getLogger()
    logger.info(f"Start processing data...{os.getpid()}")
    logparameter(args, cfg)
   
    try:
        # initialize class with static config parameters
        sw = SonnenWinkel(
            dom=cfg["dom_path"],
            planets=cfg["planets"],
            search_dist=cfg["search-dist"],
            output_path=cfg["output_path_SW"],
        )
        
        # start_sec = 10 * 3600               # 10:00:00 -> 36000 s
        # end_sec   = 10 * 3600 + 4 * 60      # 11:02:00 -> 39720 s
        # step_sec  = 120                     # 2 minutes

        # seconds_since_midnight = list(range(start_sec, end_sec + 1, step_sec))  # inclusive

        # def sec_to_hms_str(sec: int) -> str:
        #     hh = sec // 3600
        #     mm = (sec % 3600) // 60
        #     ss = sec % 60
        #     return f"{hh:02d}:{mm:02d}:{ss:02d}"

        # time_strings = [sec_to_hms_str(s) for s in seconds_since_midnight]
        
        bands = []
        transform_ref = None
        shape_ref = None

        #for t_str in time_strings:
        # calculate illuminate for every 10x10m cell of 20x20km tile
        shadow_tile, transform = sw.calc_illuminate_grid(
            e_lv95=coord_tuple[0],
            n_lv95=coord_tuple[1],
            dateoi=args["date"],
            timeoi=args["time"],
            grid_size=args["grid_size"],
            grid_step=args["grid_step"],
        )
        # output_path_tif = os.path.join(cfg["output_path_IG"], f"incidence_DOM_CH_DOY_3x3_{coord_tuple[0]}_{coord_tuple[1]}.tif")
        # if os.path.isfile(output_path_tif):
        #     os.remove(output_path_tif)
        # iw.write_geotiff(tile, transform, output_path_tif)

        if transform_ref is None:
            transform_ref = transform
            shape_ref = shadow_tile.shape
        else:
            if transform != transform_ref:
                logging.warning("Transform changed across time slices; using the first transform.")
            if shadow_tile.shape != shape_ref:
                raise ValueError(
                    f"Inconsistent tile shapes across time slices: got {shadow_tile.shape}, expected {shape_ref}"
                )

        bands.append(shadow_tile)

        if not bands:
            raise RuntimeError("No bands were created for this tile")
        
        stack = np.stack(bands, axis=0)  # [time, H, W]
        crs_value = getattr(sw, "crs", None) or "EPSG:2056"

        # Log statistics
        valid_values = shadow_tile[~np.isnan(shadow_tile)]
        if len(valid_values) > 0:
            logging.info(f"{os.getpid()} Statistik:")
            logging.info(f"  {os.getpid()} Min Sonnenwinkel: {np.min(valid_values):.2f} Grad")
            logging.info(f"  {os.getpid()} Max Sonnewinkel: {np.max(valid_values):.2f} Grad")
            logging.info(f"  {os.getpid()} Mittelwert: {np.mean(valid_values):.2f} Grad")
        else:
            logging.warning(f"{os.getpid()} Keine gueltigen Werte berechnet!")

        # if args["compare"]:
        #     chk = ImgChecker2(output_path_tif, cfg)
        #     chk.comparer()
    except Exception:
        logging.exception(f"{os.getpid()} Error inside run()")
        raise

    # except Exception as e:
    #     logging.error(e)
    #     sys.exit(-1)
    logging.info(f"{os.getpid()}"+"=" * 60)
    logging.info(f"{os.getpid()} End processing data..")
    logging.info(f"{os.getpid()}"+"=" * 60)

    sw.close()

    return stack, transform_ref, crs_value


def calc_grid(args, cfg) -> list[tuple[float, float]]:
    n_e = 4     # numbe of cells in east direction: 18
    n_n = 4    # number of cells in north direction: 12
    grid_CH = []  # list containing coordinate tuples, e.g. [(2600000, 1200000), (2620000, 1220000)]
    for e in range(n_e):
        for n in range(n_n):
            grid_CH.append( (args["east"] + e * args["grid_size"], args["north"] + n * args["grid_size"]) )

    return grid_CH


def tile_contains_valid_data(dom_path, e, n, grid_size):
    with rasterio.open(dom_path) as src:
        window = window_from_bounds(
            e,
            n,
            e + grid_size,
            n + grid_size,
            src.transform
        )
        data = src.read(1, window=window)
        nodata = src.nodata

        if nodata is not None:
            return not np.all(data == nodata)
        return not np.all(np.isnan(data))


def merge_results(tile_results, args, cfg):
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

        # write whole stack at once:
        # dataset.write(stack)
        src_files_to_mosaic.append(dataset)

    mosaic, out_transform = merge(src_files_to_mosaic)

    # Save final TIFF
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
    doy_str = f"{dt_utc.timetuple().tm_yday:03d}"
    datum = dt_utc.strftime('%Y%m%d_%H%M%S')

    output_tif = os.path.join(cfg["output_path_SW"], f"Sonnen_DOM_CH_DOY_{doy_str}_{datum}_5x5_test1.tif")
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
        # write all bands at once (newly added) 
        # dst.write(mosaic)
        
        start_sec = 10*3600
        step_sec = 120

        for i in range(mosaic.shape[0]):
            dst.write(mosaic[i], i+1)
            ms_value = (start_sec + i*step_sec) * 1000  # milliseconds for 10:00 + i*step
            dst.set_band_description(i+1, f"shadow_{ms_value}")
            dst.update_tags(i+1, TIMESTAMP_MILLISECONDS=str(ms_value))


    logging.info(f"{os.getpid()} Merged Switzerland-wide TIFF written: {output_tif}")
    
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

                grid_ch = calc_grid(__args, __cfg)
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
                n_proc = 2 #min(len(tasks), os.cpu_count() - 1)
                logging.info(f"Starting processing of {len(tasks)} tiles with {n_proc} workers")

                with ctx.Pool(processes=n_proc, initializer=setup_logging, initargs=(logging.INFO,)) as pool:
                    try:
                        mosaic_CH = pool.starmap(run, tasks)
                    except KeyboardInterrupt:
                        logging.warning("KeyboardInterrupt received. Terminating workers...")
                        pool.terminate()
                        pool.join()
                        raise
            
                # as soon as all workers have done their job: join merge
                # single process
                merge_results(mosaic_CH, __args, __cfg)

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



# def merge_results_sw(tile_results, args, cfg):
#     logging.info(f"{os.getpid()} Merging shadow tiles (illuminated only)...")

#     src_files_to_mosaic = []
#     memfiles = []

#     for shadow_buffer, transform, crs in tile_results:

#         # --- keep only illuminated ---
#         data = (shadow_buffer == 0).astype(np.uint8)

#         # Create in-memory raster
#         memfile = rasterio.io.MemoryFile()
#         memfiles.append(memfile)

#         dataset = memfile.open(
#             driver='GTiff',
#             height=data.shape[0],
#             width=data.shape[1],
#             count=1,
#             dtype=data.dtype,
#             transform=transform,
#             crs=crs
#         )

#         dataset.write(data, 1)
#         src_files_to_mosaic.append(dataset)

#     # merge
#     mosaic, out_transform = merge(src_files_to_mosaic)

#     # output file
#     dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
#     doy_str = f"{dt_utc.timetuple().tm_yday:03d}"
#     datum = dt_utc.strftime('%Y%m%d_%H%M%S')

#     output_tif = os.path.join(
#         cfg["output_path_SW"],
#         f"shadow_illuminated_DOM_CH_DOY_{doy_str}_{datum}.tif"
#     )

#     if os.path.isfile(output_tif):
#         os.remove(output_tif)

#     with rasterio.open(
#         output_tif,
#         'w',
#         driver='GTiff',
#         height=mosaic.shape[1],
#         width=mosaic.shape[2],
#         count=1,
#         dtype=mosaic.dtype,
#         crs=crs,
#         transform=out_transform,
#         tiled=True,
#         compress="LZW"
#     ) as dst:

#         dst.write(mosaic[0], 1)
#         dst.set_band_description(1, "illuminated (shadow=0)")
#         dst.update_tags(1, DESCRIPTION="Illuminated pixels (no shadow)")

#     logging.info(f"{os.getpid()} Shadow GeoTIFF written: {output_tif}")

#     # cleanup
#     for ds in src_files_to_mosaic:
#         ds.close()
#     for mf in memfiles:
#         mf.close()
