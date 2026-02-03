"""Analyses SAP Excelexports"""

import datetime
import logging
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser
from datetime import date
from pathlib import Path
from landschaftsgradient import InzidenWinkel, HelperFunctions

LOGLEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_loglevel(level_str: str) -> int:
    return getattr(logging, level_str.strip().upper())


def parse_args():
    parser = ArgumentParser(description="Add arguments of incidence_angle_grid")
    parser.add_argument(
        "--date",
        "-d",
        default="17.06.2025",
        type=str,
        help="Date (DD.MM.YYYY), default: 13.12.2025",
    )
    parser.add_argument(
        "--time",
        "-t",
        default="10:26:21",
        type=str,
        help="Time UTC (HH:MM:SS), default: 12:22:00",
    )
    parser.add_argument(
        "--east",
        "-e",
        default=2640546.44,
        type=float,
        help="Easting in LV95 [m], default: 2600000.0 (Bern)",
    )
    parser.add_argument(
        "--north",
        "-n",
        default=1180763.41,
        type=float,
        help="Northing in LV95 [m], default: 1200000.0 (Bern)",
    )
    parser.add_argument(
        "--grid_size",
        "-gs",
        default=1000,
        type=int,
        help="meters",
    )
    parser.add_argument(
        "--grid_step",
        "-gstp",
        default=10,
        type=int,
        help="meters (100m / 10m = 10 points per axis = 100 total)",
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
    logging.info("INZIDENZWINKEL-RASTER BERECHNUNG")
    logging.info("=" * 60)
    logging.info(f"Datum/Zeit (UTC): {datum}")
    logging.info(f"Standort LV95:    E={east_lv95:.2f} / N={north_lv95:.2f}")
    logging.info(f"Standort WGS84:   Lon={lon_loc:.5f} / Lat={lat_loc:.5f}")
    logging.info(
        f"Grid:       Grid size={grid_size:.2f} m / Grid step={grid_step:.2f} m"
    )
    logging.info(f"Rastergroesse: {grid_size:.2f}m x {grid_step:.2f}m")
    logging.info(f"DOM-Datei:     {dom_path}")
    logging.info("=" * 60)


def run(args, cfg):
    logging.info("Start processing data..")
    logparameter(args, cfg)
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])
    try:
        iw = InzidenWinkel(
            dom=cfg["dom_path"],
            planets=cfg["planets"],
            output_path=cfg["output_path_IG"],
        )
        grid, transform, points = iw.calc_incidence_grid(
            e_lv95=args["east"],
            n_lv95=args["north"],
            dateoi=args["date"],
            timeoi=args["time"],
            grid_size=args["grid_size"],
            grid_step=args["grid_step"],
        )

        # Determine output directory (needed for ephemeris data)
        east_lv95 = int(args['east'])
        north_lv95 = int(args['north'])
        datum = dt_utc.strftime('%Y%m%d_%H%M%S')
        
        basename = (
            f"incidence_{east_lv95}_{north_lv95}_"
            f"{datum}"
        )
        filename_tif = f"{basename}.tif"
        filename_csv = f"{basename}.csv"
        output_dir = cfg["output_path_IG"]
        os.makedirs(output_dir, exist_ok=True)
        output_path_tif = os.path.join(output_dir, filename_tif)
        output_path_csv = os.path.join(output_dir, filename_csv)
        # write files
        iw.write_geotiff(grid, transform, output_path_tif)
        iw.write_csv(points, output_path_csv)
        # Log statistics
        valid_values = grid[~np.isnan(grid)]
        if len(valid_values) > 0:
            logging.info("Statistik:")
            logging.info(f"  Min Inzidenzwinkel: {np.min(valid_values):.2f} Grad")
            logging.info(f"  Max Inzidenzwinkel: {np.max(valid_values):.2f} Grad")
            logging.info(f"  Mittelwert: {np.mean(valid_values):.2f} Grad")
        else:
            logger.warning("Keine gueltigen Werte berechnet!")
    except Exception as e:
        logging.error(e)
        sys.exit(-1)
    logging.info("=" * 60)
    logging.info("End processing data..")
    logging.info("=" * 60)


if __name__ == "__main__":
    """Entrypoint for the application"""
    __args = parse_args()
    __cfg = get_config()
    if os.path.isdir(__cfg["logfolder_path"]):
        try:
            loglvl = getattr(logging, __args["loglevel"].strip().upper())
            setup_logging(level=loglvl, logfolder=Path(__cfg["logfolder_path"]))
            run(__args, __cfg)
        except Exception as exc:
            print(exc)
            try:
                logging.fatal(exc)
            except Exception:
                pass
            sys.exit(1)
    else:
        print("Not working as logfolder path not found")