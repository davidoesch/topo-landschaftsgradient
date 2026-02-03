# Imports
import datetime
import logging
import os
import json
import sys
from argparse import ArgumentParser
from datetime import date
from rasterio.windows import from_bounds
from pathlib import Path
import numpy as np
from landschaftsgradient import ImgChecker, HelperFunctions

LOGLEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_loglevel(level_str: str) -> int:
    return getattr(logging, level_str.strip().upper())


def parse_args():
    parser = ArgumentParser(description="Add arguments of incidence_angle verification")
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


def run(args, cfg):
    logging.info("Start processing data..")
    dt_utc = HelperFunctions.parse_datetime(args["date"], args["time"])

    try:

        source_names = ["agroscope", "r_sun", "saga"]
        seasons = {
            "jun": {
                "test": cfg["test_raster_path_jun"],
                "ref": cfg["ref_raster_CH_paths_jun"],
            },
            "dec": {
                "test": cfg["test_raster_path_dec"],
                "ref": cfg["ref_raster_CH_paths_dec"],
            },
        }
        for season_name, paths in seasons.items():
            test_raster_path = paths["test"]
            ref_raster_paths = paths["ref"]
            for idx, ref_raster_path in enumerate(ref_raster_paths):
                source = source_names[idx]
                IC = ImgChecker(
                    test_raster_path=test_raster_path,
                    ref_raster_path=ref_raster_path,
                    output_path=cfg["output_path_IG"],
                )
                diff = IC.compare()

                # Determine output directory (csv))
                east_lv95 = int(args['east'])
                north_lv95 = int(args['north'])
                datum = dt_utc.strftime('%Y%m%d_%H%M%S')

                basename = (
                    f"ImgCheck_{season_name}_{source}_{east_lv95}_{north_lv95}_"
                    f"{datum}"
                )
                IC.write_csv(diff, f"{basename}.csv")

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
