"""Analyses SAP Excelexports"""

import datetime
import logging
import os
import sys
import json
from argparse import ArgumentParser
from pathlib import Path
from landschaftsgradient import SonnenWinkel, HelperFunctions

LOGLEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_loglevel(level_str: str) -> int:
    return getattr(logging, level_str.strip().upper())


def parse_args():
    parser = ArgumentParser(description="Add arguments of shadow_check_location")
    parser.add_argument(
        "--date",
        "-d",
        default="13.12.2025",
        type=str,
        help="Date (DD.MM.YYYY), default: 13.12.2025",
    )
    parser.add_argument(
        "--time",
        "-t",
        default="12:22:00",
        type=str,
        help="Time UTC (HH:MM:SS), default: 12:22:00",
    )
    parser.add_argument(
        "--east",
        "-e",
        default=2600000.0,
        type=float,
        help="Easting in LV95 [m], default: 2600000.0 (Bern)",
    )
    parser.add_argument(
        "--north",
        "-n",
        default=1200000.0,
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
    search_read = cfg["search-dist"]
    dom_read = cfg["dom_path"]

    logging.info("=" * 60)
    logging.info("SCHATTEN-CHECK FUER EINZELNEN STANDORT")
    logging.info("=" * 60)
    logging.info(f"Datum/Zeit (UTC): {datum}")
    logging.info(f"Standort LV95:    E={east_lv95:.2f} / N={north_lv95:.2f}")
    logging.info(f"Standort WGS84:   Lon={lon_loc:.5f} / Lat={lat_loc:.5f}")
    logging.info(f"Suchradius:       {search_read} km")
    logging.info(f"DOM-Datei:        {dom_read}")
    logging.info("=" * 60)


def run(args, cfg):
    logging.info("Start processing data..")
    logparameter(args, cfg)
    try:
        sw = SonnenWinkel(
            dom=cfg["dom_path"],
            planets=cfg["planets"],
            output_path=cfg["output_path_SC"],
        )
        sw.getdat_point(
            e_lv95=args["east"],
            n_lv95=args["north"],
            dateoi=args["date"],
            timeoi=args["time"],
            search_dist=cfg["search-dist"],
        )

        sw.getdat_point(
            e_lv95=2640546.44,
            n_lv95=1180763.41,
            dateoi="25.12.2023",
            timeoi="10:34:41",
            search_dist=cfg["search-dist"],
        )

    except Exception as e:
        logging.error(e)
        sys.exit(-1)
    logging.info("End processing data..")


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
