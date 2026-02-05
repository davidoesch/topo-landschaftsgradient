# Imports
import datetime
import logging
import os
import json
import sys
from glob import glob
from argparse import ArgumentParser
from pathlib import Path
from landschaftsgradient import ImgChecker2

LOGLEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_args():
    parser = ArgumentParser(description="Add arguments of incidence_angle verification")
    parser.add_argument(
        "--loglevel",
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
    logging.info("=" * 60)
    logging.info("Start processing data..")
    logging.info("=" * 60)

    for tif in glob(os.path.join(cfg["output_path_IG"], "*.tif")):
        logging.info(f"Processing {tif}...")
        imgchk2 = ImgChecker2(tif, cfg)
        imgchk2.comparer()
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
