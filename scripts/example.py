""" Analyses SAP Excelexports """
import datetime
import logging
import os
import sys
from argparse import ArgumentParser
from config import get_config
from datetime import date
from pathlib import Path
from xlsx_files import ANALYSE


def parse_args():
    parser = ArgumentParser(description="Update Analysefile")
    parser.add_argument(
        "--month",
        "-m",
        required=False,
        default=date.today().month - 1 if date.today().month > 1 else 12,
        type=int,
        help="Month in the year",
    )
    parser.add_argument(
        "--year",
        "-y",
        required=False,
        default=date.today().year if date.today().month > 1 else date.today().year-1,
        type=int,
        help="Month in the year",
    )
    parser.add_argument(
        "--loglevel",
        "-ll",
        required=False,
        default="",
        type=str,
        help="Logelevel",
    )
    return vars(parser.parse_args())


def setup_logging(level=logger.Info,
                  fmt="%(asctime)s [%(levelname)s] %(message)s",
                  logfolder: Path = None
                  ):
    loghandlers = [logging.StreamHandler()]
    logfile = ""
    if logfolder:
        logfolder.mkdir(parents=True, exist_ok=True)
        log_file = "{}_{}.log".format(
            os.path.splitext(os.path.basename(__file__))[0],
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        logfile = os.path.join(logfolder, log_file)
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(logging.Formatter(fmt))
        loghandlers.append(file_handler)
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=loghandlers
    )
    logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
    return logger, logfile


def run():
    logger, _logfile = setup_logging(loglevel)
    logger.info("Start processing data..")


if __name__ == "__main__":
    """Entrypoint for the application"""
    try:
        run()
    except Exception:
        sys.exit(1)
