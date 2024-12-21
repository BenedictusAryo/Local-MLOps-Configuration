"""
Console logging configuration
"""

import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    loglevel_mapping = {
        50: "CRITICAL",
        40: "ERROR",
        30: "WARNING",
        20: "INFO",
        10: "DEBUG",
        0: "NOTSET",
    }

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = self.loglevel_mapping[record.levelno]
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(request_id="app")
        log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> logging.Logger:
    # intercept everything at the root logger
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel("INFO")

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # configure loguru
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "serialize": False,
                "level": "INFO",
                "format": "<level>{level: <6}</level> | <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <blue>{process}</blue> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
            }
        ]
    )

    return logger.bind(request_id=None, method=None)
