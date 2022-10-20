# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os

import colorlog
import torch


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger("tqdm")

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name="default", debug=False, save_to_dir=None) -> logging.Logger:
    if debug:
        log_format = (
            "%(asctime)s - "
            "%(levelname)s : "
            "%(name)s - "
            "%(pathname)s[%(lineno)d]:"
            "%(funcName)s - "
            "%(message)s"
        )
    else:
        log_format = "%(asctime)s - " "%(levelname)s : " "%(name)s - " "%(message)s"
    bold_seq = "\033[1m"
    colorlog_format = f"{bold_seq} %(log_color)s {log_format}"
    colorlog.basicConfig(format=colorlog_format, datefmt="%y-%m-%d %H:%M:%S")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    try:
        rank = torch.distributed.get_rank()
    except Exception:
        rank = 0
    if rank != 0:
        logger.setLevel(logging.WARNING)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, "log", "debug.log"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, "log", "warning.log"))
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, "log", "error.log"))
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
