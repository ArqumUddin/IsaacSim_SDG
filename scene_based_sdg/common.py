# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities shared across the scene-based SDG system.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configures the loggers for the application.

    Args:
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    loggers = ["scene_based_sdg", "scene_based_sdg_utils"]
    formatter = logging.Formatter('[SDG] %(message)s')

    for name in loggers:
        log = logging.getLogger(name)
        log.setLevel(level)

        if not log.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            log.addHandler(handler)
