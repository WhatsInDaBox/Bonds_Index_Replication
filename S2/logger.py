"""
logger.py
---------
Tous les modules importent `get_logger()` pour obtenir la même instance.
"""

import logging
import sys


def get_logger(name: str = "bond_optimizer") -> logging.Logger:
    """
    Retourne le logger.

    Le logger est configuré une seule fois (guard sur handlers existants).
    Format  : [YYYY-MM-DD HH:MM:SS] LEVEL    - message
    Sortie  : stdout (StreamHandler)
    Niveau  : INFO par défaut, passer DEBUG pour les traces solveur.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
