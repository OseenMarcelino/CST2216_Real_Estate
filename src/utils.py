"""Utility functions for logging and configuration"""
import logging
import os
from datetime import datetime


def setup_logging(log_file='logs/application.log'):
    """
    Set up logging configuration for the application.
    Falls back to console-only logging if the log file cannot be created
    (e.g., on read-only cloud filesystems like Streamlit Cloud).

    Args:
        log_file (str): Path to the log file

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('RealEstatePrediction')
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not logger.handlers:
        # Console handler — always added
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler — optional, skip gracefully if directory is not writable
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError):
            logger.warning("Could not create log file; using console logging only.")

    return logger


# Module-level logger used by other src modules
logger = setup_logging()

