# src/utils/logging_config.py
# ------------------------------------------------------------
# Logging configuration module for FraudGuardian AI
# Provides production-grade logging with console + rotating file handler
# ------------------------------------------------------------

import logging
import logging.handlers
import os
import sys


def setup_logging(
    log_file: str = "logs/app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure root logger with console and file handlers.

    Args:
        log_file (str): Path to the log file. Default: logs/app.log
        console_level (int): Logging level for console output. Default: INFO
        file_level (int): Logging level for file output. Default: DEBUG
    """
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # capture all, handlers decide filtering

    # Clear existing handlers if re-run in interactive contexts (e.g., Jupyter)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name (str): Name of the logger (usually __name__)
        level (int): Logging level for this logger. Default: DEBUG

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# Example usage when running this file directly
if __name__ == "__main__":
    setup_logging()
    log = get_logger(__name__)
    log.info("Application started")
    log.debug("Debug details example: %s", {"key": "value"})
    log.warning("This is a warning")
    log.error("This is an error message")
