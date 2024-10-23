"""Sudoku Solver Configuration Module."""

import time
import datetime
import logging
import tracemalloc
import psutil
from logging.handlers import RotatingFileHandler
from os import environ, path
from functools import wraps

basedir = path.abspath(path.dirname(__file__))

class BaseConfig:
    """Base Configuration for Sudoku Solver."""

    @staticmethod
    def setup_logging(app, log_file_path, log_level=logging.INFO):
        """Set up logging configuration for the application.

        Args:
            log_file_path (str): Path to the log file.
            log_level (int): The logging level (e.g. logging.INFO)

        Log Message Format:
            - **%(asctime)s**: The date and time when the log message was
                created.
                - Format: `YYYY-MM-DD HH:MM:SS`
                - Example: `2024-08-02 14:55:23`
            - **%(name)s**: The name of the logger that generated the log
                message.
            - **%(levelname)s**: The severity level of the log message.
                - Examples: `INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`
            - **%(pathname)s**: The full pathname of the source file where the
                log message was generated.
                - Example: `/path/to/your/project/app.py`
            - **%(lineno)d**: The line number in the source file where the log
                message was generated.
                - Example: `42`
            - **%(message)s**: The log message content.
                - Example: `User login successful.`
            - **%(exc_info)s**: Exception information, if an exception is
                being logged.
                - Includes traceback and exception details.

        Returns:
            None
        """
        # Create a file handler for logging
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10240, backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "%Y-%m-%d %H:%M:%S"
            )
        )

        # Configure root logger
        logger = logging.getLogger()  # This gets the root logger
        logger.setLevel(log_level)  # Set the logging level

        # Add handlers if not already present (to avoid duplication)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)

            # Log to console as well
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(file_handler.formatter)
            logger.addHandler(console_handler)

        logging.info(
            f"Logging is set up. Logging to {log_file_path}" +
            f"with level {logging.getLevelName(log_level)}."
        )
        pass

    @staticmethod
    def track_performance(func):
        """Decorator to log the performance of individual operations."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start time
            start_time = time.time()
            
            # Start memory tracking
            tracemalloc.start()
            process = psutil.Process()
            cpu_start = process.cpu_percent(interval=None)

            # Execute the function
            result = func(*args, **kwargs)

            # Measure end time and memory
            elapsed_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            cpu_end = process.cpu_percent(interval=None)
            
            # Stop tracking memory
            tracemalloc.stop()

            # Log the performance metrics
            logging.info(
                f"{func.__name__} took {elapsed_time:.4f} seconds, "
                f"CPU usage: {cpu_end - cpu_start:.2f}%, "
                f"Memory usage: {current / 10**6:.2f} MB, "
                f"Peak memory usage: {peak / 10**6:.2f} MB."
            )

            return result

        return wrapper

    pass


class CSPConfig(BaseConfig):
    """Configuration for Sudoku Solver based on Constraing Propagation."""

    pass


class GAConfig(BaseConfig):
    """Configuration for Sudoku Solver based on Genetic Algorithm."""

    pass
