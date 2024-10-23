"""Logger Module.

Will soon be refactored into a base class for
automatic configuration.
"""

import time
import tracemalloc
import psutil
import logging
from functools import wraps

# Setup logging
logging.basicConfig(
    filename='sudoku_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Performance Logging Decorator with wraps
def log_performance(func):
    @wraps(func)  # This ensures the original function metadata is preserved
    def wrapper(*args, **kwargs):
        # Start time and memory tracking
        start_time = time.time()
        tracemalloc.start()
        process = psutil.Process()

        # CPU and memory usage before function call
        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        # Run the actual function
        result = func(*args, **kwargs)

        # After execution
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # CPU and memory usage after function call
        cpu_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        # Log the performance data
        logging.info(f"Function: {func.__name__}")
        logging.info(f"Execution Time: {end_time - start_time:.4f} seconds")
        logging.info(f"Memory Used: {mem_after - mem_before:.4f} MB (Peak: {peak / (1024 * 1024):.4f} MB)")
        logging.info(f"CPU Usage: {cpu_after - cpu_before:.2f}%")

        return result

    return wrapper