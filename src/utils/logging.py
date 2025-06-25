import logging
import sys
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up basic logging to console and optionally to a file."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)

    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging also to file: {log_file}")
