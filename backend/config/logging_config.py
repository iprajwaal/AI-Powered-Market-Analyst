import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from .settings import LOG_LEVEL, LOG_FILE, LOG_FILE_MAX_BYTES, LOG_FILE_BACKUP_COUNT, BASE_DIR

logs_dir = Path(BASE_DIR) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# configure logging
log_file_path = logs_dir / LOG_LEVEL

def setup_logging():
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(
        logs_dir / LOG_FILE,
        maxBytes=10*1024*1024, # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    #Disable logging for specific libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set the logging level for the root logger
    logger.setLevel(log_level)

    
    return logger
