"""
config.py
---------
Project paths and shared configuration. Loaded by every other module.
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR          = PROJ_ROOT / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
INTERIM_DATA_DIR  = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR  = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, route loguru through tqdm.write so progress bars
# don't get clobbered. Tolerant of environments where handler 0 has already
# been removed (e.g. Streamlit Cloud, some notebook setups) — without this
# guard the import raises ValueError on those platforms.
try:
    from tqdm import tqdm
    try:
        logger.remove(0)
    except ValueError:
        # Default handler already removed by the host environment — fine.
        pass
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass