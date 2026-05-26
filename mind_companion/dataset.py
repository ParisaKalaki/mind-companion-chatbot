"""
dataset.py
----------
Creates the project data directories and downloads raw datasets into
RAW_DATA_DIR.

Sources:
    - Suicide / Crisis    -> Kaggle  (nikhileswarkomati/suicide-watch)

Kaggle authentication:
    Add these to a `.env` file at the project root (and make sure
    `.env` is in `.gitignore`):

        KAGGLE_USERNAME=your_username
        KAGGLE_KEY=your_api_key_from_kaggle_settings

    Get the values from https://www.kaggle.com/settings
    -> "API" section -> "Create New Token" (downloads kaggle.json).

All cleaning, filtering, and splitting is handled in the notebook /
EDA script. This file only downloads raw data.
"""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger

# Importing config also runs load_dotenv(), so KAGGLE_USERNAME / KAGGLE_KEY
# end up in os.environ before we touch the Kaggle API.
from config import (
    PROJ_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)

REQUIRED_ENV_VARS = ("KAGGLE_USERNAME", "KAGGLE_KEY")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
KAGGLE_SUICIDE_SLUG = "nikhileswarkomati/suicide-watch"
CRISIS_RAW_FILENAME = "crisis_raw.csv"


# ---------------------------------------------------------------------------
# Environment / credentials check
# ---------------------------------------------------------------------------
def check_env_file() -> None:
    """
    Verify that `<PROJ_ROOT>/.env` exists and contains KAGGLE_USERNAME and
    KAGGLE_KEY. If anything is missing, print a clear message and exit.

    This runs BEFORE any folder creation or download attempt, so the user
    fixes credentials first instead of getting a half-set-up project.
    """
    env_path = PROJ_ROOT / ".env"

    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        print('\n\n'+'*'*50)
        print('Error: Please follow the following instructions')
        print("Please create a ../.env file with the following variables to proceed:")
        for var in REQUIRED_ENV_VARS:
            print(f"  {var}=<your_value>")
        print(
            "Get credentials from https://www.kaggle.com/settings "
            "-> API -> 'Create New Token'."
        )
        print('*'*50 +'\n\n')
        sys.exit(1)

    # Read the file directly (does NOT touch os.environ), so we are checking
    # what's actually written in .env — not whatever may already be exported
    # in the shell.
    values = dotenv_values(env_path)
    missing = [var for var in REQUIRED_ENV_VARS if not values.get(var)]

    if missing:
        print(
            f"ERROR: .env file at {env_path} is missing required variable(s): "
            f"{', '.join(missing)}"
        )
        print('\n\n'+'*'*50)
        print('Error: Please follow the following instructions')
        print("Please make these variables in your ../.env file with credentials to proceed:")
        for var in missing:
            print(f"  {var}=<your_value>")
        print(
            "Get credentials from https://www.kaggle.com/settings "
            "-> API -> 'Create New Token'."
        )
        print('*'*50 +'\n\n')
        sys.exit(1)

    logger.info(f".env file at {env_path} contains required Kaggle credentials.")


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
def create_directories() -> None:
    """Create all project data / model / report directories if missing."""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------
def _check_kaggle_credentials() -> None:
    """Raise a clear error if Kaggle creds aren't loaded into the env."""
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file.\n"
            "Get them from https://www.kaggle.com/settings -> API -> "
            "'Create New Token' (downloads kaggle.json)."
        )


def download_kaggle_suicide_dataset(force: bool = False) -> Path:
    """
    Download the `nikhileswarkomati/suicide-watch` dataset into RAW_DATA_DIR
    and rename the main CSV to `crisis_raw.csv`.

    Parameters
    ----------
    force : bool
        If True, re-download even if `crisis_raw.csv` already exists.

    Returns
    -------
    Path to the saved crisis_raw.csv file.
    """
    target_path = RAW_DATA_DIR / CRISIS_RAW_FILENAME

    if target_path.exists() and not force:
        logger.info(
            f"{CRISIS_RAW_FILENAME} already exists at {target_path}, "
            "skipping download. Pass force=True to re-download."
        )
        return target_path

    _check_kaggle_credentials()

    # Import kaggle lazily — it tries to authenticate on import in some
    # versions, and we want our own clearer error if creds are missing.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading '{KAGGLE_SUICIDE_SLUG}' to {RAW_DATA_DIR} ...")
    api.dataset_download_files(
        KAGGLE_SUICIDE_SLUG,
        path=str(RAW_DATA_DIR),
        unzip=True,
        quiet=False,
    )

    # The dataset ships as a single CSV (typically `Suicide_Detection.csv`).
    # Locate it and rename to our canonical name.
    csv_candidates = [
        f for f in RAW_DATA_DIR.glob("*.csv") if f.name != CRISIS_RAW_FILENAME
    ]
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV file found in {RAW_DATA_DIR} after Kaggle download. "
            "The dataset structure may have changed."
        )

    # If multiple CSVs ever appear, take the largest — that's the data file.
    source_csv = max(csv_candidates, key=lambda p: p.stat().st_size)
    source_csv.rename(target_path)
    logger.success(f"Saved crisis dataset to: {target_path}")

    return target_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Validating environment configuration ...")
    check_env_file()

    logger.info("Setting up project data directories ...")
    create_directories()

    logger.info("Downloading raw datasets ...")
    download_kaggle_suicide_dataset()

    logger.success("All raw data ready.")


if __name__ == "__main__":
    main()