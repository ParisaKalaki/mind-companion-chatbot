"""
download_model.py
-----------------
Downloads the fine-tuned crisis classifier weights from Google Drive into:

    models/crisis_classifier/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin

Run:
    python download_model.py

Requirements:
    pip install gdown
"""

import sys

import gdown
from loguru import logger

from config import MODELS_DIR

# ---------------------------------------------------------------------------
# Where the model files live
# ---------------------------------------------------------------------------
CRISIS_MODEL_DIR = MODELS_DIR / "crisis_classifier"

# Map filename -> Google Drive file ID
CRISIS_MODEL_FILES = {
    "config.json":           "16tvn1X7CgSvMNWRQxpdHNqyBOfH_YwXD",
    "model.safetensors":     "1kFfjErvjm2EfJW5u2DMYxXwWoDnFmQaG",
    "tokenizer.json":        "1IzTCTPK7ElMYSSj5Vl5x57M1LTJ3h3VB",
    "tokenizer_config.json": "1IHtzJajxLyd8sEJXE7_mENPUC-LZHj-j",
    "training_args.bin":     "1qHRNifMtwQs5ebq_mDAj3xkIuiof08kF",
}


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------
def download_crisis_classifier(force: bool = False) -> None:
    """
    Download the crisis classifier model files from Google Drive into
    `models/crisis_classifier/`.

    Parameters
    ----------
    force : bool
        If True, re-download files even if they already exist locally.
    """
    CRISIS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Crisis classifier directory: {CRISIS_MODEL_DIR}")

    downloaded, skipped = 0, 0

    for filename, file_id in CRISIS_MODEL_FILES.items():
        out_path = CRISIS_MODEL_DIR / filename

        if out_path.exists() and not force:
            logger.info(f"  {filename} already exists, skipping.")
            skipped += 1
            continue

        logger.info(f"  Downloading {filename} ...")
        try:
            result = gdown.download(
                id=file_id,
                output=str(out_path),
                quiet=False,
            )
            if result is None:
                # gdown returns None on failure (e.g. quota exceeded, bad ID,
                # or permission issue on the Drive file).
                raise RuntimeError(
                    f"gdown returned no path for {filename}. "
                    "Check that the file is shared as 'Anyone with the link'."
                )
            downloaded += 1
        except Exception as exc:
            logger.error(f"Failed to download {filename}: {exc}")
            # Remove a partial/empty file so reruns don't think it's already done
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
            raise

    logger.success(
        f"Crisis classifier ready | Downloaded: {downloaded} | "
        f"Skipped: {skipped} | Location: {CRISIS_MODEL_DIR}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    download_crisis_classifier()


if __name__ == "__main__":
    main()