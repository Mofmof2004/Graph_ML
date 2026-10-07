"""Download the RecipeNLG dataset from Kaggle.

Install the Kaggle client before running this script:

    python -m pip install kagglehub

Kaggle credentials must be configured according to the kagglehub
documentation. The dataset is downloaded to ``data/raw`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DATASET_HANDLE = "paultimothymooney/recipenlg"
DEFAULT_OUTPUT_DIR = Path("data/raw")


def download_dataset(output_dir: Path) -> Path:
	"""Download RecipeNLG and return the directory containing its files."""
	try:
		import kagglehub
	except ImportError as error:
		raise SystemExit(
			f"Could not import kagglehub using {sys.executable}.\n"
			"Install it for this exact interpreter with:\n"
			f'"{sys.executable}" -m pip install kagglehub\n'
			f"Original error: {error}"
		) from error

	output_dir.mkdir(parents=True, exist_ok=True)
	downloaded_path = kagglehub.dataset_download(
		DATASET_HANDLE,
		output_dir=str(output_dir),
	)
	return Path(downloaded_path)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download the RecipeNLG dataset from Kaggle."
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory in which to store the downloaded dataset "
		"(default: data/raw).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	dataset_path = download_dataset(args.output_dir)
	print(f"RecipeNLG downloaded to: {dataset_path}")


if __name__ == "__main__":
	main()
