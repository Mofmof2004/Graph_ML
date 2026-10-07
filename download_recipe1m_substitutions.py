"""Download the Recipe1MSubs evaluation files.

The files are published by the authors at:
https://dl.fbaipublicfiles.com/gismo/{train,val,test}_comments_subs.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://dl.fbaipublicfiles.com/gismo"
SPLITS = ("train", "val", "test")
DEFAULT_OUTPUT_DIR = Path("data/eval/recipe1msubs")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Recipe1MSubs substitution evaluation files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the downloaded pickle files.",
    )
    args = parser.parse_args()

    for split in SPLITS:
        filename = f"{split}_comments_subs.pkl"
        destination = args.output_dir / filename
        if destination.exists():
            print(f"Already exists: {destination}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url} -> {destination}")
        download_file(url, destination)


if __name__ == "__main__":
    main()
