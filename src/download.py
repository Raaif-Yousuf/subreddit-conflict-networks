"""Download the raw SNAP Reddit Hyperlink TSVs into data/raw/.

Source: https://snap.stanford.edu/data/soc-RedditHyperlinks.html
Files : soc-redditHyperlinks-body.tsv (~304 MB), soc-redditHyperlinks-title.tsv (~352 MB)

Run: python src/download.py           # skips files that already exist
     python src/download.py --force   # re-download
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE_URL = "https://snap.stanford.edu/data/"
FILES = ("soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv")


def download(name: str, force: bool = False) -> Path:
    """Fetch one file to data/raw/<name>, writing to a .part file first so an interrupted run leaves no half file."""
    dest = RAW / name
    if dest.exists() and not force:
        print(f"skip  {name} (exists, {dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch {BASE_URL + name}")
    with urllib.request.urlopen(BASE_URL + name) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.replace(dest)
    print(f"done  {name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name, args.force)


if __name__ == "__main__":
    main()
