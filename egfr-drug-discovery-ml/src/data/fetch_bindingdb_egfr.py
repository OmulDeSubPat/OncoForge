from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from src.config import RAW_DIR


DOWNLOAD_PAGE = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"
DIRECT_BASE = "https://www.bindingdb.org"
VARIANT_PATTERNS = {
    "articles": r"/rwd/bind/downloads/BindingDB_BindingDB_Articles_(\d{6})_tsv\.zip",
    "all": r"/rwd/bind/downloads/BindingDB_All_(\d{6})_tsv\.zip",
}


def resolve_latest_bindingdb_download(variant: str = "articles") -> tuple[str, str]:
    if variant not in VARIANT_PATTERNS:
        raise ValueError(f"Unsupported BindingDB variant: {variant}")

    with urlopen(DOWNLOAD_PAGE) as response:
        html = response.read().decode("utf-8", errors="ignore")

    matches = re.findall(VARIANT_PATTERNS[variant], html)
    if not matches:
        raise RuntimeError(f"Could not find a BindingDB TSV download for variant={variant}")

    latest_stamp = sorted(matches)[-1]
    relative_path = re.search(
        VARIANT_PATTERNS[variant].replace(r"(\d{6})", latest_stamp),
        html,
    )
    if relative_path is None:
        relative = f"/rwd/bind/downloads/BindingDB_BindingDB_Articles_{latest_stamp}_tsv.zip" if variant == "articles" else f"/rwd/bind/downloads/BindingDB_All_{latest_stamp}_tsv.zip"
    else:
        relative = relative_path.group(0)

    return latest_stamp, f"{DIRECT_BASE}{relative}"


def download_bindingdb_zip(variant: str = "articles", force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp, url = resolve_latest_bindingdb_download(variant=variant)
    filename = f"bindingdb_{variant}_{stamp}_tsv.zip"
    out_path = RAW_DIR / filename

    if out_path.exists() and not force:
        print(f"[INFO] Reusing existing BindingDB archive: {out_path}")
        return out_path

    print(f"[INFO] Downloading BindingDB {variant} archive from {url}")
    urlretrieve(url, out_path)
    print(f"[OK] Saved BindingDB archive: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the latest BindingDB TSV archive.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_PATTERNS),
        default="articles",
        help="BindingDB download variant to fetch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the archive already exists locally.",
    )
    args = parser.parse_args(argv)

    download_bindingdb_zip(variant=args.variant, force=args.force)


if __name__ == "__main__":
    main()
