from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.pipelines.reproducibility import ensure_standard_reproducibility_files, export_reproducibility_manifest, write_manifest


def _resolve(path_str: str | None) -> Path:
    if not path_str:
        return PROJECT_ROOT
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Initialize or verify reproducibility artifacts for OncoForge.")
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT), help="Repository root to operate on.")
    parser.add_argument("--check-only", action="store_true", help="Only report the expected artifacts without creating anything new.")
    parser.add_argument("--emit-manifest", action="store_true", help="Write reproducibility_manifest.json at the root.")
    args = parser.parse_args(argv)

    root = _resolve(args.root)
    expected = export_reproducibility_manifest(root)

    if args.check_only:
        print(json.dumps(expected, indent=2))
        return

    created = ensure_standard_reproducibility_files(root)
    print(f"[OK] Reproducibility artifacts present under: {root}")
    if created:
        print("[OK] Created:")
        for path in created:
            print(f" - {path}")
    else:
        print("[OK] No missing artifacts were created.")

    if args.emit_manifest:
        manifest_path = write_manifest(root=root)
        print(f"[OK] Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
