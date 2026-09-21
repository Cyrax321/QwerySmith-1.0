#!/usr/bin/env python3
"""
make_overleaf_zip.py -- Packages QwerySmith research paper, figures, and tables into an Overleaf-ready ZIP.

Usage:
    python make_overleaf_zip.py
    python make_overleaf_zip.py --paper-dir paper --out-zip qwerysmith_paper_overleaf.zip
"""
import argparse
import sys
from pathlib import Path

# Ensure paper_eval can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import paper_eval
except ImportError:
    print("[Warning] paper_eval.py not found locally; fetching from GitHub...")
    import urllib.request
    url = "https://raw.githubusercontent.com/Cyrax321/QwerySmith-1.0/main/paper_eval.py"
    urllib.request.urlretrieve(url, "paper_eval.py")
    import paper_eval


def main():
    parser = argparse.ArgumentParser(description="Package QwerySmith research paper for Overleaf.")
    parser.add_argument("--paper-dir", default="paper", help="Directory containing main.tex, figures/, and tables/.")
    parser.add_argument("--out-zip", default="qwerysmith_paper_overleaf.zip", help="Destination ZIP filename.")
    parser.add_argument("--no-download", action="store_true", help="Disable automatic Colab browser download.")
    args = parser.parse_args()

    zip_path = paper_eval.create_overleaf_package(
        paper_dir=args.paper_dir,
        output_zip=args.out_zip,
        download_in_colab=not args.no_download,
    )
    print(f"[Done] Finished! Ready for Overleaf: {zip_path}")


if __name__ == "__main__":
    main()
