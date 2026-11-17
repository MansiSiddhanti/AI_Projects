"""Backward-compatible entry point. Prefer: python -m rag ingest"""

import sys

from rag.cli import main

if __name__ == "__main__":
    sys.exit(main(["ingest", *sys.argv[1:]]))
