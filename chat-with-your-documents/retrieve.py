"""Backward-compatible entry point. Prefer: python -m rag chat"""

import sys

from rag.cli import main

if __name__ == "__main__":
    sys.exit(main(["chat"]))
