"""Batch runner that processes every XLSX chunk under inputs/chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .aut import (
    DEFAULT_CHUNK_OUTPUT_DIR,
    DEFAULT_CHUNKS_DIR,
    process_chunk_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Process every XLSX chunk under inputs/chunks.',
    )
    parser.add_argument(
        '--chunk-dir',
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help='Directory containing the chunked XLSX files. Default: inputs/chunks.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_CHUNK_OUTPUT_DIR,
        help='Directory where tagged outputs per chunk will be stored. '
        'Default: output/chunks.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_chunk_directory(args.chunk_dir, args.output_dir)


if __name__ == '__main__':
    main()
