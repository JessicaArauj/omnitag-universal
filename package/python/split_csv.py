"""Utility script to split large CSV/XLSX files into fixed-size chunks.

Example:
    python -m package.python.split_csv inputs/vigilancia.xlsx --output-dir inputs/chunks --chunk-size 500

CSV inputs generate CSV chunks, while XLS/XLSX inputs generate Excel chunks named
`<original>_part_XXX.xlsx`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def chunk_rows(reader: Iterable[List[str]], chunk_size: int):
    """Yield successive chunks from the CSV reader."""
    chunk: List[List[str]] = []
    for row in reader:
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def split_csv_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int = 500,
    encoding: str = 'utf-8',
    output_format: str = 'csv',
    fallback_encodings: Iterable[str] | None = None,
    delimiter: str | None = None,
) -> None:
    """Split the CSV into multiple files with the desired chunk size."""
    _ensure_positive_chunk(chunk_size)
    _ensure_input_exists(input_file)
    _ensure_output_dir(output_dir)

    normalized_format = output_format.lower()
    if normalized_format not in {'csv', 'xlsx'}:
        raise ValueError('output_format must be either "csv" or "xlsx".')
    extension = 'xlsx' if normalized_format == 'xlsx' else 'csv'
    if normalized_format == 'xlsx' and chunk_size > 1_048_575:
        raise ValueError(
            'chunk_size exceeds Excel row limit (1,048,575 data rows). '
            'Choose a smaller chunk_size when exporting XLSX chunks.',
        )

    delimiter = delimiter or None
    encodings_to_try = [encoding]
    if fallback_encodings:
        for fallback in fallback_encodings:
            fallback = fallback.strip()
            if fallback:
                encodings_to_try.append(fallback)

    csv_in = None
    reader = None
    header: List[str] | None = None
    used_encoding = None
    last_error: UnicodeDecodeError | None = None
    def _build_reader(file_handle):
        if delimiter:
            reader_obj = csv.reader(file_handle, delimiter=delimiter)
            return reader_obj, delimiter
        sample = file_handle.read(8192)
        file_handle.seek(0)
        if not sample:
            return csv.reader(file_handle), ','
        try:
            sniffed = csv.Sniffer().sniff(sample)
            reader_obj = csv.reader(file_handle, dialect=sniffed)
            return reader_obj, sniffed.delimiter
        except csv.Error:
            file_handle.seek(0)
            reader_obj = csv.reader(file_handle)
            return reader_obj, ','

    used_delimiter: str | None = delimiter
    for candidate in encodings_to_try:
        try:
            csv_in = input_file.open('r', encoding=candidate, newline='')
            reader, detected_delimiter = _build_reader(csv_in)
            header = next(reader)
            used_encoding = candidate
            used_delimiter = detected_delimiter
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            if csv_in:
                csv_in.close()
            csv_in = None
            reader = None
        except StopIteration:
            if csv_in:
                csv_in.close()
            print('Input CSV is empty; no files were generated.')
            return

    if reader is None or header is None or csv_in is None:
        if last_error:
            raise last_error
        raise UnicodeError(
            f'Unable to decode {input_file} with encodings: {encodings_to_try}',
        )

    try:
        for index, chunk in enumerate(chunk_rows(reader, chunk_size), start=1):
            output_path = output_dir / f'{input_file.stem}_part_{index:03d}.{extension}'
            if normalized_format == 'csv':
                with output_path.open('w', encoding=used_encoding or encoding, newline='') as csv_out:
                    writer = csv.writer(csv_out, delimiter=used_delimiter or ',')
                    writer.writerow(header)
                    writer.writerows(chunk)
            else:
                df = pd.DataFrame(chunk, columns=header)
                df.to_excel(output_path, index=False)
            print(
                f'Created {output_path} with {len(chunk)} rows.'
                f' (format: {extension}, encoding: {used_encoding}, '
                f'delimiter: {used_delimiter})'
            )
    finally:
        csv_in.close()


def split_excel_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int = 500,
) -> None:
    """Split an Excel file into XLSX chunks."""
    _ensure_positive_chunk(chunk_size)
    _ensure_input_exists(input_file)
    _ensure_output_dir(output_dir)

    df = pd.read_excel(input_file)
    if df.empty:
        print('Input spreadsheet is empty; no files were generated.')
        return

    total_rows = len(df)
    chunk_index = 1
    for start in range(0, total_rows, chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end]
        output_path = output_dir / f'{input_file.stem}_part_{chunk_index:03d}.xlsx'
        chunk.to_excel(output_path, index=False)
        print(f'Created {output_path} with {len(chunk)} rows.')
        chunk_index += 1


def _ensure_positive_chunk(chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer.')


def _ensure_input_exists(input_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Split a CSV file into CSV chunks or an XLSX file into XLSX chunks.',
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to the original CSV/XLSX file.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('inputs'),
        help='Directory where the chunked files will be stored.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Number of rows (excluding header) per file. Default: 500.',
    )
    parser.add_argument(
        '--encoding',
        default='utf-8',
        help='Encoding used to read/write the CSV. Default: utf-8.',
    )
    parser.add_argument(
        '--fallback-encodings',
        default='latin-1,cp1252',
        help=(
            'Comma-separated list of fallback encodings to try when reading CSV inputs. '
            'Leave empty to disable. Default: "latin-1,cp1252".'
        ),
    )
    parser.add_argument(
        '--output-format',
        choices={'auto', 'csv', 'xlsx'},
        default='auto',
        help=(
            'Desired chunk format for CSV inputs. '
            '"auto" keeps CSV chunks, "xlsx" converts each chunk to Excel. '
            'Ignored for Excel inputs. Default: auto.'
        ),
    )
    parser.add_argument(
        '--csv-delimiter',
        default='auto',
        help=(
            'Delimiter to use when parsing CSV inputs. '
            'Set to "auto" (default) to sniff it from the file.'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.input_file.suffix.lower()
    if suffix == '.csv':
        format_choice = args.output_format.lower()
        if format_choice == 'auto':
            format_choice = 'csv'
        fallback_encodings = [
            enc.strip()
            for enc in args.fallback_encodings.split(',')
            if enc.strip()
        ]
        split_csv_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            encoding=args.encoding,
            output_format=format_choice,
            fallback_encodings=fallback_encodings,
            delimiter=None
            if args.csv_delimiter.lower() == 'auto'
            else args.csv_delimiter,
        )
    elif suffix in {'.xlsx', '.xls'}:
        if args.output_format == 'csv':
            raise ValueError('Excel inputs can only generate XLSX chunks.')
        split_excel_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
        )
    else:
        raise ValueError(
            f'Unsupported file type "{suffix}". Provide a CSV or XLS/XLSX file.'
        )


if __name__ == '__main__':
    main()
