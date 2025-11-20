from pathlib import Path

import pandas as pd

# === CONFIGURATION ===

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'mapping_col.xlsx'

def get_columns_from_excel(file_path):
    """Return a list of sheets and their columns from an Excel file."""
    mapping = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, nrows=2)
            columns = list(df.columns)
            mapping.append(
                {
                    'file': file_path.name,
                    'sheet': sheet,
                    'columns': columns,
                }
            )
    except Exception as exc:  # noqa: BLE001 - logging the raw error
        print(f'Error reading {file_path}: {exc}')
    return mapping


def get_columns_from_csv(file_path):
    """Return columns from a CSV file."""
    try:
        df = pd.read_csv(file_path, nrows=2)
        return [
            {
                'file': file_path.name,
                'sheet': '-',
                'columns': list(df.columns),
            }
        ]
    except Exception as exc:  # noqa: BLE001 - logging the raw error
        print(f'Error reading {file_path}: {exc}')
        return []


def map_all_input_columns():
    """Map columns of all Excel/CSV files in the input folder."""
    if not INPUT_DIR.exists():
        print(f'Input directory not found: {INPUT_DIR}')
        return

    all_mappings = []

    for file_path in INPUT_DIR.iterdir():
        if file_path.suffix.lower() in {'.xlsx', '.xls'}:
            all_mappings.extend(get_columns_from_excel(file_path))
        elif file_path.suffix.lower() == '.csv':
            all_mappings.extend(get_columns_from_csv(file_path))
        else:
            print(f'Skipping unsupported file: {file_path.name}')

    if not all_mappings:
        print('No valid files found in the input directory.')
        return

    # Build a dataframe summarizing results
    rows = []
    for mapping in all_mappings:
        for column in mapping['columns']:
            rows.append(
                {
                    'File': mapping['file'],
                    'Sheet': mapping['sheet'],
                    'Column': column,
                }
            )

    df_output = pd.DataFrame(rows)
    df_output.to_excel(OUTPUT_FILE, index=False)
    print(f'Column mapping report successfully generated: {OUTPUT_FILE}')


if __name__ == '__main__':
    map_all_input_columns()
