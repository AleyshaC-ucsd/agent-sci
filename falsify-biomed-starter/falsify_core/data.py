"""Strict, allowlisted dataset access. No arbitrary file paths from the agent."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / 'datasets' / 'catalog.json'


def catalog() -> dict:
    if not CATALOG_PATH.exists():
        raise RuntimeError('Datasets not generated. Run: python generate_data.py')
    with CATALOG_PATH.open() as handle:
        return json.load(handle)


def read_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    entries = catalog()
    if dataset_id not in entries:
        raise ValueError(f'Unknown dataset ID: {dataset_id!r}. Available: {list(entries)}')
    path = ROOT / 'datasets' / 'observed' / f'{dataset_id}.csv'
    df = pd.read_csv(path)
    return df, entries[dataset_id]
