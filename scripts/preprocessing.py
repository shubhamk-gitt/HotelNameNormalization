# preprocessing.py

import pandas as pd

HOTEL_TEXT_COLUMNS = [
    "hotel_name",
    "address_line1",
    "city",
    "state",
    "country",
    "postal_code",
    "phone_number",
]


def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "record_id" not in df.columns:
        raise ValueError("CSV must contain 'record_id' column.")
    df = df.reset_index(drop=True)
    return df


def build_hotel_text(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in HOTEL_TEXT_COLUMNS if c in df.columns]
    df["hotel_text"] = df[cols].agg(" | ".join, axis=1).str.strip()
    df["hotel_text"] = df["hotel_text"].str.replace(r"\s+\|\s+", " | ", regex=True)
    return df


def compute_block_key(postal_code: str) -> str:
    if postal_code is None:
        return ""
    s = str(postal_code)
    s = s.strip().lower()
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def add_block_key(df: pd.DataFrame) -> pd.DataFrame:
    if "postal_code" not in df.columns:
        df["postal_code"] = ""
    df["block_key"] = df["postal_code"].apply(compute_block_key)
    return df
