import re

import numpy as np
import pandas as pd

# =============================================================================
# Shared helper functions (used by parsing, csp, spectra_plot, cyana, ...)
# =============================================================================

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

UNASSIGNED = "unassigned"


def force_numeric(column):
    """Converts a column to numbers, unless this would destroy valid strings like '3-1'."""
    converted = pd.to_numeric(column, errors="coerce")

    # If coercing introduces NEW NaNs, keep the original column.
    if converted.isna().sum() > column.isna().sum():
        return column

    return converted


def get_one_letter(res_name):
    """Converts 3-letter amino acid codes to 1-letter codes."""
    if pd.isna(res_name):
        return ""

    res = str(res_name).strip().upper()

    # If already one-letter, return as-is
    if len(res) == 1:
        return res

    return THREE_TO_ONE.get(res, res)


def is_unassigned(res_name):
    """True if a residue name is missing or the 'unassigned' placeholder."""
    return pd.isna(res_name) or str(res_name).strip().lower() in ("", UNASSIGNED)


def residue_label(res_name, seq_code):
    """
    Builds a short residue label, e.g. 'A24'.
    Unassigned peaks only get their sequence code / peak number (e.g. '@12' or '57').
    """
    seq = make_int_if_possible(seq_code)
    seq = "" if pd.isna(seq) else str(seq)
    if is_unassigned(res_name):
        return seq
    return f"{get_one_letter(res_name)}{seq}"


def make_int_if_possible(val):
    """Converts sequence codes like 12.0 or '12' to int 12, everything else to str."""
    if pd.isna(val) or val == "":
        return val
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return str(val)


def parse_sequence_code(val):
    """
    Converts a NEF sequence code to an integer residue number.

    Handles plain numbers ('12', 12.0), negative numbers ('-3') and the
    CCPN i-1 nomenclature ('3-1' -> 2). Returns np.nan for everything else
    (e.g. '@12', '' or NaN). Never uses eval().
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if not val or val.lower() == "nan":
        return np.nan

    m = re.fullmatch(r"(-?\d+(?:\.0+)?)\s*-\s*(\d+)", val)
    if m:
        return int(float(m.group(1))) - int(m.group(2))

    try:
        return int(float(val))
    except ValueError:
        return np.nan


def natural_number(val):
    """First integer found in a value ('@12' -> 12), used for natural sorting. NaN if none."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    m = re.search(r"-?\d+", str(val))
    return float(m.group()) if m else np.nan


def natural_sort_order(index_df, chain_col="chain_code", seq_col="sequence_code"):
    """
    Returns the row order (positional) that sorts residues by chain, then
    assigned before unassigned, then numerically ('@2' before '@10').
    """
    sort_df = index_df.reset_index(drop=True).copy()
    sort_df["_seq_num"] = pd.to_numeric(sort_df[seq_col], errors="coerce")
    sort_df["_is_unassigned"] = sort_df["_seq_num"].isna()
    sort_df["_nat_num"] = sort_df[seq_col].apply(natural_number)
    sort_df["_seq_str"] = sort_df[seq_col].astype(str)
    sort_df["_chain_str"] = sort_df[chain_col].astype(str)

    sort_df = sort_df.sort_values(
        by=["_chain_str", "_is_unassigned", "_seq_num", "_nat_num", "_seq_str"],
        kind="stable",
    )
    return sort_df.index.to_numpy()


META_IN = ["chain_code_1", "sequence_code_1", "residue_name_1"]
META_OUT = ["chain_code", "sequence_code", "residue_name"]


def index_peaklist(df, data_cols=None, name=None, warn_duplicates=True):
    """
    Converts a raw NEF peak list into a DataFrame indexed by
    (chain_code, sequence_code, residue_name).

    - Unassigned peaks get their peak_id as sequence code.
    - Sequence codes are cast to int where possible.
    - Only the first peak per residue is kept (a warning is printed).

    Args:
        df (pd.DataFrame): A peak list from extract_all_nef_data.
        data_cols (list, optional): Columns to keep. Defaults to
            ['peak_id', 'volume', 'height', 'position_1', 'position_2'].
        name (str, optional): Spectrum name, only used for warnings.
        warn_duplicates (bool): Print a warning if peaks were dropped.
    """
    if data_cols is None:
        data_cols = ["peak_id", "volume", "height", "position_1", "position_2"]

    temp_df = df.copy()

    # Standardize metadata
    for m in META_IN:
        if m not in temp_df.columns:
            temp_df[m] = pd.NA if m == "sequence_code_1" else UNASSIGNED
        elif m != "sequence_code_1":
            temp_df[m] = temp_df[m].fillna(UNASSIGNED)

    # Map unassigned peaks using the peak_id
    if "peak_id" in temp_df.columns:
        m = temp_df["sequence_code_1"].isna() | (temp_df["sequence_code_1"] == "")
        if m.any():
            temp_df["sequence_code_1"] = temp_df["sequence_code_1"].astype(object)
            temp_df.loc[m, "sequence_code_1"] = temp_df.loc[m, "peak_id"]

    # Drop rows with no sequence data
    temp_df = temp_df.dropna(subset=["sequence_code_1"])
    temp_df["sequence_code_1"] = temp_df["sequence_code_1"].apply(make_int_if_possible)

    temp_df = temp_df.set_index(META_IN)[[c for c in data_cols if c in temp_df.columns]]
    temp_df.index.names = META_OUT

    dup = temp_df.index.duplicated(keep="first")
    if dup.any() and warn_duplicates:
        label = f"'{name}'" if name else "Peak list"
        print(
            f"⚠️ {label}: {dup.sum()} additional peak(s) on an already used residue "
            "(e.g. side chains) – only the first peak per residue is kept."
        )
    return temp_df[~dup]
