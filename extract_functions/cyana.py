import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# CYANA / Xeasy Export Functions
# =============================================================================

def export_cyana_seq(sequences_dict, output_dir="results/cyana_export"):
    """
    Exports protein sequence to CYANA .seq format.
    Format: <RES_NAME> <SEQ_NUM>
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, df in sequences_dict.items():
        filename = out_path / f"{name}.seq"
        with open(filename, "w") as f:
            for _, row in df.iterrows():
                res = str(row.get('residue_name', '')).strip()
                num = str(row.get('sequence_code', '')).strip()
                if res and num and num != "nan":
                    f.write(f"{res} {num}\n")

    print(f"✅ Exported {len(sequences_dict)} .seq files to {output_dir}/")


def get_cyana_atom_name(atom_name, residue_name):
    """
    Maps NEF/standard atom names to CYANA/XEASY nomenclature.
    Applies special pseudoatom mapping for Valine and Leucine.
    """
    atom_map = {
        "H": "H", "N": "N", "C": "C", "CA": "CA", "CB": "CB", "ND2": "ND2", "ND1": "ND1",
        "NE2": "NE2", "CG": "CG", "CG1": "CG1", "CG2": "CG2", "CGx": "CG1", "CGy": "CG2",
        "CG%": "CG1", "CD": "CD", "CD1": "CD1", "CD2": "CD2", "CDx": "CD1", "CDy": "CD2",
        "CD%": "CD1", "CE": "CE", "CE1": "CE1", "CE2": "CE2", "CEx": "CE1", "CEy": "CE2",
        "CE%": "CE1", "HA": "HA", "HA%": "QA", "HAx": "HA2", "HAy": "HA3", "HB": "HB",
        "HBx": "HB2", "HBy": "HB3", "HB%": "QB", "HG": "HG", "HGx": "HG2", "HGy": "HG3",
        "HGx%": "QG1", "HGy%": "QG2", "HG%": "QG", "HG1x": "HG12", "HG1y": "HG13",
        "HG1%": "QG1", "HG2%": "QG2", "HDx": "HD2", "HDy": "HD3", "HDx%": "QD1",
        "HDy%": "QD2", "HD%": "QD", "HD1x": "HD12", "HD1": "HD1", "HD1y": "HD13",
        "HD1%": "QD1", "HD2%": "QD2", "HD2x": "HD21", "HD2y": "HD22", "HEx": "HE2",
        "HEy": "HE3", "HEx%": "QE1", "HEy%": "QE2", "HE%": "QE", "HE1x": "HE12",
        "HE1y": "HE13", "HE1%": "QE1", "HE2%": "QE2", "HE2x": "HE21", "HE2y": "HE22"
    }

    atom1_map = {
        "H": "H", "N": "N", "C": "C", "CA": "CA", "CB": "CB", "CG": "CG", "CG1": "CG1",
        "CG2": "CG2", "CGx": "CG1", "CGy": "CG2", "CG%": "CG1", "CD": "CD", "CD1": "CD1",
        "CD2": "CD2", "CDx": "CD1", "CDy": "CD2", "CD%": "CD1", "CE": "CE", "CE1": "CE1",
        "CE2": "CE2", "CEx": "CE1", "CEy": "CE2", "CE%": "CE1", "HA": "HA", "HA%": "QA",
        "HAx": "HA2", "HAy": "HA3", "HB": "HB", "HBx": "HB2", "HBy": "HB3", "HB%": "QB",
        "HG": "HG", "HGx": "HG2", "HGy": "HG3", "HGx%": "QG1", "HGy%": "QG2", "HG%": "QQG",
        "HDx": "HD2", "HDy": "HD3", "HDx%": "QD1", "HDy%": "QD2", "HD%": "QQD",
        "HD1x": "HD12", "HD1y": "HD13", "HD1%": "QD1", "HD2%": "QD2", "HEx": "HE2",
        "HEy": "HE3", "HEx%": "QE1", "HEy%": "QE2", "HE%": "QE", "HE1x": "HE12",
        "HE1y": "HE13", "HE1%": "QE1", "HE2%": "QE2"
    }

    res = str(residue_name).upper()
    atm = str(atom_name)

    if res in ["VAL", "LEU"]:
        return atom1_map.get(atm, atm)
    return atom_map.get(atm, atm)


def export_cyana_prot(shifts_dict, output_dir="results/cyana_export"):
    """
    Simplified CYANA .prot exporter using Pandas-native logic.
    1. Removes any row containing '@'.
    2. Evaluates math strings (like '3-1' -> 2).
    3. Merges special/negative shifts only if (SeqNum, Atom) is missing.
    4. Uses get_cyana_atom_name() for nomenclature mapping.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, df in shifts_dict.items():
        # 1. Row-wise cleaning: Remove any row where sequence_code starts '@'
        df = df[~df['sequence_code'].astype(str).str.startswith('@', na=False)].copy()

        # 2. Transform Sequence Codes: Evaluate math like '3-1'
        def safe_eval_seq(val):
            val = str(val).strip()
            if not val or val.lower() == 'nan':
                return np.nan
            # Safely handle '3-1' i-1 nomenclature without eval()
            if '-' in val and not val.startswith('-'):
                try:
                    parts = val.split('-')
                    return int(float(parts[0])) - int(float(parts[1]))
                except ValueError:
                    return np.nan
            try:
                return int(float(val))
            except ValueError:
                return np.nan

        df['seq_num'] = df['sequence_code'].apply(safe_eval_seq)
        df = df.dropna(subset=['seq_num'])

        # 3. Transform Atom Names: Apply CYANA mapping via helper
        def map_atom(row):
            res = str(row.get('residue_name', ''))
            atm = str(row.get('atom_name', ''))
            return get_cyana_atom_name(atm, res)

        df['cyana_atom'] = df.apply(map_atom, axis=1)

        # 4. Conflict Logic: Concat standard (>0) first, then fill gaps with special
        # Ensure 'i-1' math strings (like '3-1') are explicitly routed to the special dataframe
        is_math_string = df['sequence_code'].astype(str).str.contains(
            '-', na=False) & ~df['sequence_code'].astype(str).str.startswith('-', na=False)

        standard = df[(df['seq_num'] > 0) & ~is_math_string].copy()
        special = df[~df.index.isin(standard.index)].copy()

        # This drop_duplicates logic handles the (2 CA) vs (3-1 CA) conflict
        merged = pd.concat([standard, special]).drop_duplicates(subset=['seq_num', 'cyana_atom'])

        # 5. Define final string output format
        def format_line(row, idx):
            # Safe extraction using .get() to prevent KeyErrors if a column is entirely missing
            shift_val = row.get('value', 0.0)
            shift = f"{float(shift_val):10.3f}"

            # Check for value_err safely
            err_val = row.get('value_uncertainty', np.nan)
            if pd.notna(err_val) and str(err_val).strip() != "":
                err = f"{float(err_val):8.3f}"
            else:
                err = "   0.000"

            atom_name = f"{row.get('cyana_atom', ''):6}"
            seq = f"{int(row.get('seq_num', 0)):10}"

            return f"{str(idx+1):5} {shift} {err} {atom_name} {seq}\n"

        # 6. Write to file
        output_lines = [format_line(row, i) for i, (_, row) in enumerate(merged.iterrows())]

        with open(out_path / f"{name}.prot", "w") as f:
            f.writelines(output_lines)

    print(f"✅ Exported {len(shifts_dict)} .prot files to {output_dir}/")


def generate_xeasy_header(spectrum_type, dimensions):
    """
    Generates the XEASY header dynamically while preserving column order.
    Naming Rules:
      - 1D/2D: Uses base atom names (H, N, C). Homonuclear 2D uses H1, H2.
      - 3D+: Uses specific names (HN, HC) to distinguish directly attached protons.
      - Special Case: Two carbons are assigned C1, C2.
    """
    num_dimensions = len(dimensions)
    if num_dimensions == 0:
        return f"# Number of dimensions 0\n#FORMAT xeasy0D\n#SPECTRUM {spectrum_type}\n"

    # 1. Standardize base labels (strip isotopes)
    labels = [str(dim).upper().replace('15N', 'N').replace(
        '13C', 'C').replace('1H', 'H') for dim in dimensions]

    # 2. Identify indices of specific atoms
    hydrogen_indices = [i for i, lbl in enumerate(labels) if lbl == 'H']
    carbon_indices = [i for i, lbl in enumerate(labels) if lbl == 'C']

    has_nitrogen = 'N' in labels
    has_carbon = len(carbon_indices) > 0

    # Explicitly define our dimensionality threshold for HN/HC naming
    is_3d_or_higher = num_dimensions >= 3

    # 3. Apply assignment logic
    if len(carbon_indices) == 2:
        labels[carbon_indices[0]] = 'C2'
        labels[carbon_indices[1]] = 'C1'

        if len(hydrogen_indices) == 1:
            labels[hydrogen_indices[0]] = 'H1'
        elif len(hydrogen_indices) >= 2:
            labels[hydrogen_indices[0]] = 'H1'
            labels[hydrogen_indices[1]] = 'H2'

    else:
        if len(hydrogen_indices) == 2:
            first_h_idx = hydrogen_indices[0]
            second_h_idx = hydrogen_indices[1]

            if is_3d_or_higher:
                labels[second_h_idx] = 'H'
                if has_nitrogen:
                    labels[first_h_idx] = 'HN'
                elif has_carbon:
                    labels[first_h_idx] = 'HC'
                else:
                    labels[first_h_idx] = 'H1'
                    labels[second_h_idx] = 'H2'
            else:
                labels[first_h_idx] = 'H1'
                labels[second_h_idx] = 'H2'

        elif len(hydrogen_indices) == 1:
            h_idx = hydrogen_indices[0]

            if is_3d_or_higher:
                if has_nitrogen:
                    labels[h_idx] = 'HN'
                elif has_carbon:
                    labels[h_idx] = 'HC'
                else:
                    labels[h_idx] = 'H'
            else:
                labels[h_idx] = 'H'

    # 4. Build the header string sequentially
    iname_lines = "".join(f"#INAME {i+1} {lbl}\n" for i, lbl in enumerate(labels))
    header = (f"# Number of dimensions {num_dimensions}\n"
              f"#FORMAT xeasy{num_dimensions}D\n"
              f"{iname_lines}#SPECTRUM {spectrum_type} {' '.join(labels)}\n")

    return header


def export_cyana_peaks(peak_dict, out_path="results/cyana_export"):
    """
    Exports peak dictionaries to CYANA / Xeasy format.
    Preserves the exact column order from the input data.
    """
    output_dir = Path(out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    types_map = {
        "HNCO":             "HNCO",
        "HNcaCO":           "HNcaCO",
        "HNCA":             "HNCA",
        "HNcoCA":           "HNcoCA",
        "HNCA/CB":          "CBCANH",
        "HNcoCA/CB":        "CBCAcoNH",
        "15N HSQC/HMQC":    "N15HSQC",
        "13C HSQC/HMQC":    "C13HSQC",
        "15N NOESY-HSQC":   "N15NOESY",
        "13C NOESY-HSQC":   "C13NOESY",
        "HBcb/HAcacoNH":    "HBHAcoNH",
        "hCCH-TOCSY":       "CCHTOCSY",
        "HcCH-TOCSY":       "HCCHTOCSY",
    }

    def get_assignment(row, dim):
        """Extracts and maps sequence/atom info sequentially."""
        # Cleaned up: No more 'dim == 1' hack needed!
        seq_col = f'sequence_code_{dim}'
        res_col = f'residue_name_{dim}'
        atm_col = f'atom_name_{dim}'

        seq = row.get(seq_col, np.nan)
        res = row.get(res_col, '')
        atm = row.get(atm_col, '')

        if pd.isna(seq) or not atm or str(atm) == 'nan' or str(seq).startswith('@'):
            return "0"

        try:
            seq_num = int(eval(str(seq))) if '-' in str(seq) else int(float(seq))
            mapped_atm = get_cyana_atom_name(atm, res)
            return f"{mapped_atm}.{seq_num}"
        except Exception:
            return "0"

    for name, df in peak_dict.items():
        if df.empty:
            continue

        exp_type = df.attrs.get('experiment_type', 'Unknown')
        dims = df.attrs.get('dimensions', [])
        num_dims = len(dims)

        if exp_type not in types_map:
            print(f"⚠️  Warning: Spectrum '{name}' has an unmapped experiment type: '{exp_type}'")

        spec_type = types_map.get(exp_type, exp_type).replace("'", "")

        header = generate_xeasy_header(spec_type, dims)
        dim_indices = tuple(range(1, num_dims + 1))
        id_pad = 11 if num_dims == 3 else 7

        lines = []
        for idx, row in df.iterrows():
            pid = str(row.get('id', idx + 1)).ljust(id_pad)
            vol = row.get('volume', row.get('height', 0))
            vol_str = f"{float(vol):.9e}".ljust(21) if pd.notna(
                vol) else "0.000000000e+00".ljust(21)

            pos_str = "".join(f"{float(row.get(f'position_{i}', 0)):.4f}".ljust(11)
                              for i in dim_indices)
            assign_str = "".join(get_assignment(row, i).ljust(10) for i in dim_indices)

            line = f"{pid}{pos_str}1 U   {vol_str} 0.000000E+00 e 0  {assign_str}".strip()
            lines.append(line)

        file_path = output_dir / f"{name.replace(' ', '_').replace('/', '_')}.peaks"
        with open(file_path, "w") as f:
            f.write(header)
            if lines:
                f.write("\n".join(lines) + "\n")

    print(f"✅ Exported {len(peak_dict)} .peaks files sequentially to {out_path}/")
