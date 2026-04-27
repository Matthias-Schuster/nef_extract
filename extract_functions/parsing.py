import pynmrstar
import os
import contextlib
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# READ .nef Functions
# =============================================================================


def force_numeric(column):
    # Coerce turns un-convertible strings into NaNs
    converted = pd.to_numeric(column, errors='coerce')

    # If coercing introduces NEW NaNs (i.e., destroys valid string data like '3-1'),
    # keep the original column.
    if converted.isna().sum() > column.isna().sum():
        return column

    return converted


def extract_all_nef_data(filepath, report=False, spectra_plot=False):
    """
    Extracts and compiles sequences, chemical shift lists, and peak lists from an NEF file.

    Args:
        filepath (str or Path):
            The path to the input NMR Exchange Format (NEF) file.
        report (bool, optional):
            If True, prints a detailed summary of the NEF file's
            spectral architecture and contents. Defaults to False.
        spectra_plot (bool, optional):
            If True, generates an input configuration file of all 2D spectra tailored for
            the external NMR_2D_plot script. Defaults to False.

    Returns:
        tuple:
            A tuple containing three elements:
                - sequences (dict): Extracted sequence data.
                - shifts (dict): Extracted chemical shift lists.
                - peaks (dict): Extracted peak lists.
    """
    # Load the file
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull), contextlib.redirect_stdout(fnull):
            entry = pynmrstar.Entry.from_file(filepath)

    # Initialize dictionaries and lists for different data types
    sequences = {}
    shifts = {}
    peaks = {}
    spectra_plot_data = []

    if report:
        print("\n--Print all loops in the nef file:")
    for saveframe in entry:

        # 1. Extract saveframe-level metadata (Experiment type and Axis codes)
        spec_metadata = {}
        if saveframe.category == 'nef_nmr_spectrum':
            # Experiment Type
            exp_tags = saveframe.get_tag('_nef_nmr_spectrum.experiment_type')
            spec_metadata['experiment_type'] = exp_tags[0].strip("'") if exp_tags else "Unknown"

            # Dimensions / Axis Codes
            dim_loop = saveframe.get_loop_by_category('_nef_spectrum_dimension')
            if dim_loop:
                spec_metadata['dimensions'] = dim_loop.get_tag('_nef_spectrum_dimension.axis_code')
            else:
                spec_metadata['dimensions'] = []

            # --- Extract metadata for 2D spectra if spectra_plot=True ---
            if spectra_plot and len(spec_metadata['dimensions']) == 2:
                name = saveframe.name.replace('nef_nmr_spectrum_', '').split('`')[0]

                path_tags = saveframe.get_tag('_nef_nmr_spectrum.ccpn_spectrum_file_path')
                path = path_tags[0].strip("'\"") if path_tags else "Unknown"

                # Extract and format contour base
                contour_tags = saveframe.get_tag('_nef_nmr_spectrum.ccpn_positive_contour_base')
                contour = contour_tags[0].strip("'\"") if contour_tags else "Unknown"
                try:
                    # Format as scientific notation (e.g., 4e8)
                    # .0e gives '4e+08', replace cleans it up to '4e8'
                    contour = f"{float(contour):.0e}".replace('+0', '').replace('+', '')
                except (ValueError, TypeError):
                    pass

                color_tags = saveframe.get_tag('_nef_nmr_spectrum.ccpn_positive_contour_colour')
                color = color_tags[0].strip("'\"") if color_tags else "Unknown"

                spectra_plot_data.append([path, name, contour, color])

        # 2. Iterate through the loops inside the saveframe
        for loop in saveframe.loops:
            key = f"{saveframe.name}{loop.category}"
            if report:
                print(key)
            category = loop.category

            if loop.data:
                # Extract and clean DataFrame
                columns = [tag.split('.')[-1] for tag in loop.get_tag_names()]
                df = pd.DataFrame(loop.data, columns=columns).replace(".", np.nan)

                # Apply your working numeric conversion logic
                df = df.apply(force_numeric)

                if not df.empty:
                    # Sort into correct dictionary based on category
                    if category.endswith('_nef_sequence'):
                        sequences[saveframe.name] = df

                    elif category.endswith('_nef_chemical_shift'):
                        key = saveframe.name.replace('nef_chemical_shift_list_', '')
                        shifts[key] = df

                    elif category.endswith('_nef_peak'):
                        key = saveframe.name.replace(
                            'nef_nmr_spectrum_', '').replace('`', '_').strip('_')

                        # Attach the metadata (Experiment type/Dimensions) to the DataFrame
                        df.attrs = spec_metadata
                        peaks[key] = df

    # --- Save the extracted 2D metadata directly to a custom formatted file ---
    if spectra_plot and spectra_plot_data:
        output_csv = "results/spectra_metadata.txt"
        output_dir = Path(output_csv).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Find the common base directory
        valid_paths = [str(Path(r[0]).resolve()) for r in spectra_plot_data if r[0] != "Unknown"]
        if len(valid_paths) > 1:
            common_base = Path(os.path.commonpath(valid_paths))
        elif len(valid_paths) == 1:
            common_base = Path(valid_paths[0]).parent
        else:
            common_base = Path("")

        # 2. Pre-format the elements (make relative paths, add quotes)
        formatted_rows = []
        for row in spectra_plot_data:
            original_path = row[0]

            # Convert to relative path
            if original_path != "Unknown" and str(common_base) != ".":
                try:
                    rel_path = Path(original_path).resolve().relative_to(common_base)
                    rel_path_str = rel_path.as_posix()  # Ensure forward slashes
                except ValueError:
                    rel_path_str = original_path  # Fallback if it fails
            else:
                rel_path_str = original_path

            path_str = f"'{rel_path_str}'"
            name_str = f"'{row[1]}'"
            contour_str = str(row[2])
            color_str = f"'{row[3]}'"
            formatted_rows.append([path_str, name_str, contour_str, color_str])

        # 3. Find the maximum width for each of the first 3 columns
        max_w0 = max(len(r[0]) for r in formatted_rows)
        max_w1 = max(len(r[1]) for r in formatted_rows)
        max_w2 = max(len(r[2]) for r in formatted_rows)

        # 4. Write to file with dynamic padding and base_dir definition
        with open(output_csv, mode='w', encoding='utf-8') as file:

            # Write the import and the dynamic base_dir
            file.write("# directory of your NMR files\n")
            if str(common_base) and str(common_base) != ".":
                file.write(f"base_dir = Path('{common_base.as_posix()}')\n\n")
            else:
                file.write("base_dir = Path('')\n\n")

            # Write the formatted list
            file.write("data = [\n")
            for r in formatted_rows:
                col0 = (r[0] + ",").ljust(max_w0 + 1)
                col1 = (r[1] + ",").ljust(max_w1 + 1)
                col2 = (r[2] + ",").ljust(max_w2 + 1)
                col3 = r[3]

                line = f"    ({col0}    {col1}    {col2}    {col3}),\n"
                file.write(line)
            file.write("]\n")

        print(f"--Extracted metadata for {len(spectra_plot_data)
                                          } 2D spectra and saved to {output_csv}\n")

    if report:
        print("--Read complete!\n")
    print("--Following loops will be extracted!\n")
    for name in sequences:
        print(f"Sequence:\t {name}")
    for name in shifts:
        print(f"Shift List:\t {name}")
    for name in peaks:
        print(f"Peak List:\t {name}")

    if report:
        report_spectrum_architecture(peaks)

    return sequences, shifts, peaks


def generate_rename_template(peak_dict):
    """
    Prints a formatted dictionary template to the console.
    Copy and paste this into your main script to rename your spectra.
    """
    print("\n# --- Copy-paste this into your 'spec_rename_map' ---")

    if not peak_dict:
        print("# (No spectra found)")
        print("# ---------------------------------------------------\n")
        return

    # Find the maximum length of the spectrum names
    max_len = max(len(str(name)) for name in peak_dict.keys())

    for name in peak_dict.keys():

        # Create the left side:  'name':
        left_side = f"'{name}':"

        # Pad the left side with spaces.
        # max_len + 5 ensures enough room for the quotes, colon, and a couple of spaces.
        padded_left = left_side.ljust(max_len + 5)

        print(f"        {padded_left} '{name}',")

    print("# ---------------------------------------------------\n")


def print_spectrum_menu(peak_dict):
    """
    Displays the current spectrum mapping and provides
    the list 's' used for indexing in the analysis.
    """
    # 1. Create the list of keys (the 's' variable)
    s = list(peak_dict.keys())

    print("\n" + "="*25)
    print("      SPECTRUM MENU")
    print("="*25)

    for i, name in enumerate(s):
        print(f" {i:2} -> {name}")

    print("="*25)

    return s


def report_spectrum_architecture(peak_dict):
    """
    Diagnostic tool to print the dimensionality and atom mapping
    for every spectrum in the dataset using NEF metadata.
    """
    print("\n" + "="*40)
    print("      SPECTRUM ARCHITECTURE REPORT")
    print("="*40)

    for name, df in peak_dict.items():
        # Retrieve metadata stored in .attrs during extraction
        dims = df.attrs.get('dimensions', [])
        exp_type = df.attrs.get('experiment_type', 'Unknown')

        print(f"\nSpectrum: {name}")
        print(f"  Type:  {exp_type}")
        print(f"  Size:  {len(dims)}D")

        # This loop handles 2D or 3D dynamically
        if dims:
            for i, atom in enumerate(dims):
                # i+1 matches 'position_1', 'position_2', etc.
                print(f"  Axis {i+1}: {atom} (Mapped to 'position_{i+1}')")
        else:
            print("  ⚠️ No dimension metadata found for this spectrum.")

    print("\n" + "="*40)


def create_master_pivot(peaks_dict, ref_spectrum=None):
    """
    Combines 2D spectra into a master table and exports specific sub-tables to an Excel workbook.
    Automatically ensures that sequence codes are cast to integers where possible to
    maintain clean alignment across the dataset.

    Args:
        peaks_dict (dict):
            A dictionary containing the extracted peak lists to be combined.
        ref_spectrum (int or str, optional):
            The identifier for the reference spectrum to anchor the 2D spectra data
            (e.g., specifying the 15N or 13C HSQC). Defaults to None.

    Returns:
        pd.DataFrame:
            The compiled master pivot table containing the aligned 2D spectra.
    """

    meta_in = ['chain_code_1', 'sequence_code_1', 'residue_name_1']
    meta_out = ['chain_code', 'sequence_code', 'residue_name']
    data_cols = ['peak_id', 'volume', 'height', 'position_1', 'position_2']

    indexed_dfs = {}
    reference_dims = None
    reference_name = ""

    # --- Dimension Locking ---
    if ref_spectrum is not None:
        if ref_spectrum in peaks_dict:
            reference_name = ref_spectrum
            reference_dims = peaks_dict[ref_spectrum].attrs.get('dimensions', [])
            print(f"🎯 Locking dimensions to explicit reference: '{
                  reference_name}' {reference_dims}")
        else:
            print(f"⚠️ WARNING: '{ref_spectrum}' not found. Falling back to auto-detection.")

    # Helper function to convert sequence codes to integers where possible
    def make_int_if_possible(val):
        if pd.isna(val) or val == "":
            return val
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return str(val)

    for name, df in peaks_dict.items():
        dims = df.attrs.get('dimensions', [])
        if len(dims) != 2:
            continue

        if reference_dims is None:
            reference_dims = dims
            reference_name = name
        elif dims != reference_dims:
            continue

        temp_df = df.copy()

        # Standardize metadata efficiently
        for m in meta_in:
            if m not in temp_df.columns:
                temp_df[m] = pd.NA if m == 'sequence_code_1' else "unassigned"
            elif m != 'sequence_code_1':
                temp_df[m] = temp_df[m].fillna("unassigned")

        # Map unassigned peaks using the peak_id
        if 'peak_id' in temp_df.columns:
            m = temp_df['sequence_code_1'].isna() | (temp_df['sequence_code_1'] == "")
            if m.any():
                # Force the column to 'object' dtype so it safely accepts both numbers and strings
                temp_df['sequence_code_1'] = temp_df['sequence_code_1'].astype(object)

                # Now we can safely insert the peak_id, regardless of what type it is
                temp_df.loc[m, 'sequence_code_1'] = temp_df.loc[m, 'peak_id']

                cols = ['chain_code_1', 'residue_name_1']
                temp_df.loc[m, cols] = temp_df.loc[m, cols].fillna("unassigned")

        # Drop rows with no sequence data
        temp_df = temp_df.dropna(subset=['sequence_code_1'])

        # Apply the smart integer conversion
        temp_df['sequence_code_1'] = temp_df['sequence_code_1'].apply(make_int_if_possible)

        temp_df = temp_df.set_index(meta_in)[[c for c in data_cols if c in temp_df.columns]]
        temp_df.index.names = meta_out
        temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
        indexed_dfs[name] = temp_df

    if not indexed_dfs:
        print("❌ ERROR: No compatible 2D spectra were found.")
        return None

    # --- Combine ---
    table = pd.concat(indexed_dfs, axis=1)

    # --- Smart Sorting (Warning-Free & Mixed-Type Safe) ---
    sort_df = table.index.to_frame(index=False)
    sort_df['seq_num'] = pd.to_numeric(sort_df['sequence_code'], errors='coerce')
    sort_df['is_unassigned'] = sort_df['seq_num'].isna()

    # Create a temporary string column just to prevent pandas crashing
    # when it tries to alphabetically compare an int to a str in the final tie-breaker
    sort_df['seq_str'] = sort_df['sequence_code'].astype(str)

    # Sort and reorder the main table
    sort_df = sort_df.sort_values(by=['chain_code', 'is_unassigned', 'seq_num', 'seq_str'])
    table = table.iloc[sort_df.index]

    # --- Reorder internal columns logically ---
    preferred_order = ['peak_id', 'position_1', 'position_2', 'height', 'volume']
    existing_levels = table.columns.get_level_values(1).unique()
    new_order = [c for c in preferred_order if c in existing_levels]
    table = table.reindex(columns=new_order, level=1)

    table.attrs['dimensions'] = reference_dims

    # --- EXPORT SECTION: Single Workbook, Multiple Sheets ---
    excel_dir = Path("results")
    excel_dir.mkdir(parents=True, exist_ok=True)
    file_path = excel_dir / "spectra_analysis.xlsx"

    # Open the Excel writer once to save I/O time
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        table.to_excel(writer, sheet_name="Master_Pivot")

        if 'height' in existing_levels:
            table.xs('height', axis=1, level=1).to_excel(writer, sheet_name="Heights_Only")

        if 'volume' in existing_levels:
            table.xs('volume', axis=1, level=1).to_excel(writer, sheet_name="Volumes_Only")

        pos_cols = [c for c in ['position_1', 'position_2'] if c in existing_levels]
        if pos_cols:
            table.reindex(columns=pos_cols, level=1).to_excel(writer, sheet_name="Positions_Only")

    print(f"✅ Exported all tables as sheets to {file_path}\n")
    return table


def normalize_series(series, normalization_factor=1):
    """
    Normalizes series from 0 to the p-th percentile.
    Values above the p-th percentile are capped at 1.0.
    """
    threshold = series.quantile(normalization_factor)

    # If the threshold is 0 (or all data is 0), return 0s to avoid division by zero
    if pd.isna(threshold) or threshold <= 0:
        return series * 0.0

    # Standardize: (Value / Threshold) capped at 1.0
    return (series / threshold).clip(0, 1)


def align_to_full_sequence(pivot_df, sequence_df):
    """
    Creates a full protein skeleton and attaches NMR data.
    """
    skeleton = (
        sequence_df[['chain_code', 'sequence_code', 'residue_name']]
        .copy()
        .set_index(['chain_code', 'sequence_code', 'residue_name'])
        .sort_index()
    )

    # Fix the "MergeError" by giving the skeleton a MultiIndex header
    skeleton.columns = pd.MultiIndex.from_product(
        [skeleton.columns, ['']],
        names=pivot_df.columns.names
    )

    # Left Merge preserves sequence skeleton, filling missing NMR data with NaN
    return pd.merge(skeleton, pivot_df, left_index=True, right_index=True, how='left')


def get_one_letter(res_name):
    """Converts 3-letter amino acid codes to 1-letter codes."""
    if pd.isna(res_name):
        return ""
    mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    res = str(res_name).upper()
    return mapping.get(res, res)


def add_analysis_to_master(project_data,
                           ref_spectra, spectra_to_analyze,
                           align=False, seq_index=0,
                           alpha=None, normalization_factor=None,
                           scaling_factors=None, smoothing_window=None, filename=None):
    """
    Calculates Chemical Shift Perturbations (CSPs) and intensity/volume ratios.

    Automatically scales CSP calculations based on nuclei type (e.g., 15N or 13C)
    using NEF dimension metadata unless an alpha value is explicitly provided.

    Args:
        project_data (dict, str, or Path):
            A dictionary containing 'pivot', 'sequences',
            and 'spectra'. Alternatively, a path/string pointing to a modified Excel file.
        ref_spectra (int or str):
            The reference spectrum for the analysis.
        spectra_to_analyze (int, str, or list):
            The spectra to analyze against the
            reference. Can be a single name/index or a list of names/indices.
        align (bool, optional):
            Whether to align the NMR data against a full protein
            sequence skeleton from the NEF file. Defaults to False.
        seq_index (int, optional):
            The index of the sequence in the NEF dictionary
            to align to. Defaults to 0.
        alpha (float, optional):
            Manual scaling factor for the CSP calculation
            (e.g., 0.142 for 15N). If None, it is auto-detected from metadata.
        normalization_factor (float, optional):
            Quantile threshold to normalize
            intensities against (e.g., 0.95), minimizing errors from peak overlap.
        scaling_factors (float, list, or dict, optional):
            Multipliers to scale intensities (e.g., for different numbers of scans).
            Can be a single value, a list matching `spectra_to_analyze`,
            or a spectrum-to-value dictionary.
        smoothing_window (int, optional):
            Window size for rolling average of intensities across neighboring residues.
            Good for PRE analyses.
        filename (str, optional):
            Filename for the exported Excel table. If provided, saves to the 'results/' directory.

    Returns:
        pd.DataFrame:
            The final compiled DataFrame containing metadata, coordinates,
            CSPs, and calculated ratios, optionally aligned to the full sequence.
    """

    # --- 1. Normalize Input ---
    # If a string/path is passed, immediately wrap it in the expected dictionary structure
    if isinstance(project_data, (str, Path)):
        project_data = {'pivot': project_data, 'sequences': {}, 'spectra': None}

    # --- 2. Unpack the Dictionary ---
    # At this point, project_data is guaranteed to be a dictionary
    pivot_data = project_data.get('pivot')
    sequences_dict = project_data.get('sequences', {})
    spectrum_list = project_data.get('spectra')

    # --- 3. Resolve the Pivot Data ---
    # If the pivot data is a path, load it. Otherwise, assume it's already a DataFrame.
    if isinstance(pivot_data, (str, Path)):
        pivot_path = Path(pivot_data)
        print(f"📁 Loading curated pivot from Excel: {pivot_path.name}")
        pivot_df = pd.read_excel(pivot_path, sheet_name=0, header=[0, 1], index_col=[0, 1, 2])
    else:
        pivot_df = pivot_data

    # --- 4. Final Safety Check ---
    if pivot_df is None or not isinstance(pivot_df, pd.DataFrame):
        raise ValueError(
            "❌ ERROR: 'project_data' must contain a valid 'pivot' dataframe or Excel path.")
    # -----------------------------

    # --- SMART INDEX LOGIC: Convert integers to spectrum names ---
    def resolve_spec(spec):
        # Catch both standard Python ints and NumPy ints
        if isinstance(spec, (int, np.integer)):
            if spectrum_list is None:
                raise ValueError("To use integer indices, 'project_data' must contain 'spectra'.")
            return spectrum_list[spec]
        return spec

    # 1. Resolve the reference spectrum
    ref_spectra = resolve_spec(ref_spectra)

    # 2. Prevent crashes if a single integer/string is passed instead of a list
    if isinstance(spectra_to_analyze, (int, np.integer, str)):
        spectra_to_analyze = [spectra_to_analyze]

    # 3. Resolve the analysis spectra
    spectra_to_analyze = [resolve_spec(x) for x in spectra_to_analyze]
    # -------------------------------------------------------------

    lab_1, lab_2 = 'position_1', 'position_2'
    print("--- Analysis started ---")

    # 1. Maintain Order & Subset DataFrame
    required_cols = [c for c in spectra_to_analyze if c in pivot_df.columns.levels[0]]
    if ref_spectra not in required_cols and ref_spectra in pivot_df.columns.levels[0]:
        required_cols.insert(0, ref_spectra)

    updated_df = pivot_df.sort_index(axis=1).loc[:, (required_cols, slice(None))].copy()

    # 2. Extract Reference coordinates
    try:
        ref_pos1 = updated_df[(ref_spectra, lab_1)]
        ref_pos2 = updated_df[(ref_spectra, lab_2)]
    except KeyError:
        raise KeyError(f"Reference spectrum '{ref_spectra}' missing required {lab_1}/{lab_2} data.")

    # 3. Symmetric Scaling Logic
    dims = pivot_df.attrs.get('dimensions', [])
    w1, w2 = 1.0, 0.142  # Safe default fallback (HN)

    if len(dims) >= 2:
        dim1_str, dim2_str = str(dims[0]).upper(), str(dims[1]).upper()

        if alpha is None:
            def get_weight(dim_name):
                if 'H' in dim_name:
                    return 1.0
                if 'N' in dim_name:
                    return 0.142
                if 'C' in dim_name:
                    return 0.33
                return 1.0

            w1, w2 = get_weight(dim1_str), get_weight(dim2_str)
            print(f"✅ Auto-scaling: {dims[0]}(w={w1}) and {dims[1]}(w={w2}) -> Proton Scale")
        else:
            w1 = 1.0 if 'H' in dim1_str else float(alpha)
            w2 = 1.0 if 'H' in dim2_str else float(alpha)
            print(f"ℹ️ Using manual alpha scaling based on {dims}: w1={w1}, w2={w2}")
    else:
        if alpha is not None:
            w2 = float(alpha)
        print(f"⚠️ Warning: No metadata. Defaulting to scaling w1={w1}, w2={w2}")

    # 4. Main Calculation Loop
    for i, spec_name in enumerate(spectra_to_analyze):

        # --- CSP Calculation ---
        if (spec_name, lab_1) in updated_df.columns and (spec_name, lab_2) in updated_df.columns:
            d1 = updated_df[(spec_name, lab_1)] - ref_pos1
            d2 = updated_df[(spec_name, lab_2)] - ref_pos2
            updated_df[('CSPs', spec_name)] = np.sqrt((w1 * d1)**2 + (w2 * d2)**2)

        # --- Smart Scaling Factor Logic ---
        spec_scale = 1.0
        if scaling_factors is not None:
            if isinstance(scaling_factors, dict):
                spec_scale = float(scaling_factors.get(spec_name, 1.0))
            elif isinstance(scaling_factors, (list, tuple)):
                if i < len(scaling_factors):
                    spec_scale = float(scaling_factors[i])
                else:
                    print(f"⚠️ Warning: Missing scaling factor in list for '{
                          spec_name}'. Defaulting to 1.0")
            elif isinstance(scaling_factors, (int, float)):
                spec_scale = float(scaling_factors)

        # --- Intensity Ratio & Normalization Pipeline ---
        for met in ['height', 'volume']:
            if (spec_name, met) in updated_df.columns and (ref_spectra, met) in updated_df.columns:

                # Base Ratio Calculation
                target_vals = updated_df[(spec_name, met)] * spec_scale
                safe_ref = updated_df[(ref_spectra, met)].replace(0, np.nan)
                # Handles negative peaks
                ratio_series = target_vals.abs() / safe_ref.abs()

                # Apply Normalization if a factor was provided
                if normalization_factor is not None:
                    processed_series = normalize_series(
                        ratio_series, normalization_factor=normalization_factor)
                    prefix = 'Norm_'
                else:
                    processed_series = ratio_series
                    prefix = 'Ratio_'

                # Remove zeros to prevent them from skewing the data
                clean_series = processed_series.replace(0, np.nan)

                # Optional Smoothing Logic
                if smoothing_window and smoothing_window > 1:
                    smoothed = clean_series.rolling(
                        window=smoothing_window, center=True, min_periods=1).mean()
                    final_series = smoothed.where(clean_series.notna())
                else:
                    final_series = clean_series

                # Save to DataFrame with the dynamic prefix
                updated_df[(f'{prefix}{met.capitalize()}', spec_name)] = final_series

    # 5. Handle Alignment
    if align:
        if not sequences_dict:
            print("⚠️ Warning: 'align' is True, but no sequence was found.")
        elif len(sequences_dict) > seq_index:
            target_key = list(sequences_dict.keys())[seq_index]
            updated_df = align_to_full_sequence(updated_df, sequences_dict[target_key])
            print(f"✅ Aligned to sequence: {target_key}")
        else:
            print(f"⚠️ Warning: seq_index {seq_index} out of range. Skipping alignment.")

    # 6. Add Metadata
    res_names = updated_df.index.get_level_values('residue_name')
    one_letter_codes = [get_one_letter(r) for r in res_names]
    updated_df.insert(0, ('Metadata', 'res_single'), one_letter_codes)

    # 7. Final Column Reordering
    int_prefix = 'Norm_' if normalization_factor is not None else 'Ratio_'
    categories = ['Metadata'] + required_cols + \
        ['CSPs', f'{int_prefix}Height', f'{int_prefix}Volume']
    existing_cats = [c for c in categories if c in updated_df.columns.levels[0]]
    final_df = updated_df.reindex(columns=existing_cats, level=0)

    # 8. Saving Logic
    base_name = filename.replace('.xlsx', '') if filename else "analysis"
    if filename:
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / (base_name + '.xlsx')
        final_df.to_excel(save_path)
        print(f"✅ Saved analysis to ({save_path})")

    final_df.attrs['analysis_name'] = base_name

    # 9. Print Summary
    print("\n--- Analysis Summary ---")
    print(f"Reference Spectrum: {ref_spectra}")
    if 'CSPs' in final_df.columns.levels[0]:
        plotted_specs = list(final_df['CSPs'].columns)
        print("Spectra Analyzed:\n  - " + "\n  - ".join(plotted_specs))
    else:
        print("Spectra Analyzed:   None (No CSPs calculated)")
    print("-------------------------------------------\n")

    return final_df
