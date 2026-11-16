import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import get_one_letter, index_peaklist, is_unassigned, natural_sort_order

# =============================================================================
# Master pivot + CSP / intensity analysis
# =============================================================================

INFO_SHEET = "Info"

# Default CSP weights per nucleus (scale every dimension to the 1H range):
# CSP = sqrt(sum_i (w_i * delta_i)^2). Change them per analysis with alpha={...}.
# Typical values: 15N 0.14 (0.1-0.2); 13C depends on the position:
# CO ~0.33 (default), Calpha ~0.25, methyl ~0.25 (0.2-0.3), aromatic ~0.3.
CSP_WEIGHTS = {"H": 1.0, "N": 0.142, "C": 0.33}


def _nucleus(name):
    """'1H' -> 'H', '15N' -> 'N', '13C' -> 'C' (first letter of the axis/isotope code)."""
    letters = re.sub(r"[^A-Za-z]", "", str(name)).upper()
    return letters[:1]


def _csp_weights(dims, alpha):
    """
    CSP weight of every dimension.

    alpha = None   -> CSP_WEIGHTS (1H 1, 15N 0.142, 13C 0.33)
    alpha = number -> this value for every non-1H dimension (1H stays 1)
    alpha = dict   -> per nucleus, e.g. {"15N": 0.14, "13C": 0.25}; missing nuclei use CSP_WEIGHTS
    """
    if isinstance(alpha, dict):
        custom = {_nucleus(k): float(v) for k, v in alpha.items()}
    else:
        custom = {}

    weights = []
    for d in dims:
        nuc = _nucleus(d)
        if nuc == "H" and nuc not in custom:
            w = 1.0
        elif nuc in custom:
            w = custom[nuc]
        elif alpha is not None and not isinstance(alpha, dict):
            w = float(alpha)
        elif nuc in CSP_WEIGHTS:
            w = CSP_WEIGHTS[nuc]
        else:
            print(f"⚠️ No CSP weight for dimension '{d}' – using 1.0 (pass alpha={{'{d}': ...}}).")
            w = 1.0
        weights.append(w)
    return weights


def create_master_pivot(peaks_dict, ref_spectrum=None, output_dir=None):
    """
    Combines all 2D spectra with the same dimensions into one master table (one row per
    residue) and saves it as spectra_analysis.xlsx (sheets: Master_Pivot, Heights_Only,
    Volumes_Only, Positions_Only and Info with the dimensions for the CSP weights).
    Sequence codes are cast to integers where possible.

    The input files for the NMR_plot scripts are written by ``export_spectra_plot``.

    Args:
        peaks_dict (dict):
            A dictionary containing the extracted peak lists to be combined.
        ref_spectrum (str, optional):
            The name of the reference spectrum that defines which dimensions are
            combined (e.g., the 15N or 13C HSQC). Defaults to the first 2D spectrum.
        output_dir (str or Path, optional):
            Directory for the exported Excel workbook. Defaults to 'results'.

    Returns:
        pd.DataFrame:
            The compiled master pivot table containing the aligned 2D spectra.
    """
    data_cols = ["peak_id", "volume", "height", "position_1", "position_2"]

    indexed_dfs = {}
    reference_dims = None
    reference_name = ""
    skipped_not_2d = []
    skipped_dims = []

    # --- Dimension Locking ---
    if ref_spectrum is not None:
        if ref_spectrum in peaks_dict:
            reference_name = ref_spectrum
            reference_dims = peaks_dict[ref_spectrum].attrs.get("dimensions", [])
            print(
                f"🎯 Locking dimensions to explicit reference: '{reference_name}' {reference_dims}"
            )
        else:
            print(f"⚠️ WARNING: '{ref_spectrum}' not found. Falling back to auto-detection.")

    for name, df in peaks_dict.items():
        dims = df.attrs.get("dimensions", [])
        if len(dims) != 2:
            skipped_not_2d.append(name)
            continue

        if reference_dims is None:
            reference_dims = dims
            reference_name = name
        elif dims != reference_dims:
            skipped_dims.append(f"{name} {dims}")
            continue

        indexed_dfs[name] = index_peaklist(df, data_cols=data_cols, name=name)

    if skipped_not_2d:
        print(f"ℹ️ Skipped {len(skipped_not_2d)} non-2D spectra: {', '.join(skipped_not_2d)}")
    if skipped_dims:
        print(
            f"ℹ️ Skipped {len(skipped_dims)} spectra with other dimensions than "
            f"{reference_dims}: {', '.join(skipped_dims)}"
        )

    if not indexed_dfs:
        print("❌ ERROR: No compatible 2D spectra were found.")
        return None

    # --- Combine ---
    table = pd.concat(indexed_dfs, axis=1)

    # --- Natural sorting (chain, assigned first, '@2' before '@10') ---
    table = table.iloc[natural_sort_order(table.index.to_frame(index=False))]

    # --- Reorder internal columns logically ---
    preferred_order = ["peak_id", "position_1", "position_2", "height", "volume"]
    existing_levels = table.columns.get_level_values(1).unique()
    new_order = [c for c in preferred_order if c in existing_levels]
    table = table.reindex(columns=new_order, level=1)

    table.attrs["dimensions"] = reference_dims

    # --- EXPORT SECTION: Single Workbook, Multiple Sheets ---
    excel_dir = Path(output_dir) if output_dir is not None else Path("results")
    excel_dir.mkdir(parents=True, exist_ok=True)
    file_path = excel_dir / "spectra_analysis.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Master_Pivot")

        if "height" in existing_levels:
            table.xs("height", axis=1, level=1).to_excel(writer, sheet_name="Heights_Only")

        if "volume" in existing_levels:
            table.xs("volume", axis=1, level=1).to_excel(writer, sheet_name="Volumes_Only")

        pos_cols = [c for c in ["position_1", "position_2"] if c in existing_levels]
        if pos_cols:
            table.reindex(columns=pos_cols, level=1).to_excel(writer, sheet_name="Positions_Only")

        # Metadata needed when the pivot is re-loaded from Excel later
        pd.DataFrame(
            {
                "key": ["dimensions", "reference_spectrum"],
                "value": [", ".join(map(str, reference_dims)), reference_name],
            }
        ).to_excel(writer, sheet_name=INFO_SHEET, index=False)

    print(f"✅ Exported all tables as sheets to {file_path}\n")

    return table


def _load_pivot_from_excel(pivot_path):
    """Loads a (curated) master pivot from Excel, including its dimension metadata."""
    if not pivot_path.exists():
        raise FileNotFoundError(
            f"Pivot file not found: {pivot_path}\n"
            "-> Copy spectra_analysis.xlsx to this name and correct it by hand first."
        )
    print(f"📁 Loading curated pivot from Excel: {pivot_path.name}")
    pivot_df = pd.read_excel(pivot_path, sheet_name=0, header=[0, 1], index_col=[0, 1, 2])

    with pd.ExcelFile(pivot_path) as xls:
        sheet_names = xls.sheet_names
    if INFO_SHEET in sheet_names:
        info = pd.read_excel(pivot_path, sheet_name=INFO_SHEET).set_index("key")["value"]
        dims = info.get("dimensions")
        if isinstance(dims, str) and dims:
            pivot_df.attrs["dimensions"] = [d.strip() for d in dims.split(",")]
    return pivot_df


def smooth_by_residue(series, window, stat="mean"):
    """
    Rolling mean over neighboring residue *numbers* (not table rows), per chain.

    A window of 3 averages residues i-1, i and i+1. Residues without data
    (e.g. prolines or missing peaks) are gaps and do not pull in residues further
    away. Unassigned peaks keep their original value. Only positions that have a
    value themselves get a smoothed value. ``stat`` can also be "sum" or "count"
    (used for the error propagation of the smoothed values).
    """
    result = series.copy()
    idx = series.index.to_frame(index=False)
    seq_num = pd.to_numeric(idx["sequence_code"], errors="coerce")
    assigned = seq_num.notna() & ~idx["residue_name"].apply(is_unassigned)

    for _, rows in idx[assigned].groupby("chain_code", sort=False).groups.items():
        rows = list(rows)
        nums = seq_num.iloc[rows].astype(int).to_numpy()
        values = pd.Series(series.iloc[rows].to_numpy(dtype=float), index=nums)
        values = values[~values.index.duplicated(keep="first")]

        full = values.reindex(range(nums.min(), nums.max() + 1))
        smoothed = getattr(full.rolling(window=window, center=True, min_periods=1), stat)()

        new_vals = smoothed.reindex(nums).to_numpy()
        has_value = series.iloc[rows].notna().to_numpy()
        result.iloc[rows] = np.where(has_value, new_vals, np.nan)

    return result


def _smooth_sigma(sigma, window):
    """Error of a smoothed (averaged) value: sqrt(sum of sigma^2) / n over the same window."""
    present = pd.Series(np.where(sigma.notna(), 1.0, np.nan), index=sigma.index)
    var_sum = smooth_by_residue(sigma**2, window, stat="sum")
    count = smooth_by_residue(present, window, stat="count")  # unassigned peaks keep 1
    return np.sqrt(var_sum) / count


def _resolve_noise(noise, spec_name, ref_name, resolve_spec):
    """
    Noise (standard deviation, same units as the peak heights) for one analysed spectrum
    and the reference. ``noise`` can be a number (all spectra), a tuple
    (analysed spectra, reference) or a dict {spectrum name or menu number: noise}.
    Returns (noise_target, noise_ref) or (None, None).
    """
    if noise is None:
        return None, None
    if isinstance(noise, dict):
        resolved = {resolve_spec(k): v for k, v in noise.items()}
        return resolved.get(spec_name), resolved.get(ref_name)
    if isinstance(noise, (tuple, list)) and len(noise) == 2:
        return noise[0], noise[1]
    return noise, noise


def normalize_series(series, normalization_factor=1, clip_lower=True):
    """
    Normalizes series from 0 to the p-th percentile.
    Values above the p-th percentile are capped at 1.0. Negative values (noise of fully
    broadened peaks) are set to 0 unless ``clip_lower=False`` (used before smoothing, so
    that positive and negative noise average out instead of biasing the mean upwards).
    """
    threshold = series.quantile(normalization_factor)

    # If the threshold is 0 (or all data is 0), return 0s to avoid division by zero
    if pd.isna(threshold) or threshold <= 0:
        return series * 0.0

    normalized = (series / threshold).clip(upper=1)
    return normalized.clip(lower=0) if clip_lower else normalized


def align_to_full_sequence(pivot_df, sequence_df, match_residue_name=True):
    """
    Aligns NMR data to a full sequence skeleton.

    Final output index remains:
        chain_code, sequence_code, residue_name

    Matching ignores chain_code and uses either:
        sequence_code + residue_name
    or:
        sequence_code only
    """

    def norm_seq(x):
        if pd.isna(x):
            return pd.NA
        try:
            return str(int(float(x)))
        except Exception:
            return str(x).strip()

    # 1. Build full sequence skeleton
    skeleton = (
        sequence_df[["chain_code", "sequence_code", "residue_name"]]
        .copy()
        .set_index(["chain_code", "sequence_code", "residue_name"])
        .sort_index()
    )

    skeleton.columns = pd.MultiIndex.from_product(
        [skeleton.columns, [""]], names=pivot_df.columns.names
    )

    # 2. Build matching keys
    key_cols = ["_seq_key", "_res_key"] if match_residue_name else ["_seq_key"]

    skeleton_index_df = skeleton.index.to_frame(index=False)
    skeleton_index_df["_seq_key"] = skeleton_index_df["sequence_code"].apply(norm_seq)
    skeleton_index_df["_res_key"] = skeleton_index_df["residue_name"].apply(get_one_letter)

    pivot_index_df = pivot_df.index.to_frame(index=False)
    pivot_index_df["_seq_key"] = pivot_index_df["sequence_code"].apply(norm_seq)
    pivot_index_df["_res_key"] = pivot_index_df["residue_name"].apply(get_one_letter)

    if len(key_cols) == 1:
        skeleton_key = pd.Index(skeleton_index_df[key_cols[0]], name=key_cols[0])
        pivot_key = pd.Index(pivot_index_df[key_cols[0]], name=key_cols[0])
    else:
        skeleton_key = pd.MultiIndex.from_frame(skeleton_index_df[key_cols])
        skeleton_key.names = key_cols

        pivot_key = pd.MultiIndex.from_frame(pivot_index_df[key_cols])
        pivot_key.names = key_cols

    # 3. Reindex NMR data onto sequence skeleton
    pivot_tmp = pivot_df.copy()
    pivot_tmp.index = pivot_key

    if pivot_tmp.index.duplicated().any():
        pivot_tmp = pivot_tmp.groupby(
            level=list(range(pivot_tmp.index.nlevels)),
            sort=False,
        ).first()

    aligned_data = pivot_tmp.reindex(skeleton_key)

    # Restore original biological index
    aligned_data.index = skeleton.index

    matched = aligned_data.notna().any(axis=1).sum()
    print(f"✅ Alignment complete: matched {matched} / {len(aligned_data)} residues")

    return pd.concat([skeleton, aligned_data], axis=1)


def add_analysis_to_master(
    project_data,
    ref_spectra,
    spectra_to_analyze,
    align=False,
    seq_index=0,
    alpha=None,
    normalization_factor=None,
    scaling_factors=None,
    smoothing_window=None,
    noise=None,
    lod_sigma=3,
    filename=None,
):
    """
    Calculates Chemical Shift Perturbations (CSPs) and intensity/volume ratios.

    The CSP weights are set automatically from the NEF dimensions (1H, 15N, 13C)
    unless ``alpha`` is given (one number or a weight per nucleus).

    Args:
        project_data (dict, str, or Path):
            A dictionary containing 'pivot', 'sequences', 'spectra', 'out_dir'
            and optionally 'input_dir' and 'dimensions'. 'pivot' can also be a path
            to a (modified) spectra_analysis.xlsx. Alternatively, project_data can
            directly be such a path.
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
        alpha (float or dict, optional):
            Weights of the CSP calculation, CSP = sqrt((w1·Δδ1)² + (w2·Δδ2)²).
            None (default): automatic from the NEF dimensions, 1H = 1, 15N = 0.142,
            13C = 0.33 (see ``CSP_WEIGHTS``). A number is used for every non-1H
            dimension (1H stays 1). A dict sets the weight per nucleus, e.g.
            ``{"15N": 0.14, "13C": 0.25}`` (needed for 13C-15N spectra like CON, where
            both dimensions need their own weight); missing nuclei keep their default.
            Typical weights: 15N 0.14 (0.1-0.2). 13C depends on the carbon:
            CO ~0.33 (default), Cα ~0.25, methyl ~0.25 (0.2-0.3), aromatic ~0.3.
            The formula has no factor 1/2 (some papers use sqrt(½·(...)), √2 smaller).
        normalization_factor (float, optional):
            Quantile (e.g. 0.95) used to normalize the intensity ratios: they are divided
            by this quantile, values above are capped at 1 and negative values (noise of
            fully broadened peaks) set to 0. A quantile instead of the maximum is robust
            against single outliers (e.g. overlapping peaks). Gives 'Norm_' columns;
            without it the raw ratios are stored as 'Ratio_' columns.
        scaling_factors (float, list, or dict, optional):
            Multipliers for the intensities of the analysed spectra (not the reference),
            e.g. NS_ref / NS for different numbers of scans. Can be a single value, a list
            matching `spectra_to_analyze`, or a spectrum-to-value dictionary.
        smoothing_window (int, optional):
            Window size (in residues) for a rolling average of the intensities across
            neighboring residue numbers, separately for each chain. Missing residues
            count as gaps, unassigned peaks are not smoothed. Good for PRE analyses.
            The signed ratios are averaged first and only then set to >= 0, so the noise
            of broadened peaks averages out. Errors: sqrt(sum σ_i²) / n.
        noise (float, tuple or dict, optional):
            Spectral noise as standard deviation (SD), same units as the peak heights, for
            errors and detection limits of the height ratios. CCPN: right-click in the
            spectrum -> 'Estimate Noise...' (shortcut EN) and take the 'SD' value. Not the
            'Noise Level' (contour base, = mean + 3.5 SD, about 3.5x too large).
            A number is used for all spectra, a tuple means (analysed spectra, reference),
            a dict maps spectrum names or menu numbers to their noise. Adds the column
            groups '<Norm_/Ratio_>Height_Err' (error, sqrt(s_p^2 + r^2 s_ref^2) / |I_ref|)
            and '<Norm_/Ratio_>Height_LOD' (detection limit, lod_sigma * s_p / |I_ref|).
            Ratios below the LOD cannot be distinguished from 0 (fully broadened peak).
        lod_sigma (float, optional):
            Multiple of the noise used for the detection limit. Defaults to 3.
        filename (str, optional):
            Filename for the exported Excel table (e.g., "analysis_1"). The function
            automatically routes it to the project's specific output directory.

    Returns:
        pd.DataFrame:
            The final compiled DataFrame containing metadata, coordinates,
            CSPs, and calculated ratios, optionally aligned to the full sequence.
    """

    # --- 1. Normalize Input ---
    if isinstance(project_data, (str, Path)):
        project_data = {"pivot": project_data, "sequences": {}, "spectra": None}

    pivot_data = project_data.get("pivot")
    sequences_dict = project_data.get("sequences") or {}
    spectrum_list = project_data.get("spectra")

    # --- 2. Resolve the Pivot Data ---
    default_out_dir = Path("results")
    if isinstance(pivot_data, (str, Path)):
        pivot_path = Path(pivot_data)
        pivot_df = _load_pivot_from_excel(pivot_path)
        default_out_dir = pivot_path.parent
    else:
        pivot_df = pivot_data

    out_dir = Path(project_data.get("out_dir") or default_out_dir)

    if pivot_df is None or not isinstance(pivot_df, pd.DataFrame):
        raise ValueError(
            "❌ ERROR: 'project_data' must contain a valid 'pivot' dataframe or Excel path."
        )

    # --- SMART INDEX LOGIC: Convert integers to spectrum names ---
    def resolve_spec(spec):
        if isinstance(spec, (int, np.integer)):
            if spectrum_list is None:
                raise ValueError("To use integer indices, 'project_data' must contain 'spectra'.")
            return spectrum_list[spec]
        return spec

    ref_spectra = resolve_spec(ref_spectra)

    if isinstance(spectra_to_analyze, (int, np.integer, str)):
        spectra_to_analyze = [spectra_to_analyze]

    spectra_to_analyze = [resolve_spec(x) for x in spectra_to_analyze]

    lab_1, lab_2 = "position_1", "position_2"
    print("--- Analysis started ---")

    # 1. Maintain Order & Subset DataFrame
    available = set(pivot_df.columns.get_level_values(0))
    required_cols = [c for c in spectra_to_analyze if c in available]
    if ref_spectra not in required_cols and ref_spectra in available:
        required_cols.insert(0, ref_spectra)

    updated_df = pivot_df.sort_index(axis=1).loc[:, (required_cols, slice(None))].copy()

    # 2. Extract Reference coordinates
    try:
        ref_pos1 = updated_df[(ref_spectra, lab_1)]
        ref_pos2 = updated_df[(ref_spectra, lab_2)]
    except KeyError:
        raise KeyError(f"Reference spectrum '{ref_spectra}' missing required {lab_1}/{lab_2} data.")

    # 3. CSP weights per dimension (scaled to the 1H range)
    dims = list(project_data.get("dimensions") or pivot_df.attrs.get("dimensions", []))[:2]
    if len(dims) < 2:
        # no metadata: 1H + the nucleus named in alpha (e.g. {"13C": 0.33}), else 1H/15N
        named = [k for k in alpha if _nucleus(k) != "H"] if isinstance(alpha, dict) else []
        dims = ["1H", named[0] if len(named) == 1 else "15N"]
        print(
            f"⚠️ Warning: No dimension metadata – assuming a {dims[0]}/{dims[1]} spectrum. "
            "For other spectra pass project_data['dimensions'] (e.g. ['1H', '13C'])."
        )
    w1, w2 = _csp_weights(dims, alpha)
    mode = "automatic" if alpha is None else "manual"
    print(f"✅ CSP weights ({mode}): {dims[0]} w={w1}, {dims[1]} w={w2}")
    if isinstance(alpha, (int, float)) and "H" not in (_nucleus(dims[0]), _nucleus(dims[1])):
        print(f"ℹ️ alpha={alpha} is used for both dimensions. For separate weights pass "
              f"a dict, e.g. alpha={{'{dims[0]}': ..., '{dims[1]}': ...}}.")

    # 4. Main Calculation Loop
    for i, spec_name in enumerate(spectra_to_analyze):
        # --- CSP Calculation ---
        if (spec_name, lab_1) in updated_df.columns and (spec_name, lab_2) in updated_df.columns:
            d1 = updated_df[(spec_name, lab_1)] - ref_pos1
            d2 = updated_df[(spec_name, lab_2)] - ref_pos2
            updated_df[("CSPs", spec_name)] = np.sqrt((w1 * d1) ** 2 + (w2 * d2) ** 2)

        # --- Smart Scaling Factor Logic ---
        spec_scale = 1.0
        if scaling_factors is not None:
            if isinstance(scaling_factors, dict):
                spec_scale = float(scaling_factors.get(spec_name, 1.0))
            elif isinstance(scaling_factors, (list, tuple)):
                if i < len(scaling_factors):
                    spec_scale = float(scaling_factors[i])
                else:
                    print(
                        f"⚠️ Warning: Missing scaling factor in list for '{spec_name}'. "
                        "Defaulting to 1.0"
                    )
            elif isinstance(scaling_factors, (int, float)):
                spec_scale = float(scaling_factors)

        # --- Intensity Ratio & Normalization Pipeline ---
        for met in ["height", "volume"]:
            if (spec_name, met) in updated_df.columns and (ref_spectra, met) in updated_df.columns:
                target_vals = updated_df[(spec_name, met)] * spec_scale
                safe_ref = updated_df[(ref_spectra, met)].replace(0, np.nan)
                # Signed ratio: folded (negative) peaks are negative in both spectra and give
                # a positive ratio. A fully broadened peak only leaves noise, which scatters
                # around 0 - abs() would turn negative noise into a fake residual intensity.
                ratio_series = target_vals / safe_ref
                smooth = bool(smoothing_window and smoothing_window > 1)
                prefix = "Norm_" if normalization_factor is not None else "Ratio_"
                cat = f"{prefix}{met.capitalize()}"

                # Errors and detection limit from the spectral noise (heights only)
                n_t, n_r = _resolve_noise(noise, spec_name, ref_spectra, resolve_spec)
                err_series = lod_series = None
                if met == "height" and n_t is not None and n_r is not None:
                    s_t = float(n_t) * spec_scale
                    err_series = np.sqrt(s_t**2 + (ratio_series * float(n_r)) ** 2) / safe_ref.abs()
                    lod_series = (lod_sigma * s_t / safe_ref.abs()).where(ratio_series.notna())
                elif met == "height" and noise is not None:
                    print(f"⚠️ No noise SD for '{spec_name}' or '{ref_spectra}' – no errors.")

                if normalization_factor is not None:
                    threshold = ratio_series.quantile(normalization_factor)
                    processed_series = normalize_series(
                        ratio_series,
                        normalization_factor=normalization_factor,
                        clip_lower=not smooth,
                    )
                    if err_series is not None and pd.notna(threshold) and threshold > 0:
                        err_series = err_series / threshold
                        lod_series = lod_series / threshold
                else:
                    processed_series = ratio_series

                # A real intensity of 0 (fully broadened peak) is kept: it means I/I0 = 0.
                # Division by a zero reference is already prevented above (safe_ref).
                if smooth:
                    final_series = smooth_by_residue(processed_series, smoothing_window)
                    if normalization_factor is not None:
                        final_series = final_series.clip(lower=0)  # clip only after averaging
                    if err_series is not None:
                        err_series = _smooth_sigma(err_series, smoothing_window)
                        lod_series = _smooth_sigma(lod_series, smoothing_window)
                else:
                    final_series = processed_series

                updated_df[(cat, spec_name)] = final_series
                if err_series is not None:
                    updated_df[(f"{cat}_Err", spec_name)] = err_series
                    updated_df[(f"{cat}_LOD", spec_name)] = lod_series
                    n_below = int((final_series < lod_series).sum())
                    print(
                        f"📉 {spec_name}: {n_below} residues below the detection limit "
                        f"({lod_sigma}σ) – not distinguishable from 0."
                    )

    # 5. Handle Alignment
    if align:
        if not sequences_dict:
            print("⚠️ Warning: 'align' is True, but no sequence was found.")
        elif len(sequences_dict) > seq_index:
            target_key = list(sequences_dict.keys())[seq_index]
            print(f"🧬 Using sequence for alignment: {target_key}")
            updated_df = align_to_full_sequence(
                updated_df,
                sequences_dict[target_key],
                match_residue_name=True,
            )
        else:
            print(f"⚠️ Warning: seq_index {seq_index} out of range. Skipping alignment.")

    # 6. Add Metadata
    res_names = updated_df.index.get_level_values("residue_name")
    one_letter_codes = [get_one_letter(r) for r in res_names]
    updated_df.insert(0, ("Metadata", "res_single"), one_letter_codes)

    # 7. Final Column Reordering
    int_prefix = "Norm_" if normalization_factor is not None else "Ratio_"
    categories = (
        ["Metadata"]
        + required_cols
        + ["CSPs", f"{int_prefix}Height", f"{int_prefix}Height_Err", f"{int_prefix}Height_LOD"]
        + [f"{int_prefix}Volume"]
    )
    present = set(updated_df.columns.get_level_values(0))
    existing_cats = [c for c in categories if c in present]
    final_df = updated_df.reindex(columns=existing_cats, level=0)

    # 8. Saving Logic
    if filename:
        safe_name = str(filename).replace("/", "_").replace("\\", "_")
        if not safe_name.lower().endswith(".xlsx"):
            safe_name += ".xlsx"

        save_path = out_dir / safe_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_excel(save_path)
        print(f"✅ Saved analysis to ({save_path})")

        final_df.attrs["analysis_name"] = save_path.stem
        final_df.attrs["output_dir"] = save_path.parent
    else:
        final_df.attrs["analysis_name"] = "analysis"
        final_df.attrs["output_dir"] = out_dir

    if project_data.get("input_dir"):
        final_df.attrs["input_dir"] = Path(project_data["input_dir"])

    # 9. Print Summary
    print("\n--- Analysis Summary ---")
    print(f"Reference Spectrum: {ref_spectra}")
    if "CSPs" in present:
        plotted_specs = list(final_df["CSPs"].columns)
        print("Spectra Analyzed:\n  - " + "\n  - ".join(plotted_specs))
    else:
        print("Spectra Analyzed:   None (No CSPs calculated)")
    print("-------------------------------------------\n")

    return final_df
