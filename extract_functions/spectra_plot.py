import contextlib
import os
from pathlib import Path

import pandas as pd

from .utils import index_peaklist, natural_sort_order, residue_label

DEFAULT_COLOUR = "black"  # used if the NEF file has no contour colour (non-CCPN NEFs)

# =============================================================================
# Export for the external NMR_plot scripts (NMR_2D_plot repository)
# =============================================================================


def _safe_name(name):
    """Removes characters that would create sub-folders in a filename."""
    return str(name).replace("/", "_").replace("\\", "_")


def _format_contour(contour):
    """Formats a contour base like 510884532.6 -> '5.11e8'."""
    try:
        return f"{float(contour):.2e}".replace("+0", "").replace("+", "")
    except (ValueError, TypeError):
        return str(contour) if contour is not None else "Unknown"


def _positions_table(df, name):
    """2D: peak positions with a 'Residue' label column (e.g. 'A24' or '@12' if unassigned).
    One peak per residue (labels for the 2D plots of NMR_plot)."""
    indexed = index_peaklist(
        df, data_cols=["position_1", "position_2"], name=name, warn_duplicates=False
    )
    if indexed.empty:
        return None

    idx = indexed.index.to_frame(index=False)
    indexed = indexed.iloc[natural_sort_order(idx)]
    idx = indexed.index.to_frame(index=False)

    table = indexed.reset_index(drop=True)
    table.insert(
        0,
        "Residue",
        [residue_label(r, s) for r, s in zip(idx["residue_name"], idx["sequence_code"])],
    )
    return table.dropna(subset=["position_1", "position_2"], how="all")


def _peaks_table(df, ndim):
    """
    1D / 3D: every peak (several peaks per residue are normal here, e.g. 1H 1D, HNCA i/i-1,
    NOESY), in peak order. Columns: Residue (assignment of dimension 1, or the peak number),
    position_1..n and, if assigned, the atom of every dimension (e.g. 'A24 HA').
    """
    pos_cols = [f"position_{i}" for i in range(1, ndim + 1)]
    if not set(pos_cols).issubset(df.columns):
        return None
    table = df.dropna(subset=pos_cols, how="all")
    if table.empty:
        return None

    def col(name):
        return table[name] if name in table.columns else pd.Series(pd.NA, index=table.index)

    peak_id = col("peak_id")
    seq1 = col("sequence_code_1").where(col("sequence_code_1").notna(), peak_id)
    out = pd.DataFrame(
        {"Residue": [residue_label(r, s) for r, s in zip(col("residue_name_1"), seq1)]}
    )
    for c in pos_cols:
        out[c] = table[c].to_numpy()

    for i in range(1, ndim + 1):
        seq, res, atom = col(f"sequence_code_{i}"), col(f"residue_name_{i}"), col(f"atom_name_{i}")
        if seq.notna().any() or atom.notna().any():
            out[f"atom_{i}"] = [
                "" if pd.isna(s) and pd.isna(a)
                else f"{residue_label(r, s)} {'' if pd.isna(a) else a}".strip()
                for s, r, a in zip(seq, res, atom)
            ]
    return out


def _nef_spectra(nef_file):
    """Metadata of every spectrum in the NEF file, including spectra without peaks."""
    import pynmrstar

    from .parsing import _spectrum_metadata

    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stderr(fnull), contextlib.redirect_stdout(fnull):
            entry = pynmrstar.Entry.from_file(str(nef_file))

    spectra = {}
    for saveframe in entry:
        if saveframe.category != "nef_nmr_spectrum":
            continue
        meta = _spectrum_metadata(saveframe)
        spectra.setdefault(meta["spectrum_name"], meta)
    return spectra


def _metadata_row(name, attrs, csv_name):
    """One line of the metadata file, or None if the spectrum has no file path."""
    path = attrs.get("file_path")
    if not path or path == "_Undefined_":
        return None
    ndim = len(attrs.get("dimensions") or [])
    colour = attrs.get("slice_colour") if ndim == 1 else None
    colour = colour or attrs.get("contour_colour")
    no_colour = not colour or colour == "None"
    return {
        "path": path,
        "name": name,
        # NMR_plot: for 1D spectra the 'contour' column is a scaling factor (1 = unchanged)
        "contour": 1 if ndim == 1 else attrs.get("contour_base"),
        "colour": DEFAULT_COLOUR if no_colour else colour,
        # NMR_plot uses label CSVs only for 1D and 2D (not for 3D strips)
        "csv": "" if ndim == 3 else csv_name,
        "ndim": ndim,
        "no_colour": no_colour,
    }


def _write_metadata_file(rows, output_file, grouped=False):
    """Writes spectra_metadata(_all).txt in the data-list format of the NMR_plot templates.
    With ``grouped=True`` the spectra are sorted into 1D / 2D / 3D blocks (comment lines)."""
    # 1. Find the common base directory
    valid_paths = [str(Path(r["path"]).resolve()) for r in rows]
    if len(valid_paths) > 1:
        common_base = Path(os.path.commonpath(valid_paths))
    elif len(valid_paths) == 1:
        common_base = Path(valid_paths[0]).parent
    else:
        common_base = Path("")

    # 2. Pre-format the elements (relative paths, quotes)
    formatted_rows = []
    for r in rows:
        rel_path_str = r["path"]
        if str(common_base) != ".":
            try:
                rel_path_str = Path(r["path"]).resolve().relative_to(common_base).as_posix()
            except ValueError:
                pass

        formatted_rows.append(
            [
                f'"{rel_path_str}"',
                f'"{r["name"]}"',
                "1" if r.get("ndim") == 1 else _format_contour(r["contour"]),
                f'"{r["colour"]}"',
                f'"{r["csv"]}"' if r["csv"] else "",  # the label CSV is optional
            ]
        )

    # 3. Column widths for neat alignment
    widths = [max(len(fr[i]) for fr in formatted_rows) for i in range(4)]

    def line(fr):
        cols = [(fr[i] + ",").ljust(widths[i] + 1) for i in range(4)]
        if not fr[4]:  # no label CSV: (folder, name, contour, colour)
            return f"    ({cols[0]}    {cols[1]}    {cols[2]}    {fr[3]}),\n"
        return f"    ({cols[0]}    {cols[1]}    {cols[2]}    {cols[3]}    {fr[4]}),\n"

    # 4. Write
    with open(output_file, mode="w", encoding="utf-8") as file:
        file.write("# directory of your NMR files\n")
        if str(common_base) and str(common_base) != ".":
            file.write(f'base_dir = Path("{common_base.as_posix()}")\n\n')
        else:
            file.write('base_dir = Path("")\n\n')

        file.write("data = [\n")
        if grouped:
            for ndim in sorted({r["ndim"] for r in rows}):
                label = f"{ndim}D" if ndim else "unknown dimensions"
                file.write(f"    # --- {label} ---\n")
                for r, fr in zip(rows, formatted_rows):
                    if r["ndim"] == ndim:
                        file.write(line(fr))
        else:
            for fr in formatted_rows:
                file.write(line(fr))
        file.write("]\n")


def export_spectra_plot(peak_dict, output_dir=None, nef_file=None):
    """
    Creates the input files for the external NMR_plot scripts in one go:

    - ``spectra_metadata.txt``: 2D spectra only (paths, names, contour levels, colours).
    - ``spectra_metadata_all.txt``: the same for all 1D, 2D and 3D spectra, in 1D / 2D / 3D
      blocks, in the input format of NMR_plot (Template_all): 1D spectra get the scaling
      factor 1 as 'contour', 3D spectra and spectra without peaks have no label CSV.
      Spectra without peaks are included too (the NEF file is read again for them).
    - ``csv/<name>_pos.csv``: peak positions with residue labels for every spectrum.
      2D: one peak per residue. 1D/3D: every peak, with the assigned atom per dimension.
    - ``csv/all_peaklists/<peaklist>_pos.csv``: the same for every individual peak list.

    The names in the metadata files and the CSV filenames always match.
    Call it **after** applying your ``spec_rename_map`` so the new names are used.

    If a spectrum has several peak lists, the renamed one (or else the one with the
    most peaks) is used for the main CSV.

    Args:
        peak_dict (dict):
            Peak lists from ``extract_all_nef_data`` (optionally renamed).
        output_dir (str or Path, optional):
            Project output directory. Defaults to 'results'.
        nef_file (str or Path, optional):
            NEF file, used to add spectra without peaks to spectra_metadata_all.txt.
            Defaults to the file the peak lists were read from.
    """
    out_dir = Path(output_dir) if output_dir is not None else Path("results")
    csv_dir = out_dir / "csv"
    all_dir = csv_dir / "all_peaklists"
    all_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Group peak lists by spectrum (one spectrum can hold several peak lists) ---
    spectra = {}
    for key, df in peak_dict.items():
        ndim = len(df.attrs.get("dimensions") or [])
        spectrum = df.attrs.get("spectrum_name", key)
        renamed = key != df.attrs.get("peaklist_key", key)
        table = _positions_table(df, key) if ndim == 2 else _peaks_table(df, ndim)
        if table is None:
            continue

        table.to_csv(all_dir / f"{_safe_name(key)}_pos.csv", index=False)
        spectra.setdefault(spectrum, []).append(
            {"key": key, "renamed": renamed, "table": table, "attrs": df.attrs, "ndim": ndim}
        )

    # --- 2. One main CSV + one metadata row per spectrum ---
    rows = []
    skipped = []
    for spectrum, lists in spectra.items():
        main = max(lists, key=lambda pl: (pl["renamed"], len(pl["table"])))
        name = main["key"] if main["renamed"] else spectrum
        csv_name = f"{_safe_name(name)}_pos.csv"
        main["table"].to_csv(csv_dir / csv_name, index=False)

        row = _metadata_row(name, main["attrs"], csv_name)
        if row is None:
            skipped.append(name)
        else:
            rows.append(row)

    # --- 3. Spectra without peaks (only in the NEF file) ---
    if nef_file is None:
        nef_file = next(
            (df.attrs["nef_file"] for df in peak_dict.values() if df.attrs.get("nef_file")), None
        )
    no_peaks = []
    if nef_file is not None and Path(nef_file).exists():
        for spectrum, meta in _nef_spectra(nef_file).items():
            if spectrum in spectra:
                continue
            row = _metadata_row(spectrum, meta, "")
            if row is None:
                skipped.append(spectrum)
            else:
                rows.append(row)
                no_peaks.append(spectrum)

    if not rows:
        if spectra:
            print(f"✅ spectra_plot: position CSVs written to {csv_dir}")
        if nef_file is None and not peak_dict:
            print("⚠️ spectra_plot: no peak lists found. To export spectra without peaks "
                  "(e.g. 1D), pass the NEF file: export_spectra_plot(all_peaks, "
                  "output_dir=out_dir, nef_file=input_nef)\n")
        else:
            print("⚠️ spectra_plot: no spectrum has a file path in the NEF – no metadata file "
                  f"written. Spectra: {', '.join(skipped) or 'none'}\n")
        return

    # --- 4. Write the metadata files ---
    rows_2d = [r for r in rows if r["ndim"] == 2 and r["csv"]]
    meta_file = out_dir / "spectra_metadata.txt"
    if rows_2d:
        _write_metadata_file(rows_2d, meta_file)
        print(f"✅ spectra_plot: {len(rows_2d)} 2D spectra written to {meta_file}")

    meta_all = out_dir / "spectra_metadata_all.txt"
    _write_metadata_file(rows, meta_all, grouped=True)
    counts = ", ".join(
        f"{sum(r['ndim'] == n for r in rows)}× {n}D" for n in sorted({r["ndim"] for r in rows})
    )
    print(f"✅ spectra_plot: {len(rows)} spectra ({counts}) written to {meta_all}")
    print(f"✅ spectra_plot: position CSVs written to {csv_dir}")

    if no_peaks:
        print(f"ℹ️ spectra_plot: no peaks (no label CSV) for: {', '.join(no_peaks)}")
    if skipped:
        print(f"⚠️ spectra_plot: no spectrum file path for: {', '.join(skipped)}")
    no_colour = [r["name"] for r in rows if r["no_colour"]]
    if no_colour:
        print(f"ℹ️ spectra_plot: no contour colour in the NEF, using '{DEFAULT_COLOUR}' for: "
              f"{', '.join(no_colour)}")
    print()
