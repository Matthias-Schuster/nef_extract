import contextlib
import os

import numpy as np
import pandas as pd
import pynmrstar

from .utils import force_numeric

# =============================================================================
# READ .nef Functions
# =============================================================================


def _first_tag(saveframe, tag, default=None):
    """Returns the first value of a saveframe tag without surrounding quotes."""
    values = saveframe.get_tag(tag)
    if not values:
        return default
    return str(values[0]).strip("'\"")


def _spectrum_metadata(saveframe):
    """Collects the saveframe-level information of a nef_nmr_spectrum."""
    meta = {
        "experiment_type": _first_tag(
            saveframe, "_nef_nmr_spectrum.experiment_type", default="Unknown"
        ),
        # Spectrum name without the CCPN peak list serial (e.g. 'HSQC`2`' -> 'HSQC')
        "spectrum_name": saveframe.name.replace("nef_nmr_spectrum_", "").split("`")[0],
        "file_path": _first_tag(saveframe, "_nef_nmr_spectrum.ccpn_spectrum_file_path"),
        "contour_base": _first_tag(saveframe, "_nef_nmr_spectrum.ccpn_positive_contour_base"),
        "contour_colour": _first_tag(saveframe, "_nef_nmr_spectrum.ccpn_positive_contour_colour"),
        "slice_colour": _first_tag(saveframe, "_nef_nmr_spectrum.ccpn_slice_colour"),
    }

    try:
        dim_loop = saveframe.get_loop("_nef_spectrum_dimension")
    except KeyError:
        dim_loop = None
    meta["dimensions"] = dim_loop.get_tag("_nef_spectrum_dimension.axis_code") if dim_loop else []
    return meta


def extract_all_nef_data(filepath, report=False, spectra_plot=False, output_dir=None):
    """
    Extracts and compiles sequences, chemical shift lists, and peak lists from an NEF file.

    Every peak list carries its spectrum metadata in ``df.attrs`` (experiment type,
    dimensions, spectrum file path, contour level and colour). This metadata survives
    renaming via ``spec_rename_map`` and is used by ``export_spectra_plot``.

    Args:
        filepath (str or Path):
            The path to the input NMR Exchange Format (NEF) file.
        report (bool, optional):
            If True, prints a detailed summary of the NEF file's
            spectral architecture and contents. Defaults to False.
        spectra_plot (bool, optional):
            If True, immediately runs ``export_spectra_plot`` with the original NEF
            spectrum names (spectra_metadata.txt for 2D, spectra_metadata_all.txt for
            1D/2D/3D and csv/*_pos.csv, the input of the NMR_plot scripts).
            If you rename your spectra, call ``ef.export_spectra_plot(all_peaks, out_dir)``
            after renaming instead. Defaults to False.
        output_dir (str or Path, optional):
            Base directory for the spectra_plot export.

    Returns:
        tuple:
            A tuple containing three elements:
                - sequences (dict): Extracted sequence data.
                - shifts (dict): Extracted chemical shift lists.
                - peaks (dict): Extracted peak lists.
    """
    # Load the file
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stderr(fnull), contextlib.redirect_stdout(fnull):
            entry = pynmrstar.Entry.from_file(str(filepath))

    sequences = {}
    shifts = {}
    peaks = {}

    if report:
        print("\n--Print all loops in the nef file:")
    for saveframe in entry:
        # 1. Saveframe-level metadata (only for spectra)
        spec_metadata = {}
        if saveframe.category == "nef_nmr_spectrum":
            spec_metadata = _spectrum_metadata(saveframe)

        # 2. Iterate through the loops inside the saveframe
        for loop in saveframe.loops:
            category = loop.category
            if report:
                print(f"{saveframe.name}{category}")

            if not loop.data:
                continue

            columns = [tag.split(".")[-1] for tag in loop.get_tag_names()]
            df = pd.DataFrame(loop.data, columns=columns).replace(".", np.nan)
            df = df.apply(force_numeric)

            if df.empty:
                continue

            if category.endswith("_nef_sequence"):
                if "chain_code" in df.columns:
                    unique_chains = df["chain_code"].unique()

                    if len(unique_chains) == 1:
                        sequences[f"{saveframe.name}_{unique_chains[0]}"] = df
                    else:
                        sequences[f"{saveframe.name}_all"] = df
                        for chain in unique_chains:
                            chain_df = df[df["chain_code"] == chain].reset_index(drop=True)
                            sequences[f"{saveframe.name}_{chain}"] = chain_df
                else:
                    sequences[saveframe.name] = df

            elif category.endswith("_nef_chemical_shift"):
                key = saveframe.name.replace("nef_chemical_shift_list_", "")
                shifts[key] = df

            elif category.endswith("_nef_peak"):
                key = saveframe.name.replace("nef_nmr_spectrum_", "").replace("`", "_").strip("_")

                # Attach the spectrum metadata to the DataFrame
                df.attrs = dict(spec_metadata, peaklist_key=key, nef_file=str(filepath))
                peaks[key] = df

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

    if spectra_plot:
        from .spectra_plot import export_spectra_plot

        export_spectra_plot(peaks, output_dir=output_dir, nef_file=filepath)

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

    max_len = max(len(str(name)) for name in peak_dict.keys())

    for name in peak_dict.keys():
        left_side = f"'{name}':"
        padded_left = left_side.ljust(max_len + 5)
        print(f"        {padded_left} '{name}',")

    print("# ---------------------------------------------------\n")


def print_spectrum_menu(peak_dict):
    """
    Displays the current spectrum mapping and provides
    the list 's' used for indexing in the analysis.
    """
    s = list(peak_dict.keys())

    print("\n" + "=" * 25)
    print("      SPECTRUM MENU")
    print("=" * 25)

    for i, name in enumerate(s):
        print(f" {i:2} -> {name}")

    print("=" * 25)

    return s


def report_spectrum_architecture(peak_dict):
    """
    Diagnostic tool to print the dimensionality and atom mapping
    for every spectrum in the dataset using NEF metadata.
    """
    print("\n" + "=" * 40)
    print("      SPECTRUM ARCHITECTURE REPORT")
    print("=" * 40)

    for name, df in peak_dict.items():
        dims = df.attrs.get("dimensions", [])
        exp_type = df.attrs.get("experiment_type", "Unknown")

        print(f"\nSpectrum: {name}")
        print(f"  Type:  {exp_type}")
        print(f"  Size:  {len(dims)}D")

        if dims:
            for i, atom in enumerate(dims):
                print(f"  Axis {i + 1}: {atom} (Mapped to 'position_{i + 1}')")
        else:
            print("  ⚠️ No dimension metadata found for this spectrum.")

    print("\n" + "=" * 40)
