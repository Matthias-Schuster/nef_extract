from nef_setup import setup_nef_project

# ==========================================
# 1. AUTO-ROUTING, SYSTEM SETUP & FILE CHECK
# ==========================================
# Special case: CSP analysis with a hand-corrected master pivot (Excel).
# Use this if the automatic peak matching does not work for some residues.
FILE_NAME = "my_specific_data.nef"

PROJECT_NAME, paths = setup_nef_project(__file__, nef_filename=FILE_NAME)
import extract_functions as ef

# ==========================================
# 2. I/O HUB
# ==========================================
input_nef = paths["input_nef"]
out_dir = paths["project_out"]
input_dir = paths["input_dir"]

# %% =======================================
# 3. DATA EXTRACTION
# ==========================================
all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(input_nef)

# %% Rename your peaklists
ef.generate_rename_template(all_peaks)

spec_rename_map = {
    # Paste your rename mappings here
}

all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}
s = ef.print_spectrum_menu(all_peaks)

# %% Input files for the plot scripts: spectra_metadata.txt (2D), spectra_metadata_all.txt (1D/2D/3D), csv/*_pos.csv
ef.export_spectra_plot(all_peaks, output_dir=out_dir, nef_file=input_nef)

# %% =======================================
# 4. MASTER PIVOT -> EXCEL
# ==========================================
# Writes results/<project>/spectra_analysis.xlsx
ef.create_master_pivot(all_peaks, output_dir=out_dir, ref_spectrum=s[0])

# %% =======================================
# 5. CORRECT THE PIVOT BY HAND
# ==========================================
# 1. Copy spectra_analysis.xlsx to spectra_analysis_mod.xlsx (same folder).
#    Copy it, otherwise re-running step 4 overwrites your changes.
# 2. Correct the 'Master_Pivot' sheet (first sheet), e.g. move peaks to the right residue.
#    Keep the header rows and the first three columns (chain, sequence code, residue name).
# 3. Keep the 'Info' sheet: it stores the spectrum dimensions for the CSP weighting.
#    Without it, a 1H/15N spectrum is assumed (for 13C spectra pass alpha={"13C": 0.33}).

MOD_PIVOT = out_dir / "spectra_analysis_mod.xlsx"

project_data = {
    "pivot": MOD_PIVOT,
    "sequences": all_sequences,
    "spectra": s,
    "out_dir": out_dir,
    "input_dir": input_dir,
}

# %% =======================================
# 6. ANALYSES ON THE CORRECTED PIVOT
# ==========================================
# %% Analysis 1
analysis_1 = ef.add_analysis_to_master(
    project_data,
    ref_spectra=0,
    spectra_to_analyze=s,
    align=True,
    normalization_factor=0.95,
    # alpha={"15N": 0.142, "13C": 0.33},  # CSP weights (default); 13C: CO 0.33, Cα/methyl ~0.25
    noise=None,          # <- noise SD: NEF value first, else this number / list (one per spectrum) / dict
    filename="analysis_1_mod",
)

ef.plot_nmr_metrics(analysis_1, ylim_csp=None, CSP=True, Int=True, Vol=False)

ef.plot_combined(analysis_1, ylim_csp=None, Int=True, Vol=False)

ef.csp_to_pdb(analysis_1, pdb="input.pdb", CSP=True, Int=False, Vol=False)
