from nef_setup import setup_nef_project

# ==========================================
# 1. AUTO-ROUTING, SYSTEM SETUP & FILE CHECK
# ==========================================
FILE_NAME = "my_specific_data.nef"

PROJECT_NAME, paths = setup_nef_project(__file__, nef_filename=FILE_NAME)
import extract_functions as ef

# ==========================================
# 2. I/O HUB
# ==========================================
input_nef = paths["input_nef"]
out_dir = paths["project_out"]
input_dir = paths["input_dir"]

CYANA_EXPORT = True
CSP_CALCULATION = True

# %% =======================================
# 3. DATA EXTRACTION
# ==========================================
all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(
    input_nef, spectra_plot=True, output_dir=out_dir
)

# %% Rename your peaklists
ef.generate_rename_template(all_peaks)

spec_rename_map = {
    # Paste your rename mappings here
}

all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}
s = ef.print_spectrum_menu(all_peaks)

# %%
if CYANA_EXPORT:
    ef.export_cyana_project(all_sequences, all_shifts, all_peaks, output_dir=out_dir / "cyana")

# %%
if CSP_CALCULATION:
    master_pivot = ef.create_master_pivot(all_peaks, output_dir=out_dir, ref_spectrum=s[0])
    project_data = {"pivot": master_pivot, "sequences": all_sequences, "spectra": s}

    # %% Analysis 1
    analysis_1 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=0,
        spectra_to_analyze=s,
        align=True,
        normalization_factor=0.95,
        filename=out_dir / "analysis_1",
    )

    ef.plot_nmr_metrics(
        analysis_1,
        ylim_csp=0.5,
        CSP=True,
        Int=True,
        Vol=False,
        color_csp="#1f77b4",
        color_int="goldenrod",
    )

    ef.plot_combined(
        analysis_1,
        ylim_csp=0.5,
        Int=True,
        Vol=False,
        color_csp="#1f77b4",
        color_int="goldenrod",
    )

    ef.csp_to_pdb(analysis_1, pdb="input.pdb", CSP=True, Int=False, Vol=False)

    # %% Analysis 2
    analysis_2 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=2,
        spectra_to_analyze=[3, 4],
        align=True,
        normalization_factor=0.95,
        filename=out_dir / "analysis_2",
    )

    ef.plot_nmr_metrics(analysis_2, ylim_csp=None, CSP=True, Int=True, Vol=False)

    # %% Analysis 3
    # Point the modified pivot to the newly generated results folder
    project_data["pivot"] = out_dir / "spectra_analysis_mod.xlsx"

    analysis_3 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=0,
        spectra_to_analyze=s,
        align=False,
        normalization_factor=0.95,
        filename=out_dir / "analysis_3",
    )

    ef.plot_nmr_metrics(analysis_3, ylim_csp=None, CSP=True, Int=True, Vol=False)
