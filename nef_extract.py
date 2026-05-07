import extract_functions as ef

# Open your file and extract the data

FILE_NAME = "from_ccpn.nef"
CYANA_EXPORT = True
CSP_CALCULATION = True

all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(FILE_NAME, spectra_plot=True)

# %% Rename your peaklists

# 1. Run this to get the names
ef.generate_rename_template(all_peaks)
# 2. Paste the result below and modify the right-hand side names if needed
spec_rename_map = {

}

# 3. Apply your rename map
all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}
# 4. Run the menu helper and capture the list 's'
s = ef.print_spectrum_menu(all_peaks)

# %%
if CYANA_EXPORT:
    print("\n--- Exporting to CYANA ---")
    ef.export_cyana_seq(all_sequences)
    ef.export_cyana_prot(all_shifts)
    ef.export_cyana_peaks(all_peaks)
    print("--------------------------\n")

# %%
if CSP_CALCULATION:
    master_pivot = ef.create_master_pivot(all_peaks, ref_spectrum=s[0])
    # Bundle the core dataset into a single dictionary and pass it to the function
    project_data = {"pivot": master_pivot, "sequences": all_sequences, "spectra": s}

    # %%
    analysis_1 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=0,
        spectra_to_analyze=s,
        align=True,
        normalization_factor=0.95,
        filename="analysis_1",
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

    # %%
    analysis_2 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=2,
        spectra_to_analyze=[3, 4],
        align=True,
        normalization_factor=0.95,
        filename="analysis_2",
    )

    ef.plot_nmr_metrics(analysis_2, ylim_csp=None, CSP=True, Int=True, Vol=False)

    # %%
    project_data["pivot"] = "results/spectra_analysis_mod.xlsx"
    analysis_3 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=0,
        spectra_to_analyze=s,
        align=False,
        normalization_factor=0.95,
        filename="analysis_3",
    )

    ef.plot_nmr_metrics(analysis_3, ylim_csp=None, CSP=True, Int=True, Vol=False)
