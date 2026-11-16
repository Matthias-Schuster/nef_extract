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

SPECTRA_PLOT = True      # input files for the NMR_plot scripts (1D/2D/3D)
CYANA_EXPORT = True      # .seq, .prot and .peaks files for CYANA
CSP_CALCULATION = True   # CSPs and intensity ratios
DYNAMICS = False         # T1, T2 and hetNOE (set your delays and spectrum numbers first!)

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

project_data = {
    "sequences": all_sequences,
    "spectra": s,
    "out_dir": out_dir,
    "input_dir": input_dir,
}

# %% Input files for the plot scripts: spectra_metadata.txt (2D), spectra_metadata_all.txt (1D/2D/3D), csv/*_pos.csv
if SPECTRA_PLOT:
    ef.export_spectra_plot(all_peaks, output_dir=out_dir, nef_file=input_nef)

# %% =======================================
# 4. CYANA EXPORT
# ==========================================
if CYANA_EXPORT:
    ef.export_cyana_project(all_sequences, all_shifts, all_peaks, output_dir=out_dir / "cyana")

# %% =======================================
# 5. CSP / INTENSITY ANALYSIS
# ==========================================
if CSP_CALCULATION:
    project_data["pivot"] = ef.create_master_pivot(
        all_peaks, output_dir=out_dir, ref_spectrum=s[0]
    )

    # %% Analysis 1
    analysis_1 = ef.add_analysis_to_master(
        project_data,
        ref_spectra=0,
        spectra_to_analyze=s,
        align=True,
        normalization_factor=0.95,
        # alpha={"15N": 0.142, "13C": 0.33},  # CSP weights (default); 13C: CO 0.33, Cα/methyl ~0.25
        noise=None,          # <- noise SD for error bars + detection limit (CCPN: Estimate Noise -> SD)
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

# %% =======================================
# 6. DYNAMICS (T1, T2, hetNOE)
# ==========================================
if DYNAMICS:
    # One delay per spectrum, same order as 'spectra'. Repeated delays are used for the errors.
    t2 = ef.fit_relaxation(
        all_peaks,
        delays=[17, 34, 51, 85, 136, 17],   # <- your T2 delays in ms
        spectra=[0, 1, 2, 3, 4, 5],         # <- your T2 spectra (numbers from the spectrum menu)
        experiment="T2",
        project_data=project_data,
        align=True,
    )
    ef.plot_relaxation_curves(t2)
    ef.plot_relaxation_rates(t2)

    t1 = ef.fit_relaxation(
        all_peaks,
        delays=[20, 100, 300, 600, 1000, 100],   # <- your T1 delays in ms
        spectra=[6, 7, 8, 9, 10, 11],            # <- your T1 spectra
        experiment="T1",
        project_data=project_data,
        align=True,
    )
    ef.plot_relaxation_curves(t1)
    ef.plot_relaxation_rates(t1)

    # hetNOE: sat = weaker/negative peaks, ref = all positive (values far above 1 -> swapped).
    # Typical: folded ~0.7-0.85, flexible loops/termini and disordered proteins 0.5 to -1.
    # noise: SD of each spectrum (CCPN: right-click -> Estimate Noise (EN) -> 'SD',
    #        not the 'Noise Level' = mean + 3.5 SD)
    noise_sat = None      # <- noise SD of the saturated spectrum
    noise_ref = None      # <- noise SD of the reference spectrum

    noe = ef.calc_hetnoe(
        all_peaks,
        sat=12,           # <- saturated spectrum
        ref=13,           # <- reference spectrum
        noise=(noise_sat, noise_ref) if noise_sat and noise_ref else None,
        project_data=project_data,
        align=True,
    )
    ef.plot_relaxation_rates(noe)   # auto y-range; fixed range e.g. ylim=(-1, 1)

    ef.plot_dynamics_overview([t1, t2, noe])
    ef.csp_to_pdb(t2, pdb="input.pdb")
