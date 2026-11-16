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
    "out_dir": out_dir,
    "input_dir": input_dir,
}

# %% Input files for the plot scripts: spectra_metadata.txt (2D), spectra_metadata_all.txt (1D/2D/3D), csv/*_pos.csv
ef.export_spectra_plot(all_peaks, output_dir=out_dir, nef_file=input_nef)

# %% =======================================
# 4. T2 (R2)
# ==========================================
# delays:  one delay per spectrum, in the same order as 'spectra' (delay_unit="ms" or "s").
#          Repeated delays are allowed and are used to estimate the errors.
# model:   "exp2" = I0·exp(-R·t)   |   "exp3" = I0·exp(-R·t) + C
# error_method: "auto" (duplicates if available, else fit) | "duplicates" | "fit"
#          noise=<SD> overrides error_method; use_nef_noise=True takes the SD stored by CCPN.
#          With repeated delays better leave both out (duplicates also contain drift, not only noise).
# scale_errors: True (default) = errors are scaled up by √χ² if the points scatter more
#          around the fit than σ explains (σ is a lower limit, errors never get smaller).
t2 = ef.fit_relaxation(
    all_peaks,
    delays=[17, 34, 51, 85, 136, 17],   # <- your T2 delays in ms
    spectra=[0, 1, 2, 3, 4, 5],         # <- spectrum numbers from the menu (or s for all)
    experiment="T2",
    project_data=project_data,
    model="exp2",
    error_method="auto",
    align=True,
    filename="T2_analysis",
)

ef.plot_relaxation_curves(t2)
ef.plot_relaxation_rates(t2)   # R2.png + T2.png (show="rate" or "time" for only one)
ef.csp_to_pdb(t2, pdb="input.pdb")

# %% =======================================
# 5. T1 (R1)
# ==========================================
t1 = ef.fit_relaxation(
    all_peaks,
    delays=[20, 100, 300, 600, 1000, 100],   # <- your T1 delays in ms
    spectra=[6, 7, 8, 9, 10, 11],            # <- your T1 spectra
    experiment="T1",
    project_data=project_data,
    align=True,
    filename="T1_analysis",
)

ef.plot_relaxation_curves(t1)
ef.plot_relaxation_rates(t1)
ef.csp_to_pdb(t1, pdb="input.pdb")

# %% =======================================
# 6. {1H}-15N hetNOE
# ==========================================
# sat / ref: the saturated spectrum has the weaker (flexible regions: negative) peaks,
#            the reference is all positive. Values far above 1 mean sat and ref are swapped.
# noise:     (sat, ref) noise SD for the error bars. The SD that CCPN stored in the NEF is used
#            first (spectrum menu shows it; CCPN: Estimate Noise -> 'Set Noise Level' before the
#            export). Your numbers are only used for spectra without a stored SD; None = none.
#            Numbers: CCPN Estimate Noise (EN) -> 'SD', not the 'Noise Level' (= mean + 3.5 SD).
# Typical values: folded ~0.7-0.85, flexible loops/termini and disordered proteins 0.5 to -1.
# For repeated measurements pass lists: sat=[12, 14], ref=[13, 15]
noe = ef.calc_hetnoe(
    all_peaks,
    sat=12,           # <- saturated spectrum
    ref=13,           # <- reference spectrum
    noise=(None, None),   # <- (sat, ref): NEF SD if stored, else these numbers
    project_data=project_data,
    align=True,
    filename="hetNOE_analysis",
)

ef.plot_relaxation_rates(noe)   # auto y-range; fixed range e.g. ylim=(-1, 1)
ef.csp_to_pdb(noe, pdb="input.pdb")   # auto colour scale (min to max); or min_val=-1, max_val=1

# %% =======================================
# 7. OVERVIEW
# ==========================================
ef.plot_dynamics_overview([t1, t2, noe])
