# nef_extract: NMR Data Extraction & Analysis Pipeline

`nef_extract` is a comprehensive Python toolkit designed to streamline the extraction, analysis, and visualization of NMR data from NEF (NMR Exchange Format) files. 

This pipeline automates tedious tasks such as extracting peak lists, calculating Chemical Shift Perturbations (CSPs), evaluating intensity/volume attenuation (e.g., for PREs), exporting files in CYANA format, and directly mapping NMR metrics onto 3D protein structures using PyMOL.

## ✨ Features

* **NEF Parsing:** Extract sequences, chemical shift lists, and peak lists from standard `.nef` files.
* **Automated CSP & Attenuation Calculations:** Automatically compute CSPs with nucleus-specific weights (adjustable per nucleus) and intensity ratios / attenuation (1 - I/I0), optionally with errors and detection limits from the spectral noise (e.g. PRE).
* **Smart Auto-Routing:** The pipeline automatically resolves your project roots, inputs, and results directories based on the name of your script.
* **Data Export to CYANA:** Instantly convert and export your NMR sequences, chemical shifts, and peak lists into CYANA files in a single command.
* **Advanced Plotting:** Generate bar plots and dual-axis graphs combining CSPs and intensity attenuation.
* **Dynamics:** Fit T1/T2 relaxation decays (R1/R2 with errors from duplicate delays, a given noise SD or the fit) and calculate {1H}-15N hetNOEs.
* **Input for the NMR_plot scripts:** Spectrum metadata (paths, contour levels, colours) and peak label CSVs for 1D, 2D and 3D spectra.
* **Direct Structural Mapping:** Map computed CSPs, attenuation, R1/R2 or hetNOE values directly onto the B-factor column of PDB/CIF files and auto-generate beautifully styled PyMOL sessions (`.pse`).

## ⚙️ Installation

The package relies on standard data science libraries (`pandas`, `numpy`, `matplotlib`) as well as structural biology tools (`biopython`, `biopandas`, `pymol-open-source`, `pynmrstar`). 

To guarantee compatibility, please install the provided conda environment:

```bash
# Clone the repository
git clone https://github.com/Matthias-Schuster/nef_extract.git
cd nef_extract

# Create the conda environment from the provided yaml file
conda env create -f environment.yml

# Activate the environment
conda activate nef_extract


# Alternatively you can compare or update your environment 
conda compare -n your_env_name environment.yml
conda env update -n your_env_name -f environment.yml
```

## 📂 Project Architecture

The pipeline is designed with a clean separation between your analysis scripts and the core logic. When you run your script, the setup dynamically generates organized input and output directories.

```text
nef_extract/
│
├── extract_functions/          # Core backend module
│   ├── __init__.py             # Exposes API functions
│   ├── parsing.py              # NEF extraction (sequences, shifts, peak lists)
│   ├── csp.py                  # Master pivot, CSP & intensity analysis
│   ├── spectra_plot.py         # Input files for the plot scripts (2D and 1D/2D/3D)
│   ├── utils.py                # Shared helper functions
│   ├── dynamics.py             # T1/T2 relaxation fits & hetNOE
│   ├── plotting.py             # Matplotlib data visualization
│   ├── pymol_pdb.py            # PyMOL structure mapping & session generation
│   └── cyana.py                # CYANA format conversions & export logic
│
├── environment.yml             # Conda environment dependencies
│
├── nef_extract/                # ⬅️ Codebase Subfolder
│   ├── nef_extract_Template_all.py       # Everything (switch parts on/off)  ┐
│   ├── nef_extract_Template_cyana.py     # CYANA export only                 │ COPY AND
│   ├── nef_extract_Template_CSP.py       # CSPs / intensities                │ RENAME !
│   ├── nef_extract_Template_CSP_excel.py # CSPs from a hand-corrected pivot  │
│   ├── nef_extract_Template_dynamics.py  # T1, T2, hetNOE                    ┘
│   ├── nef_setup.py            # Auto-routing & directory generation script
│
├── input/                      # Place your inputs (.nef and .pdb) here
│   ├── my_specific_data.nef
│   └── input.pdb
│
└── results/                    # Output directory
    └── Template/               # Project-specific subfolder 
        ├── analysis_1.xlsx     # Master DataFrames with calculated metrics
        ├── spectra_metadata.txt      # 2D spectra for the NMR_plot scripts
        ├── spectra_metadata_all.txt  # 1D, 2D and 3D spectra
        ├── plots/              # Generated bar plots & combined plots
        ├── structures/         # PyMOL .pse sessions & .pdb files
        ├── csv/                # Peak label CSVs for the NMR_plot scripts
        └── cyana/              # Exported .seq, .prot, and .peaks files
```

## 🚀 Usage

`nef_extract` is designed to be used directly within a Python script. Below is a complete example demonstrating the standard workflow, utilizing the built-in auto-routing setup.

### 1. Setup and Extraction
Pick the template that fits your project (`_all`, `_cyana`, `_CSP`, `_CSP_excel` or `_dynamics`), copy it to `nef_extract_<YourProject>.py` and change the FILE_NAME to your nef file. Copy your nef file into the input folder. The part after `nef_extract_` becomes the name of your results folder.

Execute the script, optionally rename the spectra.
`export_spectra_plot` generates all input files for the NMR_plot scripts (NMR_2D_plot repository). Call it after renaming, so the metadata and the CSV files use the same new names.

* `spectra_metadata.txt`: only the 2D spectra.
* `spectra_metadata_all.txt`: all 1D, 2D and 3D spectra in the input format of `NMR_plot_Template_all.py`, sorted into `# --- 1D ---` / `# --- 2D ---` / `# --- 3D ---` blocks. 1D spectra get `1` as contour (= scaling factor), 3D spectra and spectra without peaks have no label CSV. With `nef_file=input_nef` it also lists spectra without peaks.
* `csv/<name>_pos.csv`: peak positions. 2D: one peak per residue. 1D/3D: every peak, with `position_1..n` and the assigned atom per dimension (`atom_1..n`, e.g. `A24 HA`).

The spectrum menu gives you the spectrum numbers for the analysis.

```python

# 1. Select your file
FILE_NAME = "my_specific_data.nef" 

# 2. Data Extraction
all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(input_nef)

# (Optional) Rename your peaklists for cleaner analysis
# A template will be generated in the terminal
spec_rename_map = {
    "old_spectrum_name": "new spectrum name"
}
all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}

# Select spectra via the interactive menu helper
s = ef.print_spectrum_menu(all_peaks)

# Input files for the plot scripts: spectra_metadata.txt (2D), spectra_metadata_all.txt (1D/2D/3D), csv/*_pos.csv
ef.export_spectra_plot(all_peaks, output_dir=out_dir, nef_file=input_nef)
```

`extract_all_nef_data(..., spectra_plot=True, output_dir=out_dir)` still works and runs the same export directly with the original NEF spectrum names.

### 2. Export to CYANA
Export all your sequences, chemical shifts, and peak lists perfectly formatted for CYANA structure calculations using a single wrapper.

```python
ef.export_cyana_project(all_sequences, all_shifts, all_peaks, output_dir=out_dir / "cyana")
```

### 3. Generate the Analysis
First it generates a combined data frame of similar 2D spectra. 
The ref_spectrum=s[0] needs to be defined if you have a mixture of 15N and 13C HSQCs in your nef file. 

In add_analysis_to_master you need to define a ref_spectra and spectra_to_analyze. 
The spectra_to_analyze can be a single spectrum, or a list [1, 2] from the spectra menu or s for all spectra.
With `align=True` all residues of the NEF sequence are added (residues without peaks stay empty). 
Additional functions are described in the function definition. 

CSPs and intensities are automatically calculated and stored in the data frame. 
The CSP is Δδ = √((w₁·Δδ₁)² + (w₂·Δδ₂)²) (no factor ½). The weights are set from the NEF dimensions: 1H = 1, 15N = 0.142, 13C = 0.33. Change them with `alpha`: one number for every non-1H dimension (`alpha=0.25`) or per nucleus (`alpha={"15N": 0.14, "13C": 0.25}`, needed for CON spectra). Typical 13C weights depend on the carbon: CO ~0.33, Cα ~0.25, methyl ~0.25 (0.2–0.3), aromatic ~0.3. 
The data frames are also saved as excel files.

If the automatic peak matching fails for some residues, use `nef_extract_Template_CSP_excel.py`: copy `spectra_analysis.xlsx` to `spectra_analysis_mod.xlsx`, correct the `Master_Pivot` sheet by hand and pass the file path as `"pivot"`. Keep the `Info` sheet in the file, it stores the spectrum dimensions for the correct CSP weighting (important for 13C spectra).

All generated files, plots, and PyMOL sessions will automatically be routed to their correct subfolders inside `results/YOUR_PROJECT/`.

```python
master_pivot = ef.create_master_pivot(all_peaks, output_dir=out_dir, ref_spectrum=s[0])
project_data = {"pivot": master_pivot, "sequences": all_sequences, "spectra": s, "out_dir": out_dir, "input_dir": input_dir}

# Generate an analysis
# CSPs and intensities are automatically calculated, stored in a DataFrame, and saved as an Excel file.
analysis_1 = ef.add_analysis_to_master(
    project_data,
    ref_spectra=0,
    spectra_to_analyze=s,
    normalization_factor=0.95,
    filename="analysis_1" 
)
```

**Errors and detection limit (e.g. PRE):** pass the spectral noise as standard deviation (SD, see [Noise for the error bars](#6-noise-for-the-error-bars)) with `noise=1.3e8` (all spectra), a list with one value per spectrum in `spectra_to_analyze` (`None` allowed), or a dict `{spectrum name or menu number: value}`; the reference gets `noise_ref=` (or the value from `noise`). A noise SD stored by CCPN in the NEF file is used automatically and wins over these values. A summary table shows the SD and its source for every spectrum, whether errors were calculated and how many residues are below the detection limit:

```text
📏 Noise SD – reference 25-MTS-AA-hires_2: 1.24e+08 (NEF)
   spectrum           noise SD           errors   below LOD (3σ)
   25-MTS-hires_2     1.22e+08 (NEF)     yes      14
   25-MTS-C-hires_4   1.3e+08 (given)    yes      11
``` The analysis then adds the columns `Norm_Height_Err` (σ = √(σ_p² + r²·σ_ref²) / |I_ref|) and `Norm_Height_LOD` (detection limit 3σ_p / |I_ref|, change with `lod_sigma`). Ratios below the LOD cannot be distinguished from 0 (fully broadened peak); `plot_nmr_metrics` draws these bars (and negative ratios) in grey and adds error bars. Intensity ratios keep their sign, so the noise of a vanished peak scatters around 0 instead of being turned positive. With `smoothing_window` the signed values are averaged first and only then set to ≥ 0; the error of the mean is √(Σσᵢ²)/n.

### 4. Plotting and Visualization
Because the DataFrame tracks its own location metadata, plotting and PDB mapping functions require minimal arguments. 
Additional arguments are described in the function definition. 

* `plot_nmr_metrics` generates bar-plots for individual metrics.
* `plot_combined` generates a dual-axis plot.
* With `show_info=True` (default) the intensity plots get a legend inside the plot with the analysis settings: normalization quantile, smoothing window, source of the error bars (noise SD from the NEF or given) and the grey bars below the detection limit. `show_info=False` hides it.
* `csp_to_pdb` maps the calculated metrics onto an input structure (automatically found in your project's `input/` folder).

```python
# Generate individual bar plots
ef.plot_nmr_metrics(analysis_1, ylim_csp=0.5, CSP=True, Int=True)

# Generate combined dual-axis plots
ef.plot_combined(analysis_1, ylim_csp=0.5, Int=True)

# Map computed CSPs directly onto a PDB structure (reads .pdb or .cif from your input_dir)
ef.csp_to_pdb(analysis_1, pdb="input.pdb", CSP=True, Int=False)
```

### 5. Dynamics (T1, T2, hetNOE)
Use `nef_extract_Template_dynamics.py`. The relaxation delays are not stored in the NEF file, so you write them as a list, one delay per spectrum in the same order as `spectra`. Repeated delays are allowed: they are used to estimate the spectral noise, which then defines the error bars. Alternatively pass a known noise SD with `noise=` (see section 6). Without repeats and without `noise`, the errors come from the fit.

Peaks are matched between the spectra via their assignment (or the CCPN `@` name if unassigned).

```python
t2 = ef.fit_relaxation(
    all_peaks,
    delays=[17, 34, 51, 85, 136, 17],   # ms, one per spectrum
    spectra=[0, 1, 2, 3, 4, 5],         # numbers from the spectrum menu
    experiment="T2",                    # "T1", "T2", "T1rho", ...
    project_data=project_data,
    model="exp2",                       # "exp3" adds an offset: I0·exp(-R·t) + C
    align=True,
)
ef.plot_relaxation_curves(t2)           # every decay with its fit
ef.plot_relaxation_rates(t2)            # R2 (blue) and T2 (green) per residue, two figures
ef.plot_relaxation_rates(t2, show="time", ylim=400)   # only T2 (ms); show="rate" for only R2

noe = ef.calc_hetnoe(all_peaks, sat=12, ref=13, noise=(None, None), project_data=project_data)  # noise SD: NEF first, see section 6
ef.plot_relaxation_rates(noe)           # hetNOE (auto y-range, or e.g. ylim=(-1, 1))
# hetNOE: folded ~0.7-0.85; flexible loops/termini and disordered proteins 0.5 to -1

ef.plot_dynamics_overview([t1, t2, noe])  # stacked R1 / R2 / hetNOE
ef.csp_to_pdb(t2, pdb="input.pdb")        # R2 on the structure, missing residues in grey
```

Rates are given in s⁻¹ and times in ms. The results are saved as `T2_analysis.xlsx` etc. (sheets: Results, Delays, Settings).

### 6. Noise for the error bars
`add_analysis_to_master`, `calc_hetnoe` and `fit_relaxation` take `noise=...`: the **standard deviation (SD) of the spectral noise** in the same units as the peak heights.

How to get it in CCPN:
1. Right-click in the spectrum display → **Estimate Noise...** (shortcut `EN`).
2. Choose *Visible Area* and zoom into a region without peaks, or *Random Sampling* for the whole spectrum.
3. Take the value **SD**, one value per spectrum (sat/ref, para/dia can differ).

**Directly from the NEF file:** if you press **Set Noise Level** in the Estimate Noise popup, CCPN stores the SD of that spectrum and writes it into the NEF export (CCPN internal data, not part of the NEF standard). `extract_all_nef_data` reads it and the spectrum menu shows it:

```text
 11 -> 711-1_1   (noise SD 1.48e+07)
 12 -> 711-2_1   (noise SD 1.11e+07)
```

Rule for every spectrum: **the SD stored in the NEF is used first, otherwise the value you pass with `noise`.** So you can always write your fallback values and the NEF values win where they exist:

```python
analysis = ef.add_analysis_to_master(project_data, ref_spectra=1, spectra_to_analyze=0, noise=None)        # only NEF values
analysis = ef.add_analysis_to_master(project_data, ref_spectra=4, spectra_to_analyze=[2, 3], noise=1.3e8)  # NEF, else 1.3e8
noe = ef.calc_hetnoe(all_peaks, sat=12, ref=11, noise=(None, None))          # (sat, ref): NEF, else nothing
noe = ef.calc_hetnoe(all_peaks, sat=12, ref=11, noise=(1.1e7, 1.3e7))        # NEF, else these numbers
```

The output shows where each value comes from, e.g. `📏 Noise SD: sat 711-2_1 1.11e+07 (NEF), ref 711-1_1 1.3e+07 (given)`.

* Intensities (`add_analysis_to_master`) and hetNOE use the NEF values automatically; `use_nef_noise=False` switches it off (then only `noise` counts).
* T1/T2 (`fit_relaxation`) only with `use_nef_noise=True`, because repeated delays are usually the more realistic error source.
* The SD is also saved in the `Info` sheet of `spectra_analysis.xlsx`, so it works with a hand-corrected pivot too.
* Estimate each spectrum separately. *Set Noise Level To All* copies the SD of one spectrum to all others; a warning appears if many spectra have exactly the same SD.
* The stored SD is not updated if you change the Noise Level by hand later.

Do **not** use the *Noise Level* (Spectrum Properties, contour base, peak-picking threshold). CCPN sets it to mean + 3.5·SD, so it is about 3.5× too large and makes the errors and detection limits far too big. If you only have an estimated Noise Level that was not changed by hand: SD ≈ (Noise Level − Mean) / 3.5.

Error formulas (σ = noise SD, r = ratio):
* Intensity ratio / PRE: σ_r = √(σ_p² + r²·σ_ref²) / |I_ref|, detection limit 3σ_p / |I_ref|
* hetNOE: σ_NOE = √(σ_sat² + NOE²·σ_ref²) / |I_ref|
* T1/T2: σ is the uncertainty of every point in the weighted fit; σ_R = √(cov) of the fit, σ_T = 1000·σ_R / R² (ms)

T1/T2 errors (`fit_relaxation`):

| `noise` | `error_method` | σ of every point |
|---|---|---|
| given (or `use_nef_noise=True`) | any | the given / NEF noise SD (duplicates are not used; a hint appears if the duplicates scatter clearly more) |
| – | `"auto"` (default) | pooled scatter of the repeated delays, or the fit residuals if there are no repeats |
| – | `"duplicates"` | pooled scatter of the repeated delays |
| – | `"fit"` | scatter of the points around the fitted curve |

With repeated delays it is usually better to leave out `noise`: the duplicates contain the noise **and** drift / peak-height errors between the spectra. With `scale_errors=True` (default) the errors of a peak are multiplied by √χ²_red if its points scatter more around the fit than σ explains (χ²_red > 1); they are never made smaller. The columns `("Fit", "chi2_red")` and `("Fit", "error_scale")` show χ² and the applied factor.

hetNOE errors use the noise SD stored in the NEF, otherwise the given `noise`, otherwise the peak uncertainties in the NEF file (if present).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
Distributed under the MIT License.