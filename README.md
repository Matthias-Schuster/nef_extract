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
The CSP is Δδ = √((w₁·Δδ₁)² + (w₂·Δδ₂)²) with the weights 1H = 1, 15N = 0.142 and 13C = 0.33 (taken from the NEF dimensions). Change them with `alpha`, e.g. `alpha={"13C": 0.25}`; typical 13C values are listed in the function description.
The data frames are also saved as excel files.

If the automatic peak matching fails for some residues, use `nef_extract_Template_CSP_excel.py`: copy `spectra_analysis.xlsx` to `spectra_analysis_mod.xlsx`, correct the `Master_Pivot` sheet by hand and pass the file path as `"pivot"`. Keep the `Info` sheet in the file, it stores the spectrum dimensions for the correct CSP weighting (important for 13C spectra) and the noise SDs.

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

**Errors and detection limit (e.g. PRE):** with a noise SD the analysis adds error bars and a detection limit per residue; ratios below it are shown in grey (≈ 0). The SD that CCPN stored in the NEF is used automatically, otherwise the value from `noise=` (see [section 6](#6-noise-for-the-error-bars)).

### 4. Plotting and Visualization
Because the DataFrame tracks its own location metadata, plotting and PDB mapping functions require minimal arguments. 
Additional arguments are described in the function definition. 

* `plot_nmr_metrics` generates bar-plots for individual metrics.
* `plot_combined` generates a dual-axis plot.
* The intensity plots show the analysis settings (normalization, smoothing, error bars, detection limit) in the legend; `show_info=False` hides it.
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
Use `nef_extract_Template_dynamics.py`. The relaxation delays are not stored in the NEF file, so you write them as a list, one delay per spectrum in the same order as `spectra`. Repeated delays are recommended: their scatter defines the error bars (otherwise the errors come from the fit).

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

noe = ef.calc_hetnoe(all_peaks, sat=12, ref=13, noise=(None, None), project_data=project_data)  # noise SD, see section 6
ef.plot_relaxation_rates(noe)           # hetNOE (auto y-range, or e.g. ylim=(-1, 1))
# hetNOE: folded ~0.7-0.85; flexible loops/termini and disordered proteins 0.5 to -1

ef.plot_dynamics_overview([t1, t2, noe])  # stacked R1 / R2 / hetNOE
ef.csp_to_pdb(t2, pdb="input.pdb")        # R2 on the structure, missing residues in grey
```

Rates are given in s⁻¹ and times in ms. The results are saved as `T2_analysis.xlsx` etc. (sheets: Results, Delays, Settings).

### 6. Noise for the error bars
Error bars of intensity ratios (PRE), hetNOE and T1/T2 need the **standard deviation (SD) of the spectral noise**.

In CCPN: right-click in the spectrum → **Estimate Noise...** (`EN`) → value **SD**. Press **Set Noise Level** and CCPN stores the SD in the NEF export. `extract_all_nef_data` reads it, the spectrum menu shows it, and the analyses use it automatically. Estimate every spectrum separately (not *Set Noise Level To All*).

Do **not** use the *Noise Level* (contour base): it is mean + 3.5·SD, so the errors would be about 3.5× too large.

```python
# per spectrum: the SD from the NEF first, otherwise the value given here
analysis = ef.add_analysis_to_master(project_data, ref_spectra=4, spectra_to_analyze=[2, 3], noise=1.3e8)
noe = ef.calc_hetnoe(all_peaks, sat=12, ref=11, noise=(None, None))   # (sat, ref)
```

* Intensities and hetNOE: NEF values automatically. T1/T2: only with `use_nef_noise=True`; repeated delays are usually the better error source.
* A summary after each analysis shows which SD was used for every spectrum.
* All options and the error formulas are described in the functions (`add_analysis_to_master`, `calc_hetnoe`, `fit_relaxation`).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
Distributed under the MIT License.