# nef_extract: NMR Data Extraction & Analysis Pipeline

`nef_extract` is a comprehensive Python toolkit designed to streamline the extraction, analysis, and visualization of NMR data from NEF (NMR Exchange Format) files. 

This pipeline automates tedious tasks such as extracting peak lists, calculating Chemical Shift Perturbations (CSPs), evaluating intensity/volume attenuation (e.g., for PREs), exporting files in CYANA format, and directly mapping NMR metrics onto 3D protein structures using PyMOL.

## ✨ Features

* **NEF Parsing:** Extract sequences, chemical shift lists, and peak lists from standard `.nef` files.
* **Automated CSP & Attenuation Calculations:** Automatically compute CSPs with nucleus-specific scaling factors and calculate signal attenuation (1 - I/I0).
* **Smart Auto-Routing:** Never manually define paths again. The pipeline automatically resolves your project roots, inputs, and results directories based on your script location.
* **Data Export to CYANA:** Instantly convert and export your NMR sequences, chemical shifts, and peak lists into CYANA files in a single command.
* **Advanced Plotting:** Generate bar plots and dual-axis graphs combining CSPs and intensity attenuation.
* **Direct Structural Mapping:** Map computed CSPs or attenuation values directly onto the B-factor column of PDB/CIF files and auto-generate beautifully styled PyMOL sessions (`.pse`).

## ⚙️ Installation

The package relies on standard data science libraries (`pandas`, `numpy`, `matplotlib`) as well as structural biology tools (`biopython`, `biopandas`, `pymol-open-source`, `pynmrstar`). 

To guarantee compatibility, please install the provided conda environment:

```bash
# Clone the repository
git clone [https://github.com/Matthias-Schuster/nef_extract.git](https://github.com/Matthias-Schuster/nef_extract.git)
cd nef_extract

# Create the conda environment from the provided yaml file
conda env create -f environment.yml

# Activate the environment
conda activate nef_extract
```

## 📂 Project Architecture

The pipeline is designed with a clean separation between your analysis scripts and the core logic. When you run your script, the setup dynamically generates organized input and output directories.

```text
nef_extract/
│
├── extract_functions/          # Core backend module
│   ├── __init__.py             # Exposes API functions
│   ├── parsing.py              # NEF extraction & DataFrame handling
│   ├── plotting.py             # Matplotlib data visualization
│   ├── pymol_pdb.py            # PyMOL structure mapping & session generation
│   └── cyana.py                # CYANA format conversions & export logic
│
├── environment.yml             # Conda environment dependencies
│
├── nef_extract/                # ⬅️ Codebase Subfolder
│   ├── nef_extract_Template.py # Your main analysis script (Entry Point)
│   ├── nef_setup.py            # Auto-routing & directory generation script
│
├── input/                      # (Auto-generated) Place your inputs here
│   ├── my_specific_data.nef
│   └── input.pdb
│
└── results/                    # (Auto-generated) Output directory
    └── Template/               # Project-specific subfolder
        ├── analysis_1.xlsx     # Master DataFrames with calculated metrics
        ├── spectra_metadata.txt
        ├── plots/              # Generated bar plots & combined plots
        ├── structures/         # PyMOL .pse sessions & .pdb files
        ├── csv/                # CSV exports
        └── cyana/              # Exported .seq, .prot, and .peaks files
```

## 🚀 Usage

`nef_extract` is designed to be used directly within a Python script. Below is a complete example demonstrating the standard workflow, utilizing the built-in auto-routing setup.

### 1. Setup and Extraction
Load the Nef file and optionally rename the spectra.
Spectra_plot=True generates an input file for the NMR_2D_plot script.
It also generates the spectra menu for the analysis. 
This automatically creates an organized `input/` and `results/` folder structure and handles all pathing.

```python

# 1. System Setup & Auto-Routing
FILE_NAME = "my_specific_data.nef" 
_, PROJECT_NAME, paths = setup_nef_project(__file__, nef_filename=FILE_NAME)

input_nef = paths["input_nef"]
out_dir = paths["project_out"]
input_dir = paths["input_dir"]

# 2. Data Extraction
# spectra_plot=True generates an input file for the NMR_2D_plot script.
all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(
    input_nef, spectra_plot=True, output_dir=out_dir
)

# (Optional) Rename your peaklists for cleaner analysis
# A template will be generated in the terminal
spec_rename_map = {
    "old_spectrum_name": "new spectrum name"
}
all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}

# Select spectra via the interactive menu helper
s = ef.print_spectrum_menu(all_peaks)
```

### 2. Export to CYANA
Export all your sequences, chemical shifts, and peak lists perfectly formatted for CYANA structure calculations using a single wrapper.

```python
print("\n--- Exporting to CYANA ---")
ef.export_cyana_project(all_sequences, all_shifts, all_peaks, output_dir=out_dir / "cyana")
print("--------------------------\n")
```

### 3. Generate the Analysis
First it generates a combined data frame of similar 2D spectra. 
The ref_spectrum=s[0] needs to be defined if you have a mixture of 15N and 13C HSQCs in your nef file. 

In add_analysis_to_master you need to define a ref_spectra and spectra_to_analyze. 
The spectra_to_analyze can be a single spectrum, or a list [1, 2] from the spectra menu or s for all spectra.
The align function automatically adds missing peaks from the sequence of the nef file. 
Additional functions are described in the function definition. 

CSPs and intensities are automatically calculated and stored in the data frame. 
The data frames are also saved as excel files.

By passing `out_dir` and `input_dir` into the `project_data` dictionary, all generated files, plots, and PyMOL sessions will automatically be routed to their correct subfolders inside `results/YOUR_PROJECT/`.

```python
master_pivot = ef.create_master_pivot(all_peaks, output_dir=out_dir, ref_spectrum=s[0])
project_data = {"pivot": master_pivot, "sequences": all_sequences, "spectra": s}

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

### 4. Plotting and Visualization
Because the DataFrame tracks its own location metadata, plotting and PDB mapping functions require minimal arguments. 

* `plot_nmr_metrics` generates bar-plots for individual metrics.
* `plot_combined` generates a dual-axis plot.
* `csp_to_pdb` maps the calculated metrics onto an input structure (automatically found in your project's `input/` folder).

```python
# Generate individual bar plots
ef.plot_nmr_metrics(analysis_1, ylim_csp=0.5, CSP=True, Int=True)

# Generate combined dual-axis plots
ef.plot_combined(analysis_1, ylim_csp=0.5, Int=True)

# Map computed CSPs directly onto a PDB structure (reads .pdb or .cif from your input_dir)
ef.csp_to_pdb(analysis_1, pdb="input.pdb", CSP=True, Int=False)
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
Distributed under the MIT License.