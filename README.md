# nef_extract: NMR Data Extraction & Analysis Pipeline

`nef_extract` is a comprehensive Python toolkit designed to streamline the extraction, analysis, and visualization of NMR data from NEF (NMR Exchange Format) files. 

This pipeline automates tedious tasks such as extracting peak lists, calculating Chemical Shift Perturbations (CSPs), evaluating intensity/volume attenuation (e.g., for PREs), exporting files in CYANA format, and directly mapping NMR metrics onto 3D protein structures using PyMOL.

## ✨ Features

* **NEF Parsing:** Extract sequences, chemical shift lists, and peak lists from standard `.nef` files.
* **Automated CSP & Attenuation Calculations:** Automatically compute CSPs with nucleus-specific scaling factors and calculate signal attenuation (1 - I/I0).
* **Data Export to CYANA:** Instantly convert and export your NMR sequences, chemical shifts, and peak lists into CYANA files.
* **Advanced Plotting:** Generate bar plots and dual-axis graphs combining CSPs and intensity attenuation.
* **Direct Structural Mapping:** Map computed CSPs or attenuation values directly onto the B-factor column of PDB/CIF files and auto-generate beautifully styled PyMOL sessions (`.pse`).

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
```

## 🚀 Usage
nef_extract is designed to be used directly within a Python script, allowing you to easily build your own data processing pipelines.

Below is a complete example demonstrating the standard workflow, from extracting NEF data to exporting CYANA files and mapping metrics to a PDB structure:

### Load the nef file
Load the Nef file and optionally rename the spectra.
Spectra_plot=True generates an input file for the NMR_2D_plot script.
It also generates the spectra menu for the analysis.

```Python
# Extract data from your NEF file
FILE_NAME = "from_ccpn.nef"
all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(FILE_NAME, spectra_plot=True)

# (Optional) Rename your peaklists for cleaner analysis
# A template will be generated in the terminal
spec_rename_map = {
    "old_spectrum_name": "new spectrum name"
}
all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}

# Select spectra via the interactive menu helper
s = ef.print_spectrum_menu(all_peaks)

# Export to CYANA format
print("\n--- Exporting to CYANA ---")
ef.export_cyana_seq(all_sequences)
ef.export_cyana_prot(all_shifts)
ef.export_cyana_peaks(all_peaks)
print("--------------------------\n")
```
### Generate the analysis
First it generates a combined data frame of similar 2D spectra. 
The ref_spectrum=s[0] needs to be defined if you have a mixture of 15N and 13C HSQCs in your nef file. 

In add_analysis_to_master you need to define a ref_spectra and spectra_to_analyze. 
The spectra_to_analyze can be a single spectrum, or a list [1, 2] from the spectra menu or s for all spectra.
The align function automatically adds missing peaks from the sequence of the nef file. 
Additional functions are described in the function definition. 

CSPs and intensities are automatically calculated and stored in the data frame. 
The data frames are also saved as excel files.

```Python
master_pivot = ef.create_master_pivot(all_peaks, ref_spectrum=s[0])
project_data = {'pivot': master_pivot, 'sequences': all_sequences, 'spectra': s}

# Generate an analysis
analysis_1 = ef.add_analysis_to_master(project_data,
                                       ref_spectra=0,
                                       spectra_to_analyze=s,
                                       align=True,
                                       normalization_factor=0.95,
                                       filename="analysis_1")
```
### Plotting and Visualization
plot_nmr_metrics generates a bar-plot for the calculated metrics.
plot_combined plots a combined plot for the metrics.
Colors and y-limits can be manually selected. 
csp_to_pdb maps the calculated metrics on an input.pdb (or .cif) file and saves it as a sausage plot representation. 


```Python
# 5. Plotting and Visualization
# Generate dual-axis bar plots for CSPs and Intensity
ef.plot_nmr_metrics(analysis_1, ylim_csp=0.5, CSP=True, Int=True, Vol=False,
                    color_csp="#1f77b4", color_int="goldenrod")

ef.plot_combined(analysis_1, ylim_csp=0.5, Int=True, Vol=False,
                 color_csp="#1f77b4", color_int="goldenrod")

# Map computed CSPs directly onto a PDB structure
ef.csp_to_pdb(analysis_1, pdb="input.pdb", CSP=True, Int=False, Vol=False)
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
Distributed under the MIT License.