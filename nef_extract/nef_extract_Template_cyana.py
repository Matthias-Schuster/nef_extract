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

# %% =======================================
# 3. DATA EXTRACTION
# ==========================================
all_sequences, all_shifts, all_peaks = ef.extract_all_nef_data(input_nef, report=True)

# %% Rename your peaklists (the names become the .peaks filenames)
ef.generate_rename_template(all_peaks)

spec_rename_map = {
    # Paste your rename mappings here
}

all_peaks = {spec_rename_map.get(k, k): v for k, v in all_peaks.items()}

# %% =======================================
# 4. CYANA EXPORT (.seq, .prot, .peaks)
# ==========================================
ef.export_cyana_project(all_sequences, all_shifts, all_peaks, output_dir=out_dir / "cyana")
