import numpy as np
from pathlib import Path

# =============================================================================
# CSP to PDB Functions
# =============================================================================


def csp_to_pdb(
    analysis_df,
    pdb=None,
    output_dir="results/structures",
    max_val=None,
    CSP=True,
    Int=False,
    Vol=False,
):
    """
    Maps NMR metrics (CSPs, Intensity, or Volume) onto a structure, generates
    PyMOL sessions, and cleans up all temporary conversion files.
    """
    from Bio.PDB import MMCIFParser, PDBIO
    from biopandas.pdb import PandasPdb
    from pymol import cmd

    # --- FAILSAFE 1: Path & Existence ---
    if pdb is None:
        print("❌ Error: No structure file provided.")
        return

    pdb_path = Path(pdb)
    if not pdb_path.exists():
        print(f"❌ Error: File not found: {pdb_path}")
        return

    suffix = pdb_path.suffix.lower()
    if suffix not in [".pdb", ".cif"]:
        print(f"❌ Error: {suffix} is not a valid format. Use .pdb or .cif")
        return

    # Update output directory dynamically
    out_path = Path(output_dir)
    analysis_name = analysis_df.attrs.get("analysis_name", "")
    if analysis_name:
        out_path = out_path / analysis_name
    out_path.mkdir(parents=True, exist_ok=True)

    is_cif = suffix == ".cif"
    working_pdb = str(out_path / f"TEMP_CONV_{pdb_path.stem}.pdb")

    # --- Configuration Map ---
    config = {
        "CSPs": {"active": CSP, "prefix": "CSP", "calc_attenuation": False},
        "Norm_Height": {"active": Int, "prefix": "Int", "calc_attenuation": True},
        "Norm_Volume": {"active": Vol, "prefix": "Vol", "calc_attenuation": True},
        "Ratio_Height": {"active": Int, "prefix": "Int", "calc_attenuation": True},
        "Ratio_Volume": {"active": Vol, "prefix": "Vol", "calc_attenuation": True},
    }

    # Filter for metrics that are requested AND exist in the DataFrame
    active_metrics = [
        cat for cat, s in config.items() if s["active"] and cat in analysis_df.columns.levels[0]
    ]

    try:
        # --- 1. PREPARATION & CONVERSION ---
        if is_cif:
            print(f"🔄 Converting {pdb_path.name} for processing...")
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("struct", str(pdb_path))
            io = PDBIO()
            io.set_structure(structure)
            io.save(working_pdb)
        else:
            working_pdb = str(pdb_path)

        # --- 2. DATA VALIDATION ---
        if not active_metrics:
            print("⚠️ No selected metric data found in 'analysis_df'.")
            return

        # --- FAILSAFE 3: Residue Mapping Overlap ---
        try:
            temp_check = PandasPdb().read_pdb(working_pdb)

            # 1. Extract PDB (Number, Name) pairs
            pdb_df = temp_check.df["ATOM"]
            pdb_residues = set(zip(pdb_df["residue_number"], pdb_df["residue_name"].str.upper()))

            # 2. Extract DataFrame (Number, Name) pairs from the MultiIndex
            df_seq = analysis_df.index.get_level_values("sequence_code")
            df_res = analysis_df.index.get_level_values("residue_name")
            df_residues = set(zip(df_seq, df_res.str.upper()))

            # 3. Calculate the intersection of the tuples
            overlap = pdb_residues.intersection(df_residues)

            pdb_nums = {res[0] for res in pdb_residues} if pdb_residues else set()
            df_nums = {res[0] for res in df_residues} if df_residues else set()

            if not overlap:
                print("❌ Error: No residue numbering AND type overlap between DataFrame and PDB!")
                print(f"   PDB Residues: {
                      min(pdb_nums) if pdb_nums else 'N/A'}-{max(pdb_nums) if pdb_nums else 'N/A'}")
                print(f"   DF Residues:  {
                      min(df_nums) if df_nums else 'N/A'}-{max(df_nums) if df_nums else 'N/A'}")
                return

            elif len(overlap) < len(df_residues):
                print(f"❌ Error: Incomplete mapping! Only {len(overlap)} of {
                      len(df_residues)} DataFrame residues matched the PDB.")
                unmapped = df_residues - overlap
                unmapped_nums = sorted([res[0] for res in unmapped])
                print(f"   Unmapped DF Residue Numbers: {unmapped_nums}")
            else:
                print(f"✅ Successfully mapped all {len(overlap)} DataFrame residues to the PDB.")

        except Exception as e:
            print(f"❌ Failed to parse structure for safety check: {e}")
            return

        seq_codes = analysis_df.index.get_level_values("sequence_code").fillna(-999).astype(int)

        # --- 4. THE MAPPING LOOP ---
        for category in active_metrics:
            settings = config[category]
            prefix = settings["prefix"]
            spec_list = analysis_df[category].columns

            for spec_name in spec_list:
                print(f"\n--- Mapping {prefix}: {spec_name} ---")

                # Extract data and convert to attenuation if necessary
                raw_values = analysis_df[(category, spec_name)]
                if settings["calc_attenuation"]:
                    map_values = 1.0 - raw_values
                    print(f"   (Mapping attenuation: 1 - {prefix})")
                else:
                    map_values = raw_values

                # Handle scaling
                min_val = 0.0
                if max_val is not None:
                    max_scaling = float(max_val)
                    print(f"📏 Using manual scale: 0.0 to {max_scaling}")
                else:
                    max_scaling = float(np.nanmax(map_values.values))
                    print(f"📏 Using automatic scale: 0.0 to {max_scaling:.3f}")

                ppdb = PandasPdb().read_pdb(working_pdb)
                metric_dict = dict(zip(seq_codes, map_values))

                # Map and Fill missing with 0.0
                ppdb.df["ATOM"]["b_factor"] = (
                    ppdb.df["ATOM"]["residue_number"].map(metric_dict).fillna(0.0)
                )

                safe_name = spec_name.replace(" ", "_").replace("/", "_")
                final_pdb = out_path / f"{prefix}_{safe_name}.pdb"
                ppdb.to_pdb(str(final_pdb))

                # --- 5. PyMOL Logic ---
                cmd.reinitialize()
                cmd.feedback("disable", "all", "actions")

                obj_name = f"{prefix}_{safe_name}"
                cmd.load(str(final_pdb), obj_name)
                cmd.show_as("cartoon", obj_name)
                cmd.copy("Cartoon_Obj", obj_name)

                cmd.spectrum(
                    "b",
                    "blue_white_orange_red",
                    selection="all",
                    minimum=min_val,
                    maximum=max_scaling,
                )

                cmd.ramp_new(
                    f"{prefix}_Scale",
                    "none",
                    range=[min_val, max_scaling],
                    color=["blue", "white", "orange", "red"],
                )

                cmd.cartoon("putty", obj_name)

                cmd.set("ray_shadow", "off")
                cmd.set("spec_reflect", 0)

                pse_filename = out_path / f"{prefix}_{safe_name}.pse"
                cmd.save(str(pse_filename))
                print(f"   ✅ Saved session: {pse_filename.name}")

    except Exception as e:
        print(f"🔥 Critical Failure during mapping: {e}")

    finally:
        # --- 6. THE FINAL CLEANUP ---
        working_path = Path(working_pdb)
        if is_cif and working_path.exists():
            working_path.unlink()
            print("🧹 Temporary conversion file removed.")

    print("-" * 30 + "\nMapping Complete.")
