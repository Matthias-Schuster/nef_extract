import numpy as np
import pandas as pd
from pathlib import Path

from .utils import is_unassigned

# =============================================================================
# CSP to PDB Functions
# =============================================================================


def csp_to_pdb(
    analysis_df,
    pdb=None,
    output_dir=None,
    max_val=None,
    CSP=True,
    Int=False,
    Vol=False,
    Dyn=True,
    min_val=None,
):
    """
    Maps NMR metrics (CSPs, Intensity, Volume, or R1/R2/hetNOE) onto a structure,
    generates PyMOL sessions, and cleans up all temporary conversion files.

    Args:
        analysis_df (pd.DataFrame):
            The master analysis DataFrame containing the metrics to map.
        pdb (str or Path):
            The structure file to map onto. If a simple filename is provided
            (e.g., "input.pdb"), it automatically searches the project's 'input' folder.
        output_dir (str or Path, optional):
            Directory to save the structures. If None, automatically reads from DataFrame.
        max_val (float, optional):
            Manual maximum value for the color scale (B-factor column).
        CSP, Int, Vol (bool, optional):
            Toggles for which metrics to map onto the structure. Intensities and volumes
            are mapped as attenuation 1 - I/I0 (negative ratios count as I/I0 = 0).
        Dyn (bool, optional):
            Maps R1/R2/hetNOE if the DataFrame comes from ``fit_relaxation`` or
            ``calc_hetnoe``. Residues without data (or with a failed fit) are grey and
            get the thinnest putty tube. Defaults to True.
        min_val (float, optional):
            Manual minimum of the color scale. Defaults to 0 for CSPs/intensities and
            to the smallest value for dynamics data.
    """

    from Bio.PDB import MMCIFParser, PDBIO
    from biopandas.pdb import PandasPdb
    from pymol import cmd

    # --- 1. Auto-Routing the Input PDB ---
    if pdb is None:
        print("❌ Error: No structure file provided.")
        return

    pdb_path = Path(pdb)

    # If you pass a simple filename (like "input.pdb"), route it to the project's 'input' folder
    if not pdb_path.is_absolute() and not pdb_path.exists():
        input_dir = analysis_df.attrs.get("input_dir")
        if input_dir is None:
            # results/<project>/ -> <root>/input
            base_dir = Path(analysis_df.attrs.get("output_dir", Path("results").resolve()))
            input_dir = base_dir.parent.parent / "input"
        pdb_path = Path(input_dir) / pdb

    if not pdb_path.exists():
        print(f"❌ Error: File not found: {pdb_path}")
        return

    # Check file extension
    suffix = pdb_path.suffix.lower()
    if suffix not in [".pdb", ".cif"]:
        print(f"❌ Error: {suffix} is not a valid format. Use .pdb or .cif")
        return

    # --- 2. Auto-Routing the Output Directory ---
    if output_dir is not None:
        base_path = Path(output_dir) / "structures"
    else:
        # Output directory stored in df.attrs by the analysis
        base_dir = analysis_df.attrs.get("output_dir", Path("results"))
        base_path = Path(base_dir) / "structures"

    analysis_name = analysis_df.attrs.get("analysis_name", "")
    if analysis_name:
        out_path = base_path / analysis_name
    else:
        out_path = base_path

    out_path.mkdir(parents=True, exist_ok=True)

    # --- 3. Temp File Setup ---
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

    # Dynamics results (fit_relaxation / calc_hetnoe): map only the 'value' column
    dyn_metric = analysis_df.attrs.get("dynamics_metric")
    if dyn_metric:
        config[dyn_metric] = {
            "active": Dyn,
            "prefix": dyn_metric,
            "calc_attenuation": False,
            "columns": ["value"],
            "dynamics": True,
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

        # Numeric residue numbers. Unassigned peaks (e.g. '@12' or peak numbers with
        # residue name 'unassigned') become NaN so they are never mapped onto the structure.
        seq_codes = pd.Series(
            pd.to_numeric(analysis_df.index.get_level_values("sequence_code"), errors="coerce"),
            index=analysis_df.index,
        )
        unassigned_rows = [
            is_unassigned(r) for r in analysis_df.index.get_level_values("residue_name")
        ]
        seq_codes[unassigned_rows] = np.nan

        # --- FAILSAFE 3: Residue Mapping Overlap ---
        try:
            temp_check = PandasPdb().read_pdb(working_pdb)

            # 1. Extract PDB (Number, Name) pairs
            pdb_df = temp_check.df["ATOM"]
            pdb_residues = set(zip(pdb_df["residue_number"], pdb_df["residue_name"].str.upper()))

            # 2. Extract DataFrame (Number, Name) pairs, only for numbered (assigned) residues
            df_residues = {
                (int(n), str(r).upper())
                for n, r in zip(seq_codes, analysis_df.index.get_level_values("residue_name"))
                if pd.notna(n)
            }
            n_skipped = int(seq_codes.isna().sum())
            if n_skipped:
                print(f"ℹ️ {n_skipped} rows without a numeric residue number are not mapped.")

            # 3. Calculate the intersection of the tuples
            overlap = pdb_residues.intersection(df_residues)

            pdb_nums = {res[0] for res in pdb_residues}
            df_nums = {res[0] for res in df_residues}

            if not overlap:
                pdb_range = f"{min(pdb_nums)}-{max(pdb_nums)}" if pdb_nums else "N/A"
                df_range = f"{min(df_nums)}-{max(df_nums)}" if df_nums else "N/A"
                print("❌ Error: No residue numbering AND type overlap between DataFrame and PDB!")
                print(f"   PDB Residues: {pdb_range}")
                print(f"   DF Residues:  {df_range}")
                return

            elif len(overlap) < len(df_residues):
                print(
                    f"❌ Error: Incomplete mapping! Only {len(overlap)} of "
                    f"{len(df_residues)} DataFrame residues matched the PDB."
                )
                unmapped_nums = sorted(res[0] for res in df_residues - overlap)
                print(f"   Unmapped DF Residue Numbers: {unmapped_nums}")
            else:
                print(f"✅ Successfully mapped all {len(overlap)} DataFrame residues to the PDB.")

        except Exception as e:
            print(f"❌ Failed to parse structure for safety check: {e}")
            return

        # --- 4. THE MAPPING LOOP ---
        for category in active_metrics:
            settings = config[category]
            prefix = settings["prefix"]
            is_dyn = settings.get("dynamics", False)
            spec_list = settings.get("columns", analysis_df[category].columns)

            for spec_name in spec_list:
                label = (analysis_name or prefix) if is_dyn else spec_name
                print(f"\n--- Mapping {prefix}: {label} ---")

                # Extract data and convert to attenuation if necessary
                raw_values = analysis_df[(category, spec_name)].astype(float)
                if settings["calc_attenuation"]:
                    # negative ratios are noise of fully broadened peaks -> I/I0 = 0
                    map_values = 1.0 - raw_values.clip(lower=0)
                    print(f"   (Mapping attenuation: 1 - {prefix})")
                else:
                    map_values = raw_values

                # Failed relaxation fits are not mapped
                if is_dyn and ("Fit", "fit_ok") in analysis_df.columns:
                    ok = analysis_df[("Fit", "fit_ok")].fillna(False).astype(bool)
                    map_values = map_values.where(ok)

                # Handle scaling
                if min_val is not None:
                    lo = float(min_val)
                elif is_dyn:
                    lo = float(np.nanmin(map_values.values))
                else:
                    lo = 0.0
                if max_val is not None:
                    max_scaling = float(max_val)
                else:
                    max_scaling = float(np.nanmax(map_values.values))
                mode = "manual" if max_val is not None or min_val is not None else "automatic"
                print(f"📏 Using {mode} scale: {lo:.3f} to {max_scaling:.3f}")

                ppdb = PandasPdb().read_pdb(working_pdb)
                valid = seq_codes.notna().to_numpy()
                if is_dyn:
                    valid = valid & map_values.notna().to_numpy()
                metric_dict = dict(
                    zip(seq_codes[valid].astype(int), np.asarray(map_values)[valid])
                )

                # Map and fill missing residues. For CSPs/intensities with 0.0; for dynamics
                # with the scale minimum (thin putty), they are coloured grey further below.
                # A far-off placeholder would distort PyMOL's putty scaling (mean/stdev of b).
                missing_b = lo if is_dyn else 0.0
                ppdb.df["ATOM"]["b_factor"] = (
                    ppdb.df["ATOM"]["residue_number"].map(metric_dict).fillna(missing_b)
                )

                safe_name = str(label).replace(" ", "_").replace("/", "_")
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
                    minimum=lo,
                    maximum=max_scaling,
                )

                # Residues without dynamics data (or with a failed fit) are shown in grey
                if is_dyn:
                    data_resi = sorted(int(n) for n in metric_dict)
                    if data_resi:
                        resi = "+".join(str(n).replace("-", "\\-") for n in data_resi)
                        cmd.color("grey70", f"not resi {resi}")
                    else:
                        cmd.color("grey70", "all")

                cmd.ramp_new(
                    f"{prefix}_Scale",
                    "none",
                    range=[lo, max_scaling],
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
