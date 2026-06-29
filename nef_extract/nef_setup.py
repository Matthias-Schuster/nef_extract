from pathlib import Path
import sys


def setup_nef_project(
    script_file,
    nef_filename=None,
    prefix="nef_extract_",
    results_folder="results",
    input_folder="input",
    create_output_dirs=True,
    verbose=True,
):
    # Location of the currently running extraction script
    script_path = Path(script_file).resolve()
    script_dir = script_path.parent

    # Project root is one level above nef_extract/
    root_dir = script_dir.parent

    # Make the root directory available to Python's import system
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Extract project name from filename
    script_name = script_path.stem
    project_name = script_name[len(prefix):] if script_name.startswith(prefix) else script_name

    # Define core directory paths
    project_out = root_dir / results_folder / project_name
    input_dir = root_dir / input_folder

    paths = {
        "script_dir": script_dir,
        "root_dir": root_dir,
        "project_out": project_out,
        "input_dir": input_dir,
    }

    # Create output folders dynamically
    if create_output_dirs:
        paths["project_out"].mkdir(parents=True, exist_ok=True)
        paths["input_dir"].mkdir(parents=True, exist_ok=True)

    # File checking logic
    if nef_filename:
        input_nef = input_dir / nef_filename
        if not input_nef.exists():
            raise FileNotFoundError(
                f"Could not find '{nef_filename}' in the input directory:\n-> {input_nef}"
            )
        paths["input_nef"] = input_nef

    if verbose:
        print("\n----- NEF Extraction Setup -----")
        print(f"Project name: {project_name}")
        if nef_filename:
            print(f"Input file:   {paths.get('input_nef')}")
        else:
            print(f"Data Input:   {paths['input_dir']}")
        print(f"Output dir:   {paths['project_out']}")
        print("--------------------------------\n")

    return project_name, paths
