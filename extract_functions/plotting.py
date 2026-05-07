import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# PLOT Functions
# =============================================================================


def get_nmr_plot_config(
    ylim_csp=None,
    ylim_ratio=1.1,
    CSP=True,
    Int=True,
    Vol=True,
    color_csp="tab:blue",
    color_int="tab:orange",
    color_vol="tab:red",
    int_prefix="Norm_",
):
    """
    Centralized configuration for NMR plot styling, labels, and toggles.
    Dynamically adapts to 'Norm_' or 'Ratio_' prefixes.
    """
    title_word = "Normalized" if int_prefix == "Norm_" else "Raw Ratio"

    return {
        "CSPs": {
            "active": CSP,
            "color": color_csp,
            "ylabel": r"Weighted $\Delta\delta$ (ppm)",
            "title": "CSPs",
            "prefix": "CSP",
            "label": "CSP",
            "ylim": ylim_csp,
        },
        f"{int_prefix}Height": {
            "active": Int,
            "color": color_int,
            "ylabel": "Intensity Ratio ($I/I_0$)",
            "ylabel_attenuation": r"Attenuation ($1 - I/I_0$)",
            "title": f"{title_word} Intensities",
            "prefix": "Int",
            "label": "Intensity",
            "ylim": ylim_ratio,
        },
        f"{int_prefix}Volume": {
            "active": Vol,
            "color": color_vol,
            "ylabel": "Volume Ratio ($V/V_0$)",
            "ylabel_attenuation": r"Attenuation ($1 - V/V_0$)",
            "title": f"{title_word} Volumes",
            "prefix": "Vol",
            "label": "Volume",
            "ylim": ylim_ratio,
        },
    }


def plot_nmr_metrics(
    analysis_df,
    cutoff=0.01,
    ylim_csp=None,
    ylim_ratio=1.1,
    save_path="results/plots",
    CSP=True,
    Int=True,
    Vol=True,
    color_csp="tab:blue",
    color_int="tab:orange",
    color_vol="tab:red",
):
    """
    Generates and saves individual bar plots for selected NMR metrics.

    Automatically resolves sequence assignments or unassigned peak numbers for the x-axis.
    It reads from a MultiIndex DataFrame and exports plots to a structured directory.

    Args:
        analysis_df (pd.DataFrame):
            The master analysis DataFrame containing calculated metrics and metadata
            (expected to have MultiIndex columns).
        cutoff (float, optional):
            Y-value for a horizontal threshold line drawn on CSP plots to indicate significance.
            Defaults to 0.01.
        ylim_csp (float, optional):
            Maximum y-axis limit for CSP plots. Defaults to None.
        ylim_ratio (float, optional):
            Maximum y-axis limit for ratio plots. Defaults to 1.1.
        save_path (str or Path, optional):
            Base directory where plots will be saved. A subfolder is created if `analysis_name`
            is in the DataFrame attributes. Defaults to "results/plots".
        CSP (bool, optional):
            Whether to generate CSP plots. Defaults to True.
        Int (bool, optional):
            Whether to generate Intensity ratio plots. Defaults to True.
        Vol (bool, optional):
            Whether to generate Volume ratio plots. Defaults to True.
        color_csp (str, optional):
            Matplotlib color for CSP plots. Defaults to 'tab:blue'.
        color_int (str, optional):
            Matplotlib color for Intensity plots. Defaults to 'tab:orange'.
        color_vol (str, optional):
            Matplotlib color for Volume plots. Defaults to 'tab:red'.

    Returns:
        None: Displays the plots interactively and saves them to disk.
    """

    levels = analysis_df.columns.levels[0]
    int_prefix = "Norm_" if "Norm_Height" in levels or "Norm_Volume" in levels else "Ratio_"

    config = get_nmr_plot_config(
        ylim_csp, ylim_ratio, CSP, Int, Vol, color_csp, color_int, color_vol, int_prefix
    )

    for category, settings in config.items():
        if not settings["active"] or category not in levels:
            continue

        spec_list = analysis_df[category].columns

        for spec_name in spec_list:
            df_plot = pd.DataFrame(
                {
                    "Val": analysis_df[(category, spec_name)],
                    "Res_Code": analysis_df.index.get_level_values("sequence_code"),
                    "Res_Letter": analysis_df[("Metadata", "res_single")],
                }
            )

            res_code_str = (
                df_plot["Res_Code"].fillna(0).astype(str).str.replace(r"\.0$", "", regex=True)
            )

            is_unassigned = df_plot["Res_Letter"].astype(str).str.upper() == "UNASSIGNED"

            # Combine number and residue if assigned, otherwise just use the number
            df_plot["Plot_X"] = np.where(
                is_unassigned, res_code_str, res_code_str + " " + df_plot["Res_Letter"].astype(str)
            )

            # Dynamically set the axis label
            x_axis_name = "Peak Number" if is_unassigned.all() else "Residue"

            df_plot.plot.bar(
                x="Plot_X",
                y="Val",
                figsize=(len(df_plot) * 0.2, 5),
                width=0.8,
                edgecolor="black",
                linewidth=0.35,
                color=settings["color"],
                legend=False,
            )

            plt.xticks(fontfamily="monospace", rotation=90, fontsize=9)
            plt.xlabel(x_axis_name)
            plt.title(f"{settings['title']}: {spec_name}")
            plt.ylabel(settings["ylabel"])
            plt.ylim(0, settings["ylim"])

            if category == "CSPs":
                plt.axhline(y=cutoff, color="red", linestyle=":", linewidth=1)

            if save_path:
                current_save_path = Path(save_path)
                analysis_name = analysis_df.attrs.get("analysis_name", "")
                if analysis_name:
                    current_save_path = current_save_path / analysis_name

                current_save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    current_save_path / f"{settings['prefix']}_{spec_name}.png",
                    bbox_inches="tight",
                    dpi=300,
                )

            plt.show()
            plt.close()


def plot_combined(
    analysis_df,
    ylim_csp=None,
    ylim_ratio=1.05,
    save_path="results/plots",
    Int=True,
    Vol=False,
    color_csp="tab:blue",
    color_int="tab:orange",
    color_vol="tab:red",
):
    """
    Creates a dual-axis bar plot visualizing CSPs alongside signal attenuation.

    The primary y-axis (right) displays CSPs, while the secondary y-axis (left)
    displays attenuation calculated as (1 - I/I0) or (1 - V/V0).

    Args:
        analysis_df (pd.DataFrame):
            The master analysis DataFrame. Must contain a 'CSPs' column level to proceed.
        ylim_csp (float, optional):
            Maximum y-axis limit for the CSPs. Defaults to None.
        ylim_ratio (float, optional):
            Maximum y-axis limit for the attenuation axis. Defaults to 1.05.
        save_path (str or Path, optional):
            Base directory for saving output plots. Defaults to "results/plots".
        Int (bool, optional):
            If True, plots Intensity attenuation alongside CSPs. Defaults to True.
        Vol (bool, optional):
            If True, plots Volume attenuation alongside CSPs. Defaults to False.
        color_csp (str, optional):
            Matplotlib color for CSP bars. Defaults to 'tab:blue'.
        color_int (str, optional):
            Matplotlib color for Intensity attenuation bars. Defaults to 'tab:orange'.
        color_vol (str, optional):
            Matplotlib color for Volume attenuation bars. Defaults to 'tab:red'.

    Returns:
        None: Displays the dual-axis plots interactively and saves them to disk.
        Prints a warning if required CSP or secondary metric data is missing.
    """
    levels = analysis_df.columns.levels[0]
    if "CSPs" not in levels:
        print("Required data (CSPs) missing for combined plot.")
        return

    save_path = Path(save_path)
    analysis_name = analysis_df.attrs.get("analysis_name", "")
    if analysis_name:
        save_path = save_path / analysis_name
    save_path.mkdir(parents=True, exist_ok=True)

    int_prefix = "Norm_" if "Norm_Height" in levels or "Norm_Volume" in levels else "Ratio_"

    config = get_nmr_plot_config(
        ylim_csp,
        ylim_ratio,
        CSP=True,
        Int=Int,
        Vol=Vol,
        color_csp=color_csp,
        color_int=color_int,
        color_vol=color_vol,
        int_prefix=int_prefix,
    )
    csp_config = config["CSPs"]

    secondary_metrics = {
        k: v for k, v in config.items() if k != "CSPs" and v["active"] and k in levels
    }

    if not secondary_metrics:
        print("No secondary metrics (Height or Volume) found or selected.")
        return

    spec_list = analysis_df["CSPs"].columns

    for metric_col, metric_settings in secondary_metrics.items():
        for spec_name in spec_list:
            df_plot = pd.DataFrame(
                {
                    "shifts": analysis_df[("CSPs", spec_name)],
                    "normalized": 1 - analysis_df[(metric_col, spec_name)],
                    "Res_Code": analysis_df.index.get_level_values("sequence_code"),
                    "Res_Letter": analysis_df[("Metadata", "res_single")],
                }
            )

            res_code_str = (
                df_plot["Res_Code"].fillna(0).astype(str).str.replace(r"\.0$", "", regex=True)
            )

            is_unassigned = df_plot["Res_Letter"].astype(str).str.upper() == "UNASSIGNED"

            df_plot["Plot_X"] = np.where(
                is_unassigned, res_code_str, res_code_str + " " + df_plot["Res_Letter"].astype(str)
            )

            x_axis_name = "Peak Number" if is_unassigned.all() else "Residue"

            ind = np.arange(len(df_plot))
            width = 0.4
            r1 = ind - width / 2
            r2 = ind + width / 2

            fig, ax1 = plt.subplots(figsize=(len(df_plot) * 0.2, 5))

            ax1.set_ylabel(csp_config["ylabel"], color="black")
            ax1.bar(
                r2,
                df_plot["shifts"],
                width,
                color=csp_config["color"],
                edgecolor="black",
                linewidth=0.35,
                label=csp_config["label"],
            )
            ax1.tick_params(axis="y", labelcolor="black")
            if csp_config["ylim"]:
                ax1.set_ylim(0, csp_config["ylim"])

            ax2 = ax1.twinx()
            ax2.set_ylabel(metric_settings["ylabel_attenuation"], color="black")
            ax2.bar(
                r1,
                df_plot["normalized"],
                width,
                color=metric_settings["color"],
                edgecolor="black",
                linewidth=0.35,
                label=metric_settings["label"],
            )
            ax2.tick_params(axis="y", labelcolor="black")
            ax2.set_ylim(0, metric_settings["ylim"])

            ax1.set_xlabel(x_axis_name)
            ax1.set_title(f"NMR Perturbation Analysis ({metric_settings['label']}): {spec_name}")

            ax1.set_xticks(ind)
            ax1.set_xticklabels(
                df_plot["Plot_X"], rotation=90, ha="center", fontfamily="monospace", fontsize=9
            )
            ax1.set_xlim(r1[0] - width / 2, r2[-1] + width / 2)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper right")

            plt.savefig(
                save_path / f"Combined_{metric_settings['prefix']}_{spec_name}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.show()
            plt.close()
