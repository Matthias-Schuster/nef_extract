import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# HELPER Functions
# =============================================================================


def _get_x_axis_data(analysis_df):
    """Helper to generate consistent x-axis labels across all plots."""
    res_code = analysis_df.index.get_level_values("sequence_code")
    res_letter = analysis_df[("Metadata", "res_single")]

    res_code_str = res_code.fillna(0).astype(str).str.replace(r"\.0$", "", regex=True)
    is_unassigned = res_letter.astype(str).str.upper() == "UNASSIGNED"

    # Combine number and residue if assigned, otherwise just use the number
    plot_x = np.where(is_unassigned, res_code_str, res_code_str + " " + res_letter.astype(str))

    # Dynamically set the axis label
    axis_name = "Peak Number" if is_unassigned.all() else "Residue"

    return plot_x, axis_name


def _save_plot(fig_name, analysis_df, output_dir=None):
    """Helper to handle directory routing, creation, and saving."""
    if output_dir is not None:
        save_path = Path(output_dir) / "plots"
    else:
        # Read the directory secretly stored in the DataFrame
        save_path = Path(analysis_df.attrs.get("output_dir", "results")) / "plots"

    save_path.mkdir(parents=True, exist_ok=True)

    analysis_name = analysis_df.attrs.get("analysis_name", "")
    if analysis_name:
        save_path = save_path / analysis_name
        save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path / fig_name, bbox_inches="tight", dpi=300)


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
    output_dir=None,
    CSP=True,
    Int=True,
    Vol=True,
    color_csp="tab:blue",
    color_int="tab:orange",
    color_vol="tab:red",
    show_original=False,
):
    """
    Generates and saves individual bar plots for selected NMR metrics.

    Automatically resolves sequence assignments or unassigned peak numbers for the x-axis.
    It reads from a MultiIndex DataFrame and exports plots to a structured directory.

    Args:
        analysis_df (pd.DataFrame):
            The master analysis DataFrame containing calculated metrics and metadata.
        cutoff (float, optional):
            Y-value for a horizontal threshold line drawn on CSP plots. Defaults to 0.01.
        ylim_csp (float, optional):
            Maximum y-axis limit for CSP plots. Defaults to None.
        ylim_ratio (float, optional):
            Maximum y-axis limit for ratio plots. Defaults to 1.1.
        output_dir (str or Path, optional):
            Base directory where plots will be saved.
        CSP, Int, Vol (bool, optional):
            Toggles for which plots to generate. Default to True.
        color_csp, color_int, color_vol (str, optional):
            Matplotlib colors for the respective plots.
        show_original (bool, optional):
            If True, bypasses the normalized/ratio values and plots the absolute, raw
            heights or volumes. The y-axis will auto-scale. Defaults to False.
    """

    levels = analysis_df.columns.levels[0]
    int_prefix = "Norm_" if "Norm_Height" in levels or "Norm_Volume" in levels else "Ratio_"

    config = get_nmr_plot_config(
        ylim_csp, ylim_ratio, CSP, Int, Vol, color_csp, color_int, color_vol, int_prefix
    )

    plot_x, x_axis_name = _get_x_axis_data(analysis_df)

    for category, settings in config.items():
        if not settings["active"] or category not in levels:
            continue

        spec_list = analysis_df[category].columns

        for spec_name in spec_list:

            y_vals = analysis_df[(category, spec_name)]
            current_ylabel = settings["ylabel"]
            current_title = f"{settings['title']}: {spec_name}"
            current_ylim = settings["ylim"]
            save_prefix = settings["prefix"]

            # --- show_original Logic ---
            if show_original and ("Height" in category or "Volume" in category):
                raw_metric = "height" if "Height" in category else "volume"

                if (spec_name, raw_metric) in analysis_df.columns:
                    y_vals = analysis_df[(spec_name, raw_metric)]
                    current_ylabel = f"Absolute {settings['label']}"
                    current_title = f"Original {settings['label']}s: {spec_name}"
                    current_ylim = None
                    save_prefix = f"Orig_{settings['prefix']}"
                else:
                    print(
                        f"⚠️ Raw '{raw_metric}' missing for {spec_name}. "
                        "Using calculated ratios."
                    )

            df_plot = pd.DataFrame({"Val": y_vals, "Plot_X": plot_x})

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
            plt.xlabel(x_axis_name, size=14)
            plt.title(current_title, size=15)
            plt.ylabel(current_ylabel, size=14)

            if current_ylim is not None:
                plt.ylim(0, current_ylim)

            if category == "CSPs":
                plt.axhline(y=cutoff, color="red", linestyle=":", linewidth=1)

            _save_plot(f"{save_prefix}_{spec_name}.png", analysis_df, output_dir)
            plt.show()
            plt.close()


def plot_combined(
    analysis_df,
    ylim_csp=None,
    ylim_ratio=1.05,
    output_dir=None,
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
        output_dir (str or Path, optional):
            Base directory where plots will be saved.
        Int, Vol (bool, optional):
            Toggles for plotting Intensity or Volume attenuation alongside CSPs.
        color_csp, color_int, color_vol (str, optional):
            Matplotlib colors for the respective bars.
    """
    levels = analysis_df.columns.levels[0]
    if "CSPs" not in levels:
        print("Required data (CSPs) missing for combined plot.")
        return

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

    plot_x, x_axis_name = _get_x_axis_data(analysis_df)
    spec_list = analysis_df["CSPs"].columns

    for metric_col, metric_settings in secondary_metrics.items():
        for spec_name in spec_list:
            df_plot = pd.DataFrame(
                {
                    "shifts": analysis_df[("CSPs", spec_name)],
                    "normalized": 1 - analysis_df[(metric_col, spec_name)],
                    "Plot_X": plot_x,
                }
            )

            ind = np.arange(len(df_plot))
            width = 0.4
            r1 = ind - width / 2
            r2 = ind + width / 2

            fig, ax1 = plt.subplots(figsize=(len(df_plot) * 0.2, 5))

            ax1.set_ylabel(csp_config["ylabel"], color="black", size=14)
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
            ax2.set_ylabel(metric_settings["ylabel_attenuation"], color="black", size=14)
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

            ax1.set_xlabel(x_axis_name, size=14)
            ax1.set_title(
                f"NMR Perturbation Analysis ({metric_settings['label']}): {spec_name}", size=15
            )

            ax1.set_xticks(ind)
            ax1.set_xticklabels(
                df_plot["Plot_X"], rotation=90, ha="center", fontfamily="monospace", fontsize=9
            )
            ax1.set_xlim(r1[0] - width / 2, r2[-1] + width / 2)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper right")

            _save_plot(
                f"Combined_{metric_settings['prefix']}_{spec_name}.png", analysis_df, output_dir
            )
            plt.show()
            plt.close()
