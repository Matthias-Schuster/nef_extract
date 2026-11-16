import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .utils import get_one_letter, natural_sort_order

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
        # Output directory stored in df.attrs by the analysis
        save_path = Path(analysis_df.attrs.get("output_dir", "results")) / "plots"

    save_path.mkdir(parents=True, exist_ok=True)

    analysis_name = analysis_df.attrs.get("analysis_name", "")
    if analysis_name:
        save_path = save_path / analysis_name
        save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path / fig_name, bbox_inches="tight", dpi=300)


def _bar_axes(ax, plot_x, x_axis_name, ylabel, title, ylabel_size=14):
    """Common look of all per-residue bar plots (CSP, intensities, R1/R2, hetNOE)."""
    n = len(plot_x)
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(plot_x, rotation=90, fontfamily="monospace", fontsize=9)
    if x_axis_name:
        ax.set_xlabel(x_axis_name, size=14)
    ax.set_ylabel(ylabel, size=ylabel_size)
    if title:
        ax.set_title(title, size=15)


def _ratio_noise(analysis_df, category, spec_name):
    """Errors and detection limit of a ratio column (only present if noise was given)."""
    err = analysis_df.get((f"{category}_Err", spec_name))
    lod = analysis_df.get((f"{category}_LOD", spec_name))
    err = None if err is None else err.to_numpy(dtype=float)
    lod = None if lod is None else lod.to_numpy(dtype=float)
    return err, lod


BELOW_LOD_COLOR = "0.7"  # grey bars: ratio not distinguishable from 0


def _below_lod(values, lod):
    """True where a ratio is negative or (if known) below its detection limit."""
    values = np.asarray(values, dtype=float)
    below = values < 0
    if lod is not None:
        below |= values < lod
    return below


def _below_lod_legend(ax, below, loc="upper right", extra_handles=()):
    """Legend entry for the grey bars (only if there are any)."""
    from matplotlib.patches import Patch

    handles = list(extra_handles)
    if below.any():
        handles.append(Patch(facecolor=BELOW_LOD_COLOR, edgecolor="black", linewidth=0.35,
                             label="below detection limit (≈ 0)"))
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=9, framealpha=0.9)


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

    If the analysis was run with ``noise=...``, the intensity plots show error bars and
    bars below the detection limit (or negative) are grey: 0 within the noise.
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

            err = lod = None
            if category != "CSPs" and not save_prefix.startswith("Orig_"):
                err, lod = _ratio_noise(analysis_df, category, spec_name)

            y_arr = y_vals.to_numpy(dtype=float)
            colors = np.full(len(y_arr), settings["color"], dtype=object)
            below = np.zeros(len(y_arr), dtype=bool)
            if category != "CSPs" and not save_prefix.startswith("Orig_"):
                below = _below_lod(y_arr, lod)
                colors[below] = BELOW_LOD_COLOR

            fig, ax = plt.subplots(figsize=(len(y_vals) * 0.2, 5))
            ax.bar(
                np.arange(len(y_vals)),
                y_arr,
                width=0.8,
                edgecolor="black",
                linewidth=0.35,
                color=list(colors),
                yerr=err,
                error_kw={"elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6, "ecolor": "black"},
            )
            _below_lod_legend(ax, below)
            _bar_axes(ax, plot_x, x_axis_name, current_ylabel, current_title)

            if current_ylim is not None:
                # raw ratios can be slightly negative (noise of fully broadened peaks)
                bottom = 0
                if len(y_arr) and np.nanmin(y_arr) < 0:
                    bottom = np.nanmin(y_arr - (err if err is not None else 0)) * 1.1
                ax.set_ylim(bottom, current_ylim)
                if bottom < 0:
                    ax.axhline(0, color="black", linewidth=0.6)

            if category == "CSPs":
                ax.axhline(y=cutoff, color="red", linestyle=":", linewidth=1)

            _save_plot(f"{save_prefix}_{spec_name}.png", analysis_df, output_dir)
            plt.show()
            plt.close(fig)


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

            err, lod = _ratio_noise(analysis_df, metric_col, spec_name)
            below = _below_lod(analysis_df[(metric_col, spec_name)].to_numpy(dtype=float), lod)
            int_colors = np.where(below, BELOW_LOD_COLOR, metric_settings["color"]).tolist()
            ax2 = ax1.twinx()
            ax2.set_ylabel(metric_settings["ylabel_attenuation"], color="black", size=14)
            ax2.bar(
                r1,
                df_plot["normalized"],
                width,
                color=int_colors,
                edgecolor="black",
                linewidth=0.35,
                label=metric_settings["label"],
                yerr=err,
                error_kw={"elinewidth": 0.5, "capsize": 1, "capthick": 0.5, "ecolor": "black"},
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

            # explicit handles: the first intensity bar may be grey (below detection limit)
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=c, edgecolor="black", linewidth=0.35, label=lab)
                for c, lab in [
                    (csp_config["color"], csp_config["label"]),
                    (metric_settings["color"], metric_settings["label"]),
                ]
            ]
            _below_lod_legend(ax1, below, extra_handles=handles)

            _save_plot(
                f"Combined_{metric_settings['prefix']}_{spec_name}.png", analysis_df, output_dir
            )
            plt.show()
            plt.close()


# =============================================================================
# DYNAMICS Plot Functions (T1, T2, hetNOE)
# =============================================================================


def _dynamics_category(result_df, show="rate"):
    """Returns the column category to plot: rate (R2), time (T2) or hetNOE."""
    metric = result_df.attrs.get("dynamics_metric")
    levels = set(result_df.columns.get_level_values(0))
    if metric is None:
        candidates = [c for c in levels if c not in ("Metadata", "Intensity", "Fit")]
        if not candidates:
            raise ValueError("No dynamics data (R1/R2/hetNOE) found in this DataFrame.")
        metric = sorted(candidates)[0]
    if show == "time" and metric != "hetNOE":
        return result_df.attrs.get("dynamics_time", "T" + metric[1:])
    return metric


def _dynamics_label(category):
    """Axis label with units, e.g. 'R2' -> '$R_{2}$ (s$^{-1}$)'."""
    if category == "hetNOE":
        return r"{$^1$H}-$^{15}$N hetNOE"
    sub = category[1:].replace("rho", r"\rho")
    unit = r"(s$^{-1}$)" if category.startswith("R") else "(ms)"
    return rf"${category[0]}_{{{sub}}}$ {unit}"


def _dynamics_values(result_df, category, only_ok=True):
    """Values and errors of one category, failed fits set to NaN."""
    values = result_df[(category, "value")].astype(float)
    errors = result_df[(category, "error")].astype(float)
    if only_ok and ("Fit", "fit_ok") in result_df.columns:
        ok = result_df[("Fit", "fit_ok")].fillna(False).astype(bool)
        values, errors = values.where(ok), errors.where(ok)
    return values, errors


def _dynamics_bars(ax, values, errors, color):
    ind = np.arange(len(values))
    ax.bar(
        ind,
        values,
        width=0.8,
        yerr=errors,
        color=color,
        edgecolor="black",
        linewidth=0.35,
        error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "black"},
    )
    return ind


DYNAMICS_COLORS = {"R": "tab:blue", "T": "tab:green", "hetNOE": "tab:blue"}


def plot_relaxation_rates(
    result_df,
    show="both",
    ylim=None,
    color=None,
    output_dir=None,
    only_ok=True,
):
    """
    Bar plots of R1/R2 and/or T1/T2 (or hetNOE) per residue with error bars.
    Each value gets its own figure (e.g. R2.png and T2.png).

    Args:
        result_df (pd.DataFrame):
            Output of ``fit_relaxation`` or ``calc_hetnoe``.
        show (str, optional):
            'rate' (R in s^-1), 'time' (T in ms) or 'both' (default). Ignored for hetNOE.
        ylim (float, tuple or dict, optional):
            Upper y-limit or (lower, upper). With show='both' use a dict per value,
            e.g. ``{"R2": 12, "T2": (0, 400)}``.
        color (str or dict, optional):
            Bar color, or a dict per value, e.g. ``{"R2": "navy", "T2": "olive"}``.
            Defaults to tab:blue for R1/R2 (and hetNOE) and tab:green for T1/T2.
        output_dir (str or Path, optional):
            Base directory for the plots. Defaults to the analysis folder.
        only_ok (bool, optional):
            Hide peaks whose fit failed. Defaults to True.
    """
    if show not in ("rate", "time", "both"):
        raise ValueError("show must be 'rate', 'time' or 'both'.")

    rate = _dynamics_category(result_df, "rate")
    if rate == "hetNOE":
        categories = [rate]
    elif show == "both":
        categories = [rate, _dynamics_category(result_df, "time")]
    else:
        categories = [_dynamics_category(result_df, show)]

    # One ylim does not fit R (s^-1) and T (ms) at the same time
    if ylim is not None and not isinstance(ylim, dict) and len(categories) > 1:
        print(f"ℹ️ ylim ignored for show='both' – use e.g. "
              f"ylim={{'{categories[0]}': 12, '{categories[1]}': 300}}.")
        ylim = None

    plot_x, x_axis_name = _get_x_axis_data(result_df)
    analysis_name = result_df.attrs.get("analysis_name", "")

    for category in categories:
        default = DYNAMICS_COLORS.get(category, DYNAMICS_COLORS.get(category[0], "tab:blue"))
        bar_color = color.get(category, default) if isinstance(color, dict) else color or default
        lim = ylim.get(category) if isinstance(ylim, dict) else ylim

        values, errors = _dynamics_values(result_df, category, only_ok)
        fig, ax = plt.subplots(figsize=(max(len(values) * 0.2, 6), 5))
        _dynamics_bars(ax, values, errors, bar_color)
        _bar_axes(ax, plot_x, x_axis_name, _dynamics_label(category),
                  f"{category}: {analysis_name}".rstrip(": "))
        if category == "hetNOE":
            ax.axhline(0, color="black", linewidth=0.6)
        if lim is not None:
            ax.set_ylim(lim if isinstance(lim, (tuple, list)) else (0, lim))

        _save_plot(f"{category}.png", result_df, output_dir)
        plt.show()
        plt.close(fig)


def plot_relaxation_curves(
    result_df,
    per_page=16,
    ncols=4,
    color="tab:blue",
    point_color="black",
    marker="o",
    output_dir=None,
    max_rel_error=0.2,
):
    """
    Plots the measured decay of every peak together with its fitted curve.
    Intensities are normalized to the fitted I0. Several pages are saved if needed.
    Fits with a relative error (R_err / R) above ``max_rel_error`` are marked in orange.

    Args:
        result_df (pd.DataFrame):
            Output of ``fit_relaxation`` (directly, not re-loaded from Excel).
        per_page (int, optional):
            Number of peaks per figure. Defaults to 16 (4 x 4).
        ncols (int, optional):
            Number of subplot columns. Defaults to 4.
        color (str, optional):
            Color of the fitted curve. Defaults to tab:blue.
        point_color (str, optional):
            Color of the measured points. Defaults to black.
        marker (str, optional):
            Marker of the measured points: 'o' (dots, default) or 'x' (crosses).
        output_dir (str or Path, optional):
            Base directory for the plots. Defaults to the analysis folder.
        max_rel_error (float, optional):
            Threshold for marking uncertain fits. Defaults to 0.2 (20 %).
    """
    delays = result_df.attrs.get("delays_s")
    if not delays:
        print("❌ No delay information found – use the DataFrame returned by fit_relaxation.")
        return

    rate = result_df.attrs.get("dynamics_metric", "R")
    model = result_df.attrs.get("model", "exp2")
    intensities = result_df["Intensity"]
    t_ms = np.array([delays[c] for c in intensities.columns]) * 1000
    t_fine = np.linspace(0, t_ms.max() * 1.05, 200)

    plot_x, _ = _get_x_axis_data(result_df)
    rows = np.where(intensities.notna().any(axis=1).to_numpy())[0]
    if len(rows) == 0:
        print("❌ No intensities to plot.")
        return

    pages = [rows[i : i + per_page] for i in range(0, len(rows), per_page)]
    for page_no, page in enumerate(pages, start=1):
        nrows = int(np.ceil(len(page) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 2.6, nrows * 2.2), sharex=True, sharey=True,
            squeeze=False,
        )
        for ax in axes.flat[len(page):]:
            ax.set_visible(False)

        for ax, row in zip(axes.flat, page):
            y = intensities.iloc[row].to_numpy(dtype=float)
            i0 = result_df[("Fit", "I0")].iloc[row]
            r = result_df[(rate, "value")].iloc[row]
            r_err = result_df[(rate, "error")].iloc[row]
            offset = result_df[("Fit", "offset")].iloc[row]
            ok = bool(result_df[("Fit", "fit_ok")].fillna(False).iloc[row])

            norm = i0 if ok and np.isfinite(i0) and i0 > 0 else np.nanmax(np.abs(y))
            ax.plot(t_ms, y / norm, linestyle="none", marker=marker, color=point_color,
                    markersize=4, markeredgewidth=1.2 if marker == "x" else 0.5, zorder=3)

            if ok:
                off = offset if model == "exp3" and np.isfinite(offset) else 0.0
                fit = (i0 * np.exp(-r * t_fine / 1000) + off) / norm
                uncertain = np.isfinite(r_err / r) and r_err / r > max_rel_error
                fit_color = "tab:orange" if uncertain else color
                ax.plot(t_fine, fit, "-", color=fit_color, linewidth=1)
                ax.text(
                    0.95, 0.9, f"{r:.2f} ± {r_err:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10,
                    color="tab:orange" if uncertain else "black",
                )
                if uncertain:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("tab:orange")
                        spine.set_linewidth(1.5)
            else:
                ax.text(0.95, 0.9, "fit failed", transform=ax.transAxes,
                        ha="right", va="top", fontsize=10, color="red")

            ax.set_title(plot_x[row], fontsize=12, fontfamily="monospace")
            ax.tick_params(labelsize=8)
            ax.set_ylim(-0.05, 1.15)

        # x labels on the lowest used panel of each column (last page may be incomplete)
        for col in range(ncols):
            used = [r for r in range(nrows) if r * ncols + col < len(page)]
            if used:
                ax = axes[used[-1], col]
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xlabel("Delay (ms)", fontsize=9)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$I/I_0$", fontsize=9)

        fig.suptitle(
            f"{rate} decays (s$^{{-1}}$) – page {page_no}/{len(pages)}", size=12
        )
        fig.tight_layout()
        _save_plot(f"{rate}_curves_{page_no}.png", result_df, output_dir)
        plt.show()
        plt.close(fig)


def plot_dynamics_overview(
    results,
    show="rate",
    colors=None,
    output_dir=None,
    filename="dynamics_overview",
    only_ok=True,
):
    """
    Stacks several dynamics results (e.g. R1, R2 and hetNOE) in one figure with a
    common residue axis.

    Args:
        results (list):
            DataFrames from ``fit_relaxation`` / ``calc_hetnoe``.
        show (str, optional):
            'rate' (default) or 'time' for the relaxation results.
        colors (list, optional):
            One color per result.
        output_dir (str or Path, optional):
            Base directory. Defaults to the folder of the first result.
        filename (str, optional):
            Name of the PNG file (saved in <output_dir>/plots/).
        only_ok (bool, optional):
            Hide peaks whose fit failed. Defaults to True.
    """
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    panels = {}
    for res in results:
        category = _dynamics_category(res, show)
        values, errors = _dynamics_values(res, category, only_ok)
        panels[category] = pd.DataFrame({"value": values, "error": errors})

    combined = pd.concat(panels, axis=1)
    order = natural_sort_order(combined.index.to_frame(index=False))
    combined = combined.iloc[order]

    meta = pd.DataFrame(index=combined.index)
    meta[("Metadata", "res_single")] = [
        get_one_letter(r) for r in combined.index.get_level_values("residue_name")
    ]
    meta.columns = pd.MultiIndex.from_tuples(meta.columns)
    plot_x, x_axis_name = _get_x_axis_data(meta)

    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(max(len(combined) * 0.2, 6), 2.8 * n), sharex=True, squeeze=False
    )
    for i, (category, ax) in enumerate(zip(panels, axes[:, 0])):
        _dynamics_bars(
            ax, combined[(category, "value")], combined[(category, "error")],
            colors[i % len(colors)],
        )
        is_last = i == n - 1
        _bar_axes(ax, plot_x, x_axis_name if is_last else None,
                  _dynamics_label(category), None, ylabel_size=12)
        if category == "hetNOE":
            ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()

    base = Path(output_dir) if output_dir else Path(results[0].attrs.get("output_dir", "results"))
    save_dir = base / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{filename}.png", bbox_inches="tight", dpi=300)
    print(f"✅ Saved overview to {save_dir / f'{filename}.png'}")
    plt.show()
    plt.close(fig)
