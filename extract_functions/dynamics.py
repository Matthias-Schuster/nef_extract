from pathlib import Path

import numpy as np
import pandas as pd

from .csp import align_to_full_sequence
from .utils import get_one_letter, index_peaklist, natural_sort_order

# =============================================================================
# Relaxation (T1, T2, T1rho, ...) and heteronuclear NOE analysis
# =============================================================================
#
# Result layout (same MultiIndex style as the CSP analyses, so the existing
# plotting helpers and csp_to_pdb can be reused):
#
#   ("Metadata",  "res_single")
#   ("Intensity", <spectrum>)          raw peak heights / volumes
#   (<rate>,      "value" / "error")   e.g. ("R2", "value") in s^-1
#   (<time>,      "value" / "error")   e.g. ("T2", "value") in ms
#   ("Fit",       "I0", "I0_error", "offset", "r_squared", "chi2_red", "error_scale"
#                 (√χ² factor applied to the errors), "rel_error" (R_err / R), "n_points", "fit_ok",
#                 "sign_flipped")   Intensity holds the (sign-corrected) values used in the fit
#
# For the hetNOE the metric is ("hetNOE", "value" / "error").

_DELAY_FACTORS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_spectra(peaks_dict, spectra):
    """Converts spectrum indices (from the spectrum menu) to names."""
    names = list(peaks_dict.keys())
    if spectra is None:
        return names
    if isinstance(spectra, (int, np.integer, str)):
        spectra = [spectra]

    resolved = []
    for spec in spectra:
        name = names[spec] if isinstance(spec, (int, np.integer)) else spec
        if name not in peaks_dict:
            raise KeyError(f"Spectrum '{name}' not found in the peak lists.")
        resolved.append(name)
    return resolved


def _collect_intensities(peaks_dict, spectra, metric="height"):
    """
    Combines the selected peak lists into one table:
    rows = residues (chain_code, sequence_code, residue_name), columns = spectra.
    Also returns the per-peak uncertainties ('<metric>_uncertainty') if CCPN exported them.
    """
    dims = {tuple(peaks_dict[s].attrs.get("dimensions", [])) for s in spectra}
    if len(dims) > 1:
        print(f"⚠️ Warning: The selected spectra have different dimensions: {sorted(dims)}")

    unc_col = f"{metric}_uncertainty"
    values, uncertainties = {}, {}
    for spec in spectra:
        indexed = index_peaklist(peaks_dict[spec], data_cols=[metric, unc_col], name=spec)
        if metric not in indexed.columns:
            raise KeyError(f"Spectrum '{spec}' has no '{metric}' column.")
        values[spec] = indexed[metric]
        if unc_col in indexed.columns:
            uncertainties[spec] = indexed[unc_col]

    table = pd.concat(values, axis=1)
    table = table.iloc[natural_sort_order(table.index.to_frame(index=False))]
    table = table.apply(pd.to_numeric, errors="coerce")

    unc = None
    if uncertainties:
        unc = pd.concat(uncertainties, axis=1).reindex(index=table.index, columns=table.columns)
        unc = unc.apply(pd.to_numeric, errors="coerce")
        if unc.isna().all().all():
            unc = None

    n_complete = int(table.notna().all(axis=1).sum())
    print(f"📊 Collected {metric}s of {len(table)} peaks in {len(spectra)} spectra "
          f"({n_complete} peaks present in all spectra).")
    return table, unc


def _finalize(result, project_data, filename, align, seq_index, extra_sheets=None):
    """Alignment, metadata column, Excel export and attrs for plotting/PDB mapping."""
    project_data = project_data or {}
    sequences = project_data.get("sequences") or {}
    out_dir = Path(project_data.get("out_dir") or "results")

    if align:
        if not sequences:
            print("⚠️ Warning: 'align' is True, but no sequence was found.")
        elif len(sequences) > seq_index:
            target_key = list(sequences.keys())[seq_index]
            print(f"🧬 Using sequence for alignment: {target_key}")
            attrs = dict(result.attrs)
            result = align_to_full_sequence(result, sequences[target_key])
            result.attrs.update(attrs)
        else:
            print(f"⚠️ Warning: seq_index {seq_index} out of range. Skipping alignment.")

    res_names = result.index.get_level_values("residue_name")
    result.insert(0, ("Metadata", "res_single"), [get_one_letter(r) for r in res_names])

    safe_name = str(filename).replace("/", "_").replace("\\", "_")
    if safe_name.lower().endswith(".xlsx"):
        safe_name = safe_name[:-5]
    save_path = out_dir / f"{safe_name}.xlsx"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Results")
        for sheet, df in (extra_sheets or {}).items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"✅ Saved {result.attrs.get('dynamics_metric')} analysis to ({save_path})\n")

    result.attrs["analysis_name"] = safe_name
    result.attrs["output_dir"] = save_path.parent
    if project_data.get("input_dir"):
        result.attrs["input_dir"] = Path(project_data["input_dir"])
    return result


# -----------------------------------------------------------------------------
# Relaxation fits (T1, T2, T1rho, ...)
# -----------------------------------------------------------------------------


def _model_function(model):
    if model == "exp2":
        return lambda t, i0, r: i0 * np.exp(-r * t)
    if model == "exp3":
        return lambda t, i0, r, c: i0 * np.exp(-r * t) + c
    raise ValueError("model must be 'exp2' (I0·exp(-R·t)) or 'exp3' (I0·exp(-R·t) + C).")


def _duplicate_noise(table, delays_s):
    """
    Estimates the spectral noise SD (absolute intensity units) from duplicate delays:
    pooled standard deviation of all repeated measurements over all peaks.
    Returns (sigma, number of duplicate delay groups) or (None, 0).
    """
    rounded = np.round(delays_s, 9)
    groups = [np.where(rounded == d)[0] for d in np.unique(rounded)]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        return None, 0

    ss, dof = 0.0, 0
    for g in groups:
        vals = table.iloc[:, g].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals).all(axis=1)]
        if len(vals) == 0:
            continue
        ss += ((vals - vals.mean(axis=1, keepdims=True)) ** 2).sum()
        dof += vals.shape[0] * (vals.shape[1] - 1)

    if dof == 0:
        return None, 0
    return float(np.sqrt(ss / dof)), len(groups)


def _fit_single(t, y, sigma, model, func):
    """Fits one decay curve. Returns a dict with parameters and errors."""
    from scipy.optimize import curve_fit

    n_par = 2 if model == "exp2" else 3
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    sig = sigma[ok] if sigma is not None else None
    out = {"n_points": int(ok.sum()), "fit_ok": False}

    if len(t) < n_par + 1:
        return out

    # --- Initial guesses: log-linear fit on positive points ---
    pos = y > 0
    r0 = 1.0 / max(np.ptp(t), 1e-9)
    if pos.sum() >= 2 and np.ptp(t[pos]) > 0:
        slope = np.polyfit(t[pos], np.log(y[pos]), 1)[0]
        if np.isfinite(slope) and slope < 0:
            r0 = -slope
    i0 = float(np.max(np.abs(y[t == t.min()]))) * np.exp(r0 * t.min())

    if model == "exp2":
        p0, bounds = [i0, r0], ([0, 0], [np.inf, np.inf])
    else:
        p0, bounds = [i0, r0, 0.0], ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(
            func, t, y, p0=p0, sigma=sig, absolute_sigma=sig is not None,
            bounds=bounds, maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return out

    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
    residuals = y - func(t, *popt)
    ss_tot = np.sum((y - y.mean()) ** 2)
    # Reduced chi-square with the given point uncertainty (only defined if sigma is known)
    dof = len(t) - n_par
    chi2_red = np.sum((residuals / sig) ** 2) / dof if sig is not None and dof > 0 else np.nan

    out.update(
        {
            "I0": popt[0],
            "I0_error": perr[0],
            "R": popt[1],
            "R_error": perr[1],
            "offset": popt[2] if model == "exp3" else 0.0,
            "r_squared": 1 - np.sum(residuals**2) / ss_tot if ss_tot > 0 else np.nan,
            "chi2_red": chi2_red,
            "fit_ok": bool(np.isfinite(perr[1]) and popt[1] > 0),
        }
    )
    return out


def fit_relaxation(
    peaks_dict,
    delays,
    spectra=None,
    experiment="T2",
    project_data=None,
    delay_unit="ms",
    model="exp2",
    metric="height",
    error_method="auto",
    noise=None,
    use_nef_noise=False,
    scale_errors=True,
    align=False,
    seq_index=0,
    filename=None,
):
    """
    Fits relaxation decays (T2, T1, T1rho, ...) for every peak.

    Model 'exp2':  I(t) = I0 · exp(-R · t)
    Model 'exp3':  I(t) = I0 · exp(-R · t) + C   (for curves that do not decay to zero)

    Peaks are matched between spectra via their assignment (or the CCPN '@' NmrResidue
    name / peak id if unassigned), so the peak lists do not need identical peak ids.

    Args:
        peaks_dict (dict):
            Peak lists from ``extract_all_nef_data`` (optionally renamed).
        delays (list):
            Relaxation delays, one per spectrum in ``spectra`` (same order).
            Repeated delays are allowed and used for the error estimation.
        spectra (list, optional):
            Spectrum names or indices from the spectrum menu. Defaults to all spectra.
        experiment (str, optional):
            'T2', 'T1', 'T1rho', ... Defines the column names (R2/T2, R1/T1, ...).
        project_data (dict, optional):
            Same dictionary as for the CSP analysis: 'out_dir', 'input_dir', 'sequences'.
        delay_unit (str, optional):
            Unit of ``delays``: 'ms' (default), 's' or 'us'.
        model (str, optional):
            'exp2' (default) or 'exp3'.
        metric (str, optional):
            'height' (default) or 'volume'.
        error_method (str, optional):
            How the uncertainties of R and T are calculated:
            - 'auto' (default): 'duplicates' if repeated delays exist, otherwise 'fit'.
            - 'duplicates': the spectral noise is estimated from the repeated delays
              (pooled over all peaks) and used as the point uncertainty of the fit.
            - 'fit': uncertainties from the fit residuals (covariance matrix).
        noise (float, optional):
            Known spectral noise as standard deviation (SD, same units as the peak
            heights), used as uncertainty of every point. Overrides ``error_method``.
            CCPN: 'Estimate Noise...' (shortcut EN) -> 'SD', not the 'Noise Level'
            (= mean + 3.5 SD). With repeated delays it is usually better to leave it out:
            the duplicates also contain drift and peak-height errors, not only the noise.
        use_nef_noise (bool, optional):
            True: use the noise SD stored by CCPN in the NEF file for every spectrum
            (spectra without one get ``noise``; if that is None too, ``error_method`` is
            used). Defaults to False, because repeated delays are usually more realistic.
        scale_errors (bool, optional):
            For 'duplicates' and 'noise': if the points of a peak scatter more around the
            fit than σ explains (reduced χ² > 1), its errors are multiplied by √χ².
            σ is a lower limit, errors are never made smaller. Defaults to True.
            ('fit' errors already scale with the scatter.)
        align (bool, optional):
            Align the result to the full sequence (needs project_data['sequences']).
        seq_index (int, optional):
            Which sequence to align to. Defaults to 0.
        filename (str, optional):
            Name of the Excel output. Defaults to '<experiment>_analysis'.

    Returns:
        pd.DataFrame: Results with rates (s^-1), times (ms), fit statistics and intensities.
    """
    if delay_unit not in _DELAY_FACTORS:
        raise ValueError(f"delay_unit must be one of {list(_DELAY_FACTORS)}.")

    spectra = _resolve_spectra(peaks_dict, spectra)
    delays = list(delays)
    if len(delays) != len(spectra):
        raise ValueError(
            f"{len(delays)} delays were given for {len(spectra)} spectra. "
            "Provide exactly one delay per spectrum (same order as 'spectra')."
        )

    experiment = str(experiment)
    rate = "R" + experiment[1:] if experiment.upper().startswith("T") else f"R_{experiment}"
    time = experiment if experiment.upper().startswith("T") else f"T_{experiment}"
    func = _model_function(model)

    print(f"\n--- {experiment} relaxation analysis ({model}) ---")
    delays_s = np.asarray(delays, dtype=float) * _DELAY_FACTORS[delay_unit]
    table, _ = _collect_intensities(peaks_dict, spectra, metric=metric)

    # --- Negative peaks (e.g. picked on noise): flip the whole decay to positive ---
    # Only peaks whose median intensity is negative are flipped; single negative
    # points of a positive decay (noise at long delays) are kept as they are.
    flipped = table.median(axis=1, skipna=True) < 0
    if flipped.any():
        table = table.mul(np.where(flipped, -1.0, 1.0), axis=0)
        labels = [str(i[1]) for i in table.index[flipped][:15]]
        print(
            f"⚠️ {int(flipped.sum())} peak(s) with negative intensities were made positive "
            f"for the fit (check them!): {', '.join(labels)}"
            f"{' ...' if flipped.sum() > 15 else ''}"
        )

    # --- Error model ---
    if error_method not in ("auto", "duplicates", "fit"):
        raise ValueError("error_method must be 'auto', 'duplicates' or 'fit'.")
    sigma_value, method_used = None, "fit"
    nef_noise = _nef_noise(peaks_dict, spectra) if use_nef_noise else {}
    if nef_noise:
        per_spec = [nef_noise.get(sp, noise) for sp in spectra]
        missing = [sp for sp, v in zip(spectra, per_spec) if v is None]
        if missing:
            names = ", ".join(missing[:4]) + (f" (+{len(missing) - 4} more)" if len(missing) > 4 else "")
            print(f"⚠️ No noise SD in the NEF for {names} and no noise= given – using "
                  f"error_method='{error_method}' instead.")
        else:
            sigma_value = np.array(per_spec, dtype=float)  # one σ per spectrum
            noise = sigma_value
            n_nef = sum(sp in nef_noise for sp in spectra)
            print(f"📏 Noise SD: {np.min(sigma_value):.3g} – {np.max(sigma_value):.3g} "
                  f"({n_nef} from the NEF, {len(spectra) - n_nef} given)")
    elif use_nef_noise:
        print("ℹ️ use_nef_noise=True, but no noise SD is stored in the NEF for these spectra.")
    if noise is not None and sigma_value is None:
        sigma_value = float(noise)
        print(f"📏 Using the given noise SD σ = {sigma_value:.4g}")
    if noise is not None:
        method_used = "noise"
        dup_sigma, _ = _duplicate_noise(table, delays_s)
        given = float(np.median(sigma_value))
        if dup_sigma is not None and given < 0.75 * dup_sigma:
            print(
                f"ℹ️ The given noise σ = {given:.3g} is smaller than the scatter of the "
                f"duplicate delays (σ = {dup_sigma:.3g}) – the errors may be too small. "
                "Leave out noise= to use the duplicates."
            )
    elif error_method in ("auto", "duplicates"):
        sigma_value, n_groups = _duplicate_noise(table, delays_s)
        if sigma_value is not None:
            method_used = "duplicates"
            rel = sigma_value / np.nanmedian(table.to_numpy(dtype=float))
            print(
                f"📏 Noise estimated from {n_groups} duplicate delay(s): σ = {sigma_value:.4g} "
                f"({rel:.1%} of the median intensity)"
            )
        elif error_method == "duplicates":
            print("⚠️ No duplicate delays found – falling back to errors from the fit.")
    if method_used == "fit":
        print("📏 Uncertainties from the fit residuals (covariance matrix).")

    # --- Fit every peak ---
    fits = []
    for _, row in table.iterrows():
        y = row.to_numpy(dtype=float)
        sigma = None if sigma_value is None else np.broadcast_to(sigma_value, y.shape).astype(float)
        fits.append(_fit_single(delays_s, y, sigma, model, func))
    fits = pd.DataFrame(fits, index=table.index)
    for col in ["I0", "I0_error", "R", "R_error", "offset", "r_squared", "chi2_red"]:
        if col not in fits.columns:
            fits[col] = np.nan
    fits["sign_flipped"] = flipped

    # --- Scale the errors up where the scatter around the fit is larger than σ ---
    fits["error_scale"] = 1.0
    scaled = scale_errors and method_used in ("duplicates", "noise")
    if scaled:
        factor = np.sqrt(fits["chi2_red"].clip(lower=1.0)).fillna(1.0)
        fits["error_scale"] = factor
        fits["R_error"] *= factor
        fits["I0_error"] *= factor
        up = fits["fit_ok"].astype(bool) & (factor > 1.0)
        if up.any():
            print(
                f"📏 Errors of {int(up.sum())} / {int(fits['fit_ok'].sum())} peaks scaled up by "
                f"√χ² (scatter around the fit > σ), median factor {factor[up].median():.2f}"
            )
    fits["rel_error"] = fits["R_error"] / fits["R"]

    r, r_err = fits["R"], fits["R_error"]
    result = pd.concat(
        {
            "Intensity": table,
            rate: pd.DataFrame({"value": r, "error": r_err}),
            time: pd.DataFrame({"value": 1000 / r, "error": 1000 * r_err / r**2}),
            "Fit": fits[
                [
                    "I0", "I0_error", "offset", "r_squared", "chi2_red", "error_scale",
                    "rel_error", "n_points", "fit_ok", "sign_flipped",
                ]
            ],
        },
        axis=1,
    )

    n_ok = int(fits["fit_ok"].sum())
    print(f"✅ Fitted {n_ok} / {len(fits)} peaks. "
          f"Median {rate} = {r[fits['fit_ok']].median():.2f} s⁻¹, "
          f"{time} = {1000 / r[fits['fit_ok']].median():.1f} ms")
    uncertain = fits["fit_ok"].astype(bool) & (fits["rel_error"] > 0.2)
    if uncertain.any():
        labels = [str(i[1]) for i in fits.index[uncertain][:15]]
        print(f"ℹ️ {int(uncertain.sum())} fit(s) with a relative error > 20 %: {', '.join(labels)}"
              f"{' ...' if uncertain.sum() > 15 else ''}")
    failed = fits.index[~fits["fit_ok"].astype(bool)]
    if len(failed):
        labels = [str(i[1]) for i in failed[:15]]
        print(f"⚠️ Fit failed / too few points for: {', '.join(labels)}"
              f"{' ...' if len(failed) > 15 else ''}")

    result.attrs.update(
        {
            "dynamics_metric": rate,
            "dynamics_time": time,
            "experiment": experiment,
            "model": model,
            "delays_s": dict(zip(spectra, delays_s)),
        }
    )

    delay_sheet = pd.DataFrame(
        {"spectrum": spectra, f"delay ({delay_unit})": delays, "delay (s)": delays_s}
    )
    settings_sheet = pd.DataFrame(
        {
            "setting": [
                "experiment", "model", "metric", "error_method", "noise_sigma", "scale_errors",
            ],
            "value": [
                experiment, model, metric, method_used,
                None if sigma_value is None else float(np.median(sigma_value)), bool(scaled),
            ],
        }
    )
    return _finalize(
        result,
        project_data,
        filename or f"{experiment}_analysis",
        align,
        seq_index,
        extra_sheets={"Delays": delay_sheet, "Settings": settings_sheet},
    )


# -----------------------------------------------------------------------------
# Heteronuclear NOE
# -----------------------------------------------------------------------------


def _nef_noise(peaks_dict, spectra):
    """CCPN noise SD stored in the NEF file ({spectrum: SD}), only spectra that have one."""
    out = {}
    for spec in spectra:
        sd = peaks_dict[spec].attrs.get("noise_sd")
        if sd is not None:
            out[spec] = float(sd)
    return out


def _noise_for(noise, spec, position, nef_noise=None):
    """
    Noise SD of one spectrum: the SD stored by CCPN in the NEF file if there is one,
    otherwise the given value (number, list/tuple entry or dict entry, may be None).
    Returns (value, source) with source 'NEF', 'given' or None.
    """
    if nef_noise and spec in nef_noise:
        return float(nef_noise[spec]), "NEF"
    if isinstance(noise, dict):
        value = noise.get(spec)
    elif isinstance(noise, (list, tuple)):
        value = noise[position] if position < len(noise) else None
    else:
        value = noise
    return (None, None) if value is None else (float(value), "given")


def calc_hetnoe(
    peaks_dict,
    sat,
    ref,
    project_data=None,
    noise=None,
    use_nef_noise=True,
    metric="height",
    align=False,
    seq_index=0,
    filename="hetNOE_analysis",
):
    """
    Calculates the heteronuclear NOE (I_sat / I_ref) for every peak.

    Args:
        peaks_dict (dict):
            Peak lists from ``extract_all_nef_data`` (optionally renamed).
        sat (int, str or list):
            Saturated spectrum (name or index). A list of several saturated spectra
            (repeats) is allowed; then ``ref`` must be a list of the same length.
        ref (int, str or list):
            Reference (unsaturated) spectrum, or a list matching ``sat``.
        project_data (dict, optional):
            Same dictionary as for the CSP analysis: 'out_dir', 'input_dir', 'sequences'.
        noise (float, tuple or dict, optional):
            Spectral noise as standard deviation (SD, same units as the peak heights) for
            the error propagation. Per spectrum the SD stored by CCPN in the NEF file is
            used first (shown in the spectrum menu); the value given here is only used for
            spectra without a stored SD. A single value for all spectra, a tuple
            (noise_sat, noise_ref) that may contain None, or a dict {spectrum_name: noise}.
            CCPN: 'Estimate Noise...' (EN) -> 'SD', not the 'Noise Level' (= mean + 3.5 SD).
            Without any noise SD, the peak uncertainties from the NEF file are used if
            available.
        use_nef_noise (bool, optional):
            Use the noise SD stored in the NEF file first. False: only ``noise``.
            Defaults to True.
        metric (str, optional):
            'height' (default) or 'volume'.
        align (bool, optional):
            Align the result to the full sequence (needs project_data['sequences']).
        seq_index (int, optional):
            Which sequence to align to. Defaults to 0.
        filename (str, optional):
            Name of the Excel output. Defaults to 'hetNOE_analysis'.

    Error calculation:
        σ_NOE = sqrt(σ_sat² + NOE² · σ_ref²) / |I_ref|
        (= |NOE| · sqrt((σ_sat / I_sat)² + (σ_ref / I_ref)²), but also valid for I_sat ≈ 0)
        With repeated pairs the NOE is the mean of all pairs. Its error is the
        propagated error of the mean, or the standard error of the pairs if no
        noise is known (the larger of both if both are available).

    Returns:
        pd.DataFrame: Results with ('hetNOE', 'value'/'error') and the intensities.
    """
    sat = _resolve_spectra(peaks_dict, sat)
    ref = _resolve_spectra(peaks_dict, ref)
    if len(sat) != len(ref):
        raise ValueError("'sat' and 'ref' must contain the same number of spectra.")

    print("\n--- Heteronuclear NOE analysis ---")
    spectra = list(dict.fromkeys(sat + ref))
    table, nef_unc = _collect_intensities(peaks_dict, spectra, metric=metric)

    # Noise SD per spectrum: stored by CCPN in the NEF first, otherwise the given value
    nef_noise = _nef_noise(peaks_dict, spectra) if use_nef_noise else {}

    ratios, ratio_errs = [], []
    for s_name, r_name in zip(sat, ref):
        i_sat, i_ref = table[s_name], table[r_name].replace(0, np.nan)
        noe = i_sat / i_ref

        n_sat, src_sat = _noise_for(noise, s_name, 0, nef_noise)
        n_ref, src_ref = _noise_for(noise, r_name, 1, nef_noise)
        if n_sat is not None and n_ref is not None:
            print(f"📏 Noise SD: sat {s_name} {n_sat:.3g} ({src_sat}), "
                  f"ref {r_name} {n_ref:.3g} ({src_ref})")
        elif n_sat is not None or n_ref is not None:
            missing = [n for n, v in ((s_name, n_sat), (r_name, n_ref)) if v is None]
            print(f"⚠️ No noise SD for {', '.join(map(repr, missing))} (not in the NEF and not "
                  "given). Pass noise=... or in CCPN: Estimate Noise -> 'Set Noise Level'.")
        if n_sat is None and nef_unc is not None:
            n_sat = nef_unc[s_name]
        if n_ref is None and nef_unc is not None:
            n_ref = nef_unc[r_name]

        if n_sat is not None and n_ref is not None:
            # same as |NOE|·sqrt((n_sat/I_sat)² + (n_ref/I_ref)²), but no division by I_sat
            err = np.sqrt(n_sat**2 + (noe * n_ref) ** 2) / i_ref.abs()
        else:
            err = pd.Series(np.nan, index=table.index)
        ratios.append(noe)
        ratio_errs.append(err)

    ratios = pd.concat(ratios, axis=1)
    ratio_errs = pd.concat(ratio_errs, axis=1)
    n = ratios.notna().sum(axis=1)

    value = ratios.mean(axis=1)
    propagated = np.sqrt((ratio_errs**2).sum(axis=1, min_count=1)) / n
    if len(sat) > 1:
        sem = ratios.std(axis=1, ddof=1) / np.sqrt(n)
        error = pd.concat([propagated, sem], axis=1).max(axis=1)
    else:
        error = propagated

    if error.isna().all():
        print("⚠️ No noise SD (given or in the NEF file) and no peak uncertainties – hetNOE "
              "errors are empty. Pass noise=... (CCPN: Estimate Noise -> 'SD') or press "
              "'Set Noise Level' in CCPN before the NEF export.")

    result = pd.concat(
        {
            "Intensity": table,
            "hetNOE": pd.DataFrame({"value": value, "error": error}),
        },
        axis=1,
    )
    print(f"✅ hetNOE for {int(value.notna().sum())} peaks, median = {value.median():.3f}")

    result.attrs.update({"dynamics_metric": "hetNOE", "experiment": "hetNOE"})
    pairs_sheet = pd.DataFrame({"saturated": sat, "reference": ref})
    return _finalize(
        result, project_data, filename, align, seq_index, extra_sheets={"Pairs": pairs_sheet}
    )
