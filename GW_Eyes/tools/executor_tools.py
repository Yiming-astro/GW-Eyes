from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from pathlib import Path
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP

from typing import Literal, Any, Dict, Optional, List

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from GW_Eyes.tools.executor_utils import (
    _load_index,
    _sanitize_filename,
    _compute_credible_level,
    _render_skymap_with_multiple_markers,
    _render_multiple_skymaps_with_em_marker,
    _create_transparent_cmap,
    _compute_distance_statistics,
    _render_distance_distribution,
)
from GW_Eyes.src.config import get_em_csv_paths, get_gw_index_file, get_output_path

# Default index path (adjust if you keep it elsewhere)
DEFAULT_INDEX_PATH = get_gw_index_file()
DEFAULT_CACHE_PATH = get_output_path()
DEFAULT_SNE_PATHS = get_em_csv_paths()

mcp = FastMCP("executor_tools")


@mcp.tool()
def redshift_to_luminosity_distance(
    redshift: Any,
    cosmo_model: str = "Planck18",
) -> Dict[str, Any]:
    """
    Convert redshift to luminosity distance using astropy cosmology.

    This tool supports both single redshift values and lists/arrays of redshifts.
    The output luminosity distance is in units of Mpc.

    Parameters
    ----------
    redshift : float or list[float]
        Redshift value(s). Can be a single float or a list of floats.
    cosmo_model : str, default "Planck18"
        Cosmology model to use. Currently supports "Planck18" (Planck 2018 results).

    Returns
    -------
    Dict[str, Any]
        {
            "status": "ok" | "error",
            "input_redshift": float | list[float],
            "luminosity_distance": float | list[float],
            "unit": "Mpc",
            "cosmology_model": str,
            "message": str
        }

    Examples
    --------
    # Single redshift
    result = redshift_to_luminosity_distance(redshift=0.01)
    # result["luminosity_distance"] -> 44.68 (float)

    # List of redshifts
    result = redshift_to_luminosity_distance(redshift=[0.01, 0.05, 0.1])
    # result["luminosity_distance"] -> [44.68, 230.5, 490.2] (list)
    """
    from astropy.cosmology import Planck18

    # Map cosmology model name to object
    cosmo_map = {
        "Planck18": Planck18,
    }

    if cosmo_model not in cosmo_map:
        return {
            "status": "error",
            "input_redshift": redshift,
            "luminosity_distance": None,
            "unit": "Mpc",
            "cosmology_model": cosmo_model,
            "message": f"Unsupported cosmology model: {cosmo_model}. Supported: {list(cosmo_map.keys())}",
        }

    selected_cosmo = cosmo_map[cosmo_model]

    try:
        # Convert input to numpy array for uniform processing
        z_input = np.atleast_1d(redshift)
        z_array = np.array(z_input, dtype=float)

        # Check for negative redshifts
        if np.any(z_array < 0):
            return {
                "status": "error",
                "input_redshift": redshift,
                "luminosity_distance": None,
                "unit": "Mpc",
                "cosmology_model": cosmo_model,
                "message": "Redshift cannot be negative",
            }

        # Compute luminosity distance
        dL = selected_cosmo.luminosity_distance(z_array)
        dL_mpc = dL.to(u.Mpc).value

        # Return single value if input was single, otherwise return list
        is_single = not isinstance(redshift, (list, tuple, np.ndarray))

        if is_single:
            dL_result = float(dL_mpc[0])
        else:
            dL_result = dL_mpc.tolist()

        return {
            "status": "ok",
            "input_redshift": redshift,
            "luminosity_distance": dL_result,
            "unit": "Mpc",
            "cosmology_model": cosmo_model,
            "message": f"Successfully converted redshift to luminosity distance for {len(z_array)} value(s)",
        }

    except Exception as e:
        return {
            "status": "error",
            "input_redshift": redshift,
            "luminosity_distance": None,
            "unit": "Mpc",
            "cosmology_model": cosmo_model,
            "message": f"Failed to convert redshift to luminosity distance: {e}",
        }

@mcp.tool()
def query_skymaps(
    event: str,
    match: Literal["auto", "short", "full"] = "auto",
    return_mode: Literal["summary", "paths", "both"] = "both",
) -> dict[str, Any]:
    """
    Look up gravitational-wave skymap FITS files for a given event using the local index.jsonl.

    IMPORTANT WORKFLOW RULES (READ CAREFULLY)
    -----------------------------------------
    - The local index.jsonl is the authoritative inventory for the currently installed dataset
      (e.g., the already-downloaded GWTC-4 skymap release on disk).
    - This tool MUST be used as the first step before any downstream operation
      (plotting, reading FITS, statistics).
    - If this tool returns status="not_found", treat it as a TERMINAL condition:
        * The event is not present in the local dataset and should be reported as unavailable.
        * Do NOT attempt to download GWTC releases or call other tools to "search online" or "fetch data".
        * In this project, the dataset is assumed complete; not_found almost always indicates an invalid
          event name (typo) or an event outside the installed release.

    What this tool provides
    -----------------------
    Given an event identifier, it returns:
    - whether the event exists in the local dataset
    - how many skymap FITS files exist for the event
    - which full event IDs (full_name) are present (useful if one short_name maps to multiple triggers)
    - the available waveform variants and their FITS file paths

    Parameters
    ----------
    event:
        GW event identifier to query. It can be either:
        - short name: "GW230518"
        - full name:  "GW230518_125908"
    match:
        How to interpret `event`:
        - "auto": if `event` contains "_", treat it as a full name; otherwise treat as a short name
        - "short": force match on short_name
        - "full":  force match on full_name
    return_mode:
        Choose the output depending on what you will do next:
        - "summary": Use when you only need to report availability to the user.
          Example: "Is GW230518 present? How many waveforms and which ones?"
          Returns the grouped "by_full_name" structure (may omit the flat candidates list).
        - "paths": Use when you are about to call a downstream tool that needs file paths.
          Returns the flat "candidates" list, which is easy to select from programmatically.
        - "both": Returns both representations (default).

    Returns
    -------
    dict
        On found:
        - status="ok"
        - by_full_name (if summary/both)
        - candidates (if paths/both)

        On not found (TERMINAL):
        - status="not_found"
        - exists=false, num_files=0
        - The caller should stop and report that the event is unavailable in the local dataset.
    """
    index_path = str(DEFAULT_INDEX_PATH)
    idx_path = Path(index_path)

    mode = match
    if mode == "auto":
        mode = "full" if "_" in event else "short"

    records = _load_index(idx_path)

    key = "full_name" if mode == "full" else "short_name"
    filtered = [r for r in records if r.get(key) == event]

    if not filtered:
        return {
            "status": "not_found",
            "query": {"event": event, "match": mode, "index_path": str(idx_path)},
            "exists": False,
            "num_files": 0,
            "full_names": [],
        }

    by_full_name: dict[str, list[dict[str, str]]] = {}
    candidates: list[dict[str, str]] = []

    for r in filtered:
        full_name = str(r.get("full_name"))
        item = {
            "full_name": full_name,
            "waveform": str(r.get("waveform")),
            "path": str(r.get("path")),
        }
        candidates.append(item)
        by_full_name.setdefault(full_name, []).append(
            {"waveform": item["waveform"], "path": item["path"]}
        )

    full_names = sorted(by_full_name.keys())

    result: dict[str, Any] = {
        "status": "ok",
        "query": {"event": event, "match": mode, "index_path": str(idx_path)},
        "exists": True,
        "num_files": len(candidates),
        "full_names": full_names,
    }

    if return_mode in ("summary", "both"):
        result["by_full_name"] = by_full_name

    if return_mode in ("paths", "both"):
        result["candidates"] = candidates

    return result

@mcp.tool()
def visual_skymap(
    skymap_file_paths: List[str],
    save_file: str,
    cmap_names: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    plot_style: Optional[Literal["auto", "line", "density"]] = "auto",
) -> Dict[str, Any]:
    """
    Render one or more gravitational-wave skymap FITS files to a PNG image and save it to the cache directory.

    If only asked to show the skymap of a GWxxxxxx event without any additional requirements, display any one
    of the skymaps (DO NOT SHOW ALL OF THEM). This helps keep the image simple.

    Intended usage:
    - First call `query_skymaps(event=...)` to get candidate FITS paths.
    - Select one or multiple candidate paths (e.g., different waveforms for the same event).
    - Call this tool with the chosen FITS paths and a desired output filename.

    Parameters
    ----------
    skymap_file_paths:
        List of paths to *.fits skymap files (e.g., from `query_skymaps` candidates).
        Multiple paths will be overlaid on the same plot with different colormaps.
    save_file:
        Output image filename (e.g., "GW230518_IMRPhenomXPHM_Skymap.png").
        The image will be saved under DEFAULT_CACHE_PATH.
        If it does not end with ".png", ".png" will be appended.
    cmap_names:
        Optional list of colormap names for each skymap. If not provided, defaults to
        ["Blues", "Purples", "Greens", "Oranges", "Greys", "PuBu", "PuRd", "YlOrRd", "YlGnBu"].
        The last skymap will always use "Reds" colormap for emphasis.
    labels:
        Optional list of custom labels for the legend. If provided, must have the same length as
        `skymap_file_paths`. Labels should be concise (e.g., "GWxxxxx" (the name of GW event)
        "IMRPhenomXPHM", "SEOBNRv4" (used waveforms)).
        If not provided, defaults to using FITS file stem names.
        Please use simple labels whenever possible, and avoid long labels that may affect the page layout.
    plot_style:
        Control the rendering style for multiple skymaps:
        - "auto" (default): Automatically choose based on the number of skymaps.
          If > 3 skymaps, draw 90% confidence contours; if <= 3, draw density maps.
        - "line": Always draw 90% confidence contours, regardless of the number of skymaps.
        - "density": Always draw density maps, regardless of the number of skymaps.
        
        If the feedback indicates that the Skymap has too much overlap or poor resolution, 
        please choose Line, as it is usually clearer. If there are no additional requirements, 
        just select Auto.

    Returns
    -------
    dict
        {
          "status": "ok" | "error",
          "input_fits": ["<input path 1>", "<input path 2>", ...],
          "output_png": "<saved png path>",
          "message": "<human-readable message>"
        }
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ligo.skymap.io import read_sky_map
    from astropy.visualization.wcsaxes import WCSAxes
    import ligo.skymap.plot.allsky as allsky
    from ligo.skymap.postprocess import find_greedy_credible_levels
    import astropy.units as u

    if cmap_names is None:
        cmap_names = [
            "Blues",
            "Greens",
            "Purples",
            "Oranges",
            "Greys",
            "PuBu",
            "PuRd",
            "YlOrRd",
            "YlGnBu",
        ]

    # Validate input paths
    fits_paths = []
    for path_str in skymap_file_paths:
        p = Path(path_str)
        if not p.is_file():
            return {
                "status": "error",
                "input_fits": [str(x) for x in fits_paths],
                "output_png": None,
                "message": f"Input FITS file not found: {path_str}",
            }
        fits_paths.append(p)

    if not fits_paths:
        return {
            "status": "error",
            "input_fits": [],
            "output_png": None,
            "message": "No skymap file paths provided.",
        }

    DEFAULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    out_name = save_file
    if not out_name.lower().endswith(".png"):
        out_name += ".png"
    out_name = _sanitize_filename(out_name)

    out_path = DEFAULT_CACHE_PATH / out_name

    try:
        # Read all skymaps
        probs = []
        headers = []
        plot_labels = []
        for i, fp in enumerate(fits_paths):
            prob, hdr = read_sky_map(str(fp))
            probs.append(prob)
            headers.append(hdr)
            # Use provided labels if available, otherwise fall back to file stem
            if labels is not None and i < len(labels):
                plot_labels.append(labels[i])
            else:
                plot_labels.append(fp.stem)

        n_maps = len(probs)

        # Create figure with consistent style (mollweide projection)
        fig = plt.figure(figsize=(10, 5.8))
        ax = plt.axes(projection="astro degrees mollweide")
        ax.grid()

        # Determine rendering style based on plot_style parameter
        if plot_style is None or plot_style == "auto":
            # Auto mode: use contours for > 3 skymaps, density maps for <= 3
            use_contours = n_maps > 3
        elif plot_style == "line":
            # Force contour style
            use_contours = True
        elif plot_style == "density":
            # Force density map style
            use_contours = False
        else:
            # Default to auto behavior
            use_contours = n_maps > 3

        if use_contours:
            # Draw 90% CL contours for each skymap
            legend_handles = []
            for i, (prob, hdr, label) in enumerate(zip(probs, headers, plot_labels)):
                # Compute credible levels
                credible_levels = find_greedy_credible_levels(prob)
                # Find the probability threshold corresponding to 90% credible level
                p90 = prob[credible_levels <= 0.9].min()

                # Determine if nested ordering
                nested = hdr.get("ORDERING", "RING").upper() == "NESTED"

                # Draw 90% contour with smooth parameter
                cs = ax.contour_hpx(
                    (prob, 'ICRS'),
                    levels=[p90],
                    colors=[f"C{i}"],
                    linewidths=2,
                    nested=nested,
                    smooth=(1.0 * u.deg),
                )

                legend_handles.append(
                    plt.Line2D([], [], color=f"C{i}", linewidth=2, label=label)
                )

            # Setup legend
            fig.subplots_adjust(bottom=0.22)
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(max(len(legend_handles), 1), 4),
                frameon=False,
                fontsize=9,
            )
        else:
            # Density map logic for <= 3 skymaps or when density style is forced
            # Plot each skymap with appropriate colormap
            for i, (prob, label) in enumerate(zip(probs, plot_labels)):
                if i == n_maps - 1:
                    # Last skymap uses "Reds" for emphasis
                    display_cmap = "Reds"
                    ax.imshow_hpx(
                        prob,
                        cmap=display_cmap,
                        alpha=0.8,
                        zorder=500,
                    )
                else:
                    # Other skymaps use different colormaps with transparency
                    base_cmap_name = cmap_names[i % len(cmap_names)]
                    cmap = _create_transparent_cmap(base_cmap_name, threshold=0.05)
                    ax.imshow_hpx(
                        prob,
                        cmap=cmap,
                        alpha=0.4,
                        zorder=1000,
                    )

            # Build legend
            legend_handles = []
            for i, (label) in enumerate(plot_labels):
                if i == n_maps - 1:
                    cmap_obj = plt.get_cmap("Reds")
                else:
                    cmap_obj = plt.get_cmap(cmap_names[i % len(cmap_names)])
                legend_color = cmap_obj(0.75)
                legend_handles.append(
                    plt.matplotlib.patches.Patch(
                        facecolor=legend_color,
                        edgecolor="none",
                        label=label,
                    )
                )

            fig.subplots_adjust(bottom=0.22)
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(max(len(legend_handles), 1), 4),
                frameon=False,
                fontsize=9,
            )

        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "ok",
            "input_fits": [str(p) for p in fits_paths],
            "output_png": str(out_path),
            "message": f"Skymap(s) rendered and saved ({n_maps} file(s)).",
        }

    except Exception as e:
        return {
            "status": "error",
            "input_fits": [str(p) for p in fits_paths],
            "output_png": None,
            "message": f"Failed to render skymap: {e}",
        }

@mcp.tool()
def visual_distance_distribution(
    skymap_file_paths: List[str],
    save_file: str,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Render marginalized luminosity distance distributions from one or more GW skymap FITS files.

    This tool computes and plots the all-sky marginalized distance PDF for each input skymap,
    using an efficient method that selects high-probability pixels for accurate computation.

    Parameters
    ----------
    skymap_file_paths:
        List of paths to *.fits 3D skymap files (e.g., from `query_skymaps` candidates).
        Multiple paths will be plotted on the same figure with different colors.
    save_file:
        Output image filename (e.g., "GW230518_distance_distribution.png").
        The image will be saved under DEFAULT_CACHE_PATH.
        If it does not end with ".png", ".png" will be appended.
    labels:
        Optional list of custom labels for the legend. If provided, must have the same length as
        `skymap_file_paths`. Labels should be concise (e.g., "GWxxxxx", "IMRPhenomXPHM", "SEOBNRv4").
        If not provided, defaults to using FITS file stem names.

    Returns
    -------
    dict
        {
          "status": "ok" | "error",
          "input_fits": ["<input path 1>", "<input path 2>", ...],
          "output_png": "<saved png path>",
          "message": "<human-readable message>"
        }
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.distance import marginal_pdf, marginal_ppf

    # Validate input paths
    fits_paths = []
    for path_str in skymap_file_paths:
        p = Path(path_str)
        if not p.is_file():
            return {
                "status": "error",
                "input_fits": [str(x) for x in fits_paths],
                "output_png": None,
                "message": f"Input FITS file not found: {path_str}",
            }
        fits_paths.append(p)

    if not fits_paths:
        return {
            "status": "error",
            "input_fits": [],
            "output_png": None,
            "message": "No skymap file paths provided.",
        }

    DEFAULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    out_name = save_file
    if not out_name.lower().endswith(".png"):
        out_name += ".png"
    out_name = _sanitize_filename(out_name)

    out_path = DEFAULT_CACHE_PATH / out_name

    try:
        # Read all skymaps and compute marginalized PDFs
        all_r = []
        all_pdf = []
        plot_labels = []

        for i, fp in enumerate(fits_paths):
            (prob, distmu, distsigma, distnorm), _ = read_sky_map(str(fp), distances=True)

            # Select high-probability pixels (keep pixels until cumulative prob > 0.99)
            idx = np.argsort(prob)[::-1]
            prob_cumsum = np.cumsum(prob[idx])
            idx_keep = idx[prob_cumsum <= 0.99]

            prob_sel = prob[idx_keep]
            distmu_sel = distmu[idx_keep]
            distsigma_sel = distsigma[idx_keep]
            distnorm_sel = distnorm[idx_keep]

            # Renormalize selected probabilities
            prob_sel = prob_sel / np.sum(prob_sel)

            # Compute distance grid and PDF
            r_max = marginal_ppf(0.999, prob_sel, distmu_sel, distsigma_sel, distnorm_sel)
            r = np.linspace(0.0, r_max, 100)
            pdf = marginal_pdf(r, prob_sel, distmu_sel, distsigma_sel, distnorm_sel)

            all_r.append(r)
            all_pdf.append(pdf)

            # Use provided labels if available, otherwise fall back to file stem
            if labels is not None and i < len(labels):
                plot_labels.append(labels[i])
            else:
                plot_labels.append(fp.stem)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        # Color cycle for multiple curves
        colors = ["#4E74B4", "#EB6E68", "#2E8B57", "#D2691E", "#8B4513", "#4682B4"]

        # Plot all PDFs
        for i, (r, pdf, label) in enumerate(zip(all_r, all_pdf, plot_labels)):
            color = colors[i % len(colors)]
            ax.plot(r, pdf, lw=2, color=color, label=label)

        ax.set_xlabel("Luminosity distance [Mpc]", fontsize=12)
        ax.set_ylabel("Probability density", fontsize=12)
        ax.set_title("All-sky Marginalized Luminosity Distance PDF", fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "ok",
            "input_fits": [str(p) for p in fits_paths],
            "output_png": str(out_path),
            "message": f"Distance distribution(s) rendered and saved ({len(fits_paths)} file(s)).",
        }

    except Exception as e:
        return {
            "status": "error",
            "input_fits": [str(p) for p in fits_paths],
            "output_png": None,
            "message": f"Failed to render distance distribution: {e}",
        }

@mcp.tool()
def assess_coordinates_on_skymap(
    skymap_file_path: str,
    coordinates: List[Dict[str, Any]],
    coord_unit: Literal["rad", "deg"] = "deg",
    filename: Optional[str] = None,
    skymap_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate multiple sky coordinates on a GW skymap and render them on one image.

    Parameters
    ----------
    skymap_file_path : str
        Path to GW skymap FITS file.
    coordinates : List[Dict]
        List of coordinate dictionaries:
        [
            {"ra": float, "dec": float, "label": Optional[str]},
            ...
        ]
    coord_unit : "rad" | "deg"
        Unit of input coordinates.
    filename : str, optional
        Output image filename. If not provided, defaults to
        "{skymap_stem}_multi_overlay.png". Use this to customize the output filename.
        If the filename seems too complex, consider using a simpler one.
    skymap_label : str, optional
        Optional label for the skymap in the legend. If not provided, defaults to
        the FITS file stem name.

    Returns
    -------
    {
        "status": "ok" | "error",
        "results": [
            {
                "label": str,
                "ra": float,
                "dec": float,
                "credible_level": float,
                "in_50": bool,
                "in_90": bool
            },
            ...
        ],
        "best_candidate": {...} | None,
        "output_png": str,
        "message": str
    }
    """

    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fits_path = Path(skymap_file_path)

    if not fits_path.is_file():
        return {
            "status": "error",
            "results": [],
            "best_candidate": None,
            "output_png": None,
            "message": "Input FITS file not found.",
        }

    if not coordinates:
        return {
            "status": "error",
            "results": [],
            "best_candidate": None,
            "output_png": None,
            "message": "No coordinates provided.",
        }

    results = []

    # ---------- Compute credible levels ----------
    for idx, coord in enumerate(coordinates):

        ra = coord.get("ra")
        dec = coord.get("dec")
        label = coord.get("label", f"point_{idx}")

        if ra is None or dec is None:
            continue

        credible_level = _compute_credible_level(
            skymap_file_path=str(fits_path),
            ra=ra,
            dec=dec,
            coord_unit=coord_unit,
            nest=True,
        )

        results.append({
            "label": label,
            "ra": ra,
            "dec": dec,
            "credible_level": credible_level,
            "in_50": credible_level <= 0.5,
            "in_90": credible_level <= 0.9,
        })

    if not results:
        return {
            "status": "error",
            "results": [],
            "best_candidate": None,
            "output_png": None,
            "message": "No valid coordinates found.",
        }

    # ---------- Determine best candidate ----------
    best_candidate = min(results, key=lambda x: x["credible_level"])

    # ---------- Render skymap with multiple markers ----------
    DEFAULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{fits_path.stem}_multi_overlay.png"
    elif not filename.lower().endswith(".png"):
        filename += ".png"

    filename = _sanitize_filename(filename)
    output_png = DEFAULT_CACHE_PATH / filename

    try:
        # Use provided skymap_label or default to file stem
        if skymap_label is None:
            skymap_label = fits_path.stem

        _render_skymap_with_multiple_markers(
            skymap_file_path=str(fits_path),
            coordinates=results,
            output_png=str(output_png),
            coord_unit=coord_unit,
            skymap_label=skymap_label,
        )

    except Exception as e:
        return {
            "status": "error",
            "results": results,
            "best_candidate": best_candidate,
            "output_png": None,
            "message": f"Failed to render skymap: {e}",
        }

    return {
        "status": "ok",
        "results": results,
        "best_candidate": best_candidate,
        "output_png": str(output_png),
        "message": "Multiple coordinates assessed and plotted.",
    }

@mcp.tool()
def filter_electromagnetic_events_by_time(
    gw_short_name: str,
    time_before_days: int = 3,
    time_after_days: int = 7,
    date_column: str = "maxdate"
) -> List[Dict[str, Any]]:
    """
    Filter electromagnetic events based on the time window around a given gravitational wave short name.

    Parameters
    ----------
    gw_short_name : str
        The gravitational wave event short name (e.g., "GW230518").
    time_before_days : int, optional
        The number of days before the gravitational wave trigger time to consider, default is 3.
    time_after_days : int, optional
        The number of days after the gravitational wave trigger time to consider, default is 7.
    date_column : str, optional
        The column name to use for date filtering. Options are "maxdate" or "discoverdate".
        Default is "maxdate". If set to "maxdate" but the column doesn't exist,
        will fallback to "discoverdate". If neither column exists, raises an error.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the filtered electromagnetic events. If no events match, an empty list is returned.
    """

    year = "20" + gw_short_name[2:4]
    month = gw_short_name[4:6]
    day = gw_short_name[6:8]

    gw_trigger_time_str = f"{year}/{month}/{day}"
    gw_trigger_time = datetime.strptime(gw_trigger_time_str, "%Y/%m/%d")

    all_filtered_events = []

    for sne_file_path in DEFAULT_SNE_PATHS:
        sne_path = Path(sne_file_path)

        if not sne_path.is_file():
            return {"status": "error", "message": f"File not found: {sne_file_path}"}

        df = pd.read_csv(sne_path)

        # Determine which date column to use
        actual_date_column = date_column
        if date_column == "maxdate":
            # Default logic: prefer maxdate, fallback to discoverdate
            if "maxdate" in df.columns:
                actual_date_column = "maxdate"
            elif "discoverdate" in df.columns:
                actual_date_column = "discoverdate"
            else:
                return {"status": "error", "message": f"Neither 'maxdate' nor 'discoverdate' column found in {sne_file_path}"}
        elif date_column == "discoverdate":
            # Explicitly use discoverdate
            if "discoverdate" in df.columns:
                actual_date_column = "discoverdate"
            elif "maxdate" in df.columns:
                actual_date_column = "maxdate"
            else:
                return {"status": "error", "message": f"Neither 'discoverdate' nor 'maxdate' column found in {sne_file_path}"}
        else:
            # Use specified column directly
            if date_column not in df.columns:
                return {"status": "error", "message": f"Column '{date_column}' not found in {sne_file_path}"}
            actual_date_column = date_column

        df[actual_date_column] = pd.to_datetime(df[actual_date_column], format='%Y/%m/%d', errors='coerce')

        start_time = gw_trigger_time - timedelta(days=time_before_days)
        end_time = gw_trigger_time + timedelta(days=time_after_days)

        filtered_events = df[(df[actual_date_column] >= start_time) & (df[actual_date_column] <= end_time)]

        if not filtered_events.empty:
            all_filtered_events.extend(filtered_events[['name', actual_date_column, 'ra', 'dec', 'redshift']].to_dict(orient='records'))

    return all_filtered_events if all_filtered_events else []

@mcp.tool()
def query_electromagnetic_event_by_name(
    event_name: str
) -> List[Dict[str, Any]]:
    """
    Query electromagnetic events by event name.

    Parameters
    ----------
    event_name : str
        The name of the electromagnetic event (e.g., "SN2005cv").

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing matching electromagnetic events.
        If no events match, an empty list is returned.
    """

    all_matches = []

    for sne_file_path in DEFAULT_SNE_PATHS:
        sne_path = Path(sne_file_path)

        if not sne_path.is_file():
            continue

        df = pd.read_csv(sne_path)

        if "discoverdate" in df.columns:
            df["discoverdate"] = pd.to_datetime(
                df["discoverdate"],
                format="%Y/%m/%d",
                errors="coerce"
            )

        matched = df[df["name"] == event_name]

        if not matched.empty:
            all_matches.extend(
                matched[["name", "discoverdate", "ra", "dec", "redshift"]]
                .to_dict(orient="records")
            )

    return all_matches

@mcp.tool()
def query_gw_events_by_time(
    date_str: str,
    time_window_days: int = 7
) -> List[str]:
    """
    Query gravitational-wave events within ±N days of a given date.

    Parameters
    ----------
    date_str : str
        Target date in "YYYY/MM/DD" format.
    time_window_days : int, optional
        Number of days before and after the given date to search, default is 7.

    Returns
    -------
    List[str]
        A list of matching GW full_names within the time window.
        Returns an empty list if no events are found.
    """

    idx_path = Path(DEFAULT_INDEX_PATH)
    if not idx_path.is_file():
        return []

    try:
        target_time = datetime.strptime(date_str, "%Y/%m/%d")
    except Exception:
        return []

    start_time = target_time - timedelta(days=time_window_days)
    end_time = target_time + timedelta(days=time_window_days)

    matched_full_names = set()

    with open(idx_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue

            short_name = record.get("short_name")
            full_name = record.get("full_name")

            if not short_name or not full_name:
                continue

            if not short_name.startswith("GW") or len(short_name) < 8:
                continue

            try:
                year = "20" + short_name[2:4]
                month = short_name[4:6]
                day = short_name[6:8]
                gw_time = datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d")
            except Exception:
                continue

            if start_time <= gw_time <= end_time:
                matched_full_names.add(full_name)

    return sorted(list(matched_full_names))

@mcp.tool()
def match_em_coordinate_to_gw_skymaps(
    em_counterpart: Dict[str, Any],
    gw_skymaps: List[Dict[str, Any]],
    coord_unit: Literal["rad", "deg"] = "deg",
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Given an EM counterpart coordinate, evaluate its credible level on multiple
    GW skymaps and render a combined all-sky plot overlaying all GW skymaps
    with the EM marker.

    If only asked to show the skymap of a GWxxxxxx event without any additional requirements, display any one 
    of the skymaps (DO NOT SHOW ALL waveforms). This helps keep the image simple.

    Parameters
    ----------
    em_counterpart : dict
        {"ra": float, "dec": float, "label": str}
    gw_skymaps : list[dict]
        Each item should be like:
        {"path": str, "label": str (optional)}
    coord_unit : "rad" | "deg"
        Unit of input EM coordinate.
    filename : str, optional
        Output image filename. If not provided, defaults to
        "EM_{label}_vs_{n}GW_overlay.png". Use this to customize the output filename.
        If the filename seems too complex, consider using a simpler one.

    Returns
    -------
    {
      "status": "ok" | "error",
      "em_counterpart": {...},
      "results": [
        {
          "skymap_path": str,
          "skymap_label": str,
          "credible_level": float,
          "in_50": bool,
          "in_90": bool
        },
        ...
      ],
      "best_match": {...} | None,
      "output_png": str | None,
      "message": str
    }
    """
    if not isinstance(em_counterpart, dict):
        return {
            "status": "error",
            "em_counterpart": em_counterpart,
            "results": [],
            "best_match": None,
            "output_png": None,
            "message": "em_counterpart must be a dict with keys: ra, dec, label.",
        }

    ra = em_counterpart.get("ra")
    dec = em_counterpart.get("dec")
    label = em_counterpart.get("label", "EM")

    if ra is None or dec is None:
        return {
            "status": "error",
            "em_counterpart": em_counterpart,
            "results": [],
            "best_match": None,
            "output_png": None,
            "message": "em_counterpart must include ra and dec.",
        }

    if not gw_skymaps:
        return {
            "status": "error",
            "em_counterpart": em_counterpart,
            "results": [],
            "best_match": None,
            "output_png": None,
            "message": "No GW skymaps provided.",
        }

    existing_skymaps: List[Dict[str, str]] = []
    for item in gw_skymaps:
        if not isinstance(item, dict):
            continue

        raw_path = item.get("path")
        raw_label = item.get("label")

        if raw_path and Path(raw_path).is_file():
            p = Path(raw_path)
            existing_skymaps.append(
                {
                    "path": str(p),
                    "label": str(raw_label) if raw_label is not None else p.stem,
                }
            )

    if not existing_skymaps:
        return {
            "status": "error",
            "em_counterpart": em_counterpart,
            "results": [],
            "best_match": None,
            "output_png": None,
            "message": "None of the provided skymap paths exist on disk.",
        }

    results: List[Dict[str, Any]] = []
    for item in existing_skymaps:
        skymap_path = item["path"]
        skymap_label = item["label"]

        credible_level = _compute_credible_level(
            skymap_file_path=skymap_path,
            ra=ra,
            dec=dec,
            coord_unit=coord_unit,
            nest=True,
        )

        results.append(
            {
                "skymap_path": skymap_path,
                "skymap_label": skymap_label,
                "credible_level": float(credible_level),
                "in_50": bool(credible_level <= 0.5),
                "in_90": bool(credible_level <= 0.9),
            }
        )

    best_match = min(results, key=lambda x: x["credible_level"]) if results else None

    DEFAULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    if filename is None:
        safe_label = _sanitize_filename(str(label))
        filename = f"EM_{safe_label}_vs_{len(existing_skymaps)}GW_overlay.png"
    elif not filename.lower().endswith(".png"):
        filename += ".png"

    filename = _sanitize_filename(filename)
    out_path = DEFAULT_CACHE_PATH / filename

    try:
        saved_png = _render_multiple_skymaps_with_em_marker(
            gw_skymaps=existing_skymaps,
            em_coord={"ra": ra, "dec": dec, "label": label},
            output_png=str(out_path),
            coord_unit=coord_unit,
        )
    except Exception as e:
        return {
            "status": "error",
            "em_counterpart": em_counterpart,
            "results": results,
            "best_match": best_match,
            "output_png": None,
            "message": f"Computed credible levels but failed to render overlay: {e}",
        }

    return {
        "status": "ok",
        "em_counterpart": {"ra": ra, "dec": dec, "label": label},
        "results": sorted(results, key=lambda x: x["credible_level"]),
        "best_match": best_match,
        "output_png": str(saved_png),
        "message": "EM coordinate assessed against multiple GW skymaps and overlay saved.",
    }

@mcp.tool()
def valid_distance_on_skymaps(
    skymap_file_path: str,
    coordinate: Dict[str, Any],
    distance: Optional[float] = None,
    redshift: Optional[float] = None,
    visual: bool = True,
    visual_file_name: Optional[str] = None,
    coord_unit: Literal["rad", "deg"] = "deg",
) -> Dict[str, Any]:
    """
    Evaluate the consistency of a luminosity distance value with a gravitational-wave skymap.

    Given a GW skymap and a sky coordinate (right ascension and declination), this tool:
    1. Computes the conditional luminosity-distance distribution at that coordinate
    2. If a distance value is provided, evaluates where it falls within the distribution
    3. Optionally visualizes the distribution with the target distance marked

    Parameters
    ----------
    skymap_file_path : str
        Path to the 3D GW skymap FITS file (e.g., from query_skymaps).
    coordinate : Dict[str, Any]
        Sky coordinate dictionary with keys:
        - "ra": float (right ascension)
        - "dec": float (declination)
        - "label": str (optional, for plotting)
    distance : float, optional
        Target luminosity distance in Mpc. If provided, the tool computes:
        - CDF value at this distance
        - Symmetric CDF score (2 * min(CDF, 1-CDF)), where values closer to 0
          indicate the distance is in the tail of the distribution
        If None, only the distribution statistics are returned.
    redshift : float, optional
        Target redshift. If provided, it will be converted to luminosity distance
        internally using astropy cosmology (Planck18). Cannot be used together with
        the 'distance' parameter.
    visual : bool, default True
        Whether to generate a visualization of the distance distribution.
    visual_file_name : str, optional
        Output filename for the plot. If not provided and visual=True,
        defaults to "{skymap_stem}_distance_dist.png".
    coord_unit : "rad" | "deg", default "deg"
        Unit of the input coordinate.

    Returns
    -------
    Dict[str, Any]
        {
            "status": "ok" | "error",
            "coordinate": {"ra": float, "dec": float, "label": str},
            "skymap_path": str,
            "statistics": {
                "mean_distance": float,
                "median_distance": float,
                "std_distance": float,
                "distance_90_lower": float,  # 5th percentile
                "distance_90_upper": float,  # 95th percentile
            },
            "target_distance": float | None,
            "target_cdf": float | None,
            "target_symmetric_score": float | None,
            "output_png": str | None,
            "message": str
        }

    Example
    -------
    # Check if a EM event at 100 Mpc is consistent with GW230518 skymap
    result = valid_distance_on_skymaps(
        skymap_file_path="path/to/skymap.fits",
        coordinate={"ra": 120.0, "dec": -30.0, "label": "EM2023xxx"},
        distance=100.0,
        visual=True
    )

    # Or use redshift directly
    result = valid_distance_on_skymaps(
        skymap_file_path="path/to/skymap.fits",
        coordinate={"ra": 120.0, "dec": -30.0, "label": "EM2023xxx"},
        redshift=0.023,
        visual=True
    )
    """
    from pathlib import Path

    fits_path = Path(skymap_file_path)

    # Handle redshift to distance conversion
    actual_distance = distance
    distance_source = "distance"  # Track whether we used 'distance' or 'redshift'

    if distance is not None and redshift is not None:
        return {
            "status": "error",
            "coordinate": coordinate,
            "skymap_path": str(fits_path),
            "statistics": None,
            "target_distance": None,
            "target_cdf": None,
            "target_symmetric_score": None,
            "output_png": None,
            "message": "Cannot specify both 'distance' and 'redshift' parameters. Use only one.",
        }

    if redshift is not None:
        if redshift < 0:
            return {
                "status": "error",
                "coordinate": coordinate,
                "skymap_path": str(fits_path),
                "statistics": None,
                "target_distance": None,
                "target_cdf": None,
                "target_symmetric_score": None,
                "output_png": None,
                "message": "Redshift cannot be negative",
            }
        try:
            z_array = np.atleast_1d(redshift)
            dL = cosmo.luminosity_distance(z_array)
            dL_mpc = dL.to(u.Mpc).value
            actual_distance = float(dL_mpc[0])
            distance_source = f"redshift (z={redshift})"
        except Exception as e:
            return {
                "status": "error",
                "coordinate": coordinate,
                "skymap_path": str(fits_path),
                "statistics": None,
                "target_distance": None,
                "target_cdf": None,
                "target_symmetric_score": None,
                "output_png": None,
                "message": f"Failed to convert redshift to luminosity distance: {e}",
            }
    if not fits_path.is_file():
        return {
            "status": "error",
            "coordinate": coordinate,
            "skymap_path": str(fits_path),
            "statistics": None,
            "target_distance": actual_distance,
            "target_cdf": None,
            "target_symmetric_score": None,
            "output_png": None,
            "message": f"Skymap file not found: {skymap_file_path}",
        }

    ra = coordinate.get("ra")
    dec = coordinate.get("dec")
    label = coordinate.get("label", "unknown")

    if ra is None or dec is None:
        return {
            "status": "error",
            "coordinate": coordinate,
            "skymap_path": str(fits_path),
            "statistics": None,
            "target_distance": actual_distance,
            "target_cdf": None,
            "target_symmetric_score": None,
            "output_png": None,
            "message": "Coordinate must include 'ra' and 'dec' keys.",
        }

    try:
        stats = _compute_distance_statistics(
            skymap_path=str(fits_path),
            ra=ra,
            dec=dec,
            distance=actual_distance,
            coord_unit=coord_unit,
            nest=False,
        )
    except Exception as e:
        return {
            "status": "error",
            "coordinate": coordinate,
            "skymap_path": str(fits_path),
            "statistics": None,
            "target_distance": actual_distance,
            "target_cdf": None,
            "target_symmetric_score": None,
            "output_png": None,
            "message": f"Failed to compute distance statistics: {e}",
        }

    statistics_summary = {
        "mean_distance": stats["mean_distance"],
        "median_distance": stats["median_distance"],
        "std_distance": stats["std_distance"],
        "distance_90_lower": stats["distance_90_lower"],
        "distance_90_upper": stats["distance_90_upper"],
    }

    output_png = None

    if visual:
        DEFAULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

        if visual_file_name is None:
            visual_file_name = f"{fits_path.stem}_distance_dist.png"
        elif not visual_file_name.lower().endswith(".png"):
            visual_file_name += ".png"

        visual_file_name = _sanitize_filename(visual_file_name)
        output_path = DEFAULT_CACHE_PATH / visual_file_name

        try:
            output_png = _render_distance_distribution(
                distance_grid=stats["distance_grid"],
                pdf=stats["pdf"],
                ra=ra,
                dec=dec,
                output_png=str(output_path),
                target_distance=actual_distance,
                coord_unit=coord_unit,
            )
        except Exception as e:
            return {
                "status": "error",
                "coordinate": coordinate,
                "skymap_path": str(fits_path),
                "statistics": statistics_summary,
                "target_distance": actual_distance,
                "target_cdf": stats.get("target_cdf"),
                "target_symmetric_score": stats.get("target_symmetric_score"),
                "output_png": None,
                "message": f"Computed statistics but failed to render plot: {e}",
            }

    message_parts = []
    message_parts.append(
        f"Median distance: {stats['median_distance']:.1f} Mpc "
        f"(90% CI: {stats['distance_90_lower']:.1f}-{stats['distance_90_upper']:.1f} Mpc)"
    )

    if actual_distance is not None:
        cdf_val = stats.get("target_cdf")
        score_val = stats.get("target_symmetric_score")
        if cdf_val is not None and score_val is not None:
            if score_val < 0.1:
                consistency = "in the tail (< 5%)"
            elif score_val < 0.5:
                consistency = "marginally consistent (5-25%)"
            else:
                consistency = "well consistent (> 25%)"
            message_parts.append(
                f"Target {distance_source} = {actual_distance:.1f} Mpc: CDF={cdf_val:.3f}, "
                f"symmetric score={score_val:.3f} ({consistency})"
            )

    if output_png:
        message_parts.append(f"Plot saved to {output_png}")

    return {
        "status": "ok",
        "coordinate": {"ra": ra, "dec": dec, "label": label},
        "skymap_path": str(fits_path),
        "statistics": statistics_summary,
        "target_distance": actual_distance,
        "target_cdf": stats.get("target_cdf"),
        "target_symmetric_score": stats.get("target_symmetric_score"),
        "output_png": output_png,
        "message": ". ".join(message_parts),
    }

if __name__ == "__main__":
    mcp.run()
