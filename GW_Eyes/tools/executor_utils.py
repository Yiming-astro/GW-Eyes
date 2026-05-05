from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Literal, Optional, List, Dict, Tuple

import numpy as np
import healpy as hp
from ligo.skymap.io import read_sky_map
from ligo.skymap.distance import conditional_pdf, marginal_ppf

def _load_index(index_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def _build_skymap_index(directory: str, index_name: str = "index.jsonl") -> None:
    base_path = Path(directory)
    index_path = base_path / index_name

    records = []

    for fits_file in sorted(base_path.glob("*.fits")):
        name = fits_file.name

        suffix = "_Skymap_PEDataRelease.fits"
        if not name.endswith(suffix):
            continue

        stem = name[: -len(suffix)]
        left, waveform = stem.rsplit("-", 1)
        left, full_name = left.rsplit("-", 1)

        catalog_label = left
        short_name = full_name.split("_", 1)[0]

        record = {
            "catalog_label": catalog_label,
            "short_name": short_name,
            "full_name": full_name,
            "waveform": waveform,
            "path": str(fits_file),
        }
        records.append(record)

    with open(index_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def _sanitize_filename(name: str) -> str:
    """
    Make a filesystem-friendly filename.
    Keeps letters/numbers/._- and replaces other characters with underscore.
    """
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name

def _infer_event_name(skymap_file_path: str) -> str:
    name = Path(skymap_file_path).name
    m = re.search(r"(GW\d{6}_\d{6})", name)
    if m:
        return m.group(1)
    m = re.search(r"(GW\d{6})", name)
    if m:
        return m.group(1)
    return "unknown_event"

def _compute_credible_level(
    skymap_file_path: str,
    ra: float,
    dec: float,
    coord_unit: Literal["rad", "deg"] = "deg",
    nest: bool = True,
) -> float:
    if coord_unit == "deg":
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)

    prob, _ = read_sky_map(skymap_file_path, nest=nest)
    prob = prob / prob.sum()

    theta = 0.5 * np.pi - dec
    phi = ra

    nside = hp.get_nside(prob)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)

    order = np.argsort(prob)[::-1]
    cumsum = np.cumsum(prob[order])

    credible = np.empty_like(cumsum)
    credible[order] = cumsum

    return float(credible[pix])

def _render_skymap_with_multiple_markers(
    skymap_file_path: str,
    coordinates: List[Dict[str, Any]],
    output_png: str,
    coord_unit: Literal["rad", "deg"] = "deg",
    skymap_label: Optional[str] = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from pathlib import Path
    from astropy.visualization.wcsaxes import WCSAxes
    import ligo.skymap.plot.allsky as allsky
    from ligo.skymap.io import read_sky_map

    # Read skymap and plot using consistent style with _render_multiple_skymaps_with_em_marker
    prob, _hdr = read_sky_map(skymap_file_path)

    fig = plt.figure(figsize=(10, 5.8))
    ax = plt.axes(projection="astro degrees mollweide")
    ax.grid()

    # Plot the skymap with "Reds" colormap (consistent style)
    ax.imshow_hpx(
        prob,
        cmap="Reds",
        alpha=0.7,
        zorder=100,
    )

    # Setup for markers
    dx_pt, dy_pt = 5, -5
    marker_z = 5000
    text_z = 5001

    default_color_list = ['#4E74B4', '#EB6E68', '#96D1E0', '#FB9680',
                          '#E9A375', '#FDBCA7']

    legend_handles = []

    # Add skymap label to legend
    if skymap_label is not None:
        cmap_obj = plt.get_cmap("Reds")
        legend_color = cmap_obj(0.75)
        legend_handles.append(
            mpatches.Patch(
                facecolor=legend_color,
                edgecolor="none",
                label=str(skymap_label),
            )
        )

    for idx, c in enumerate(coordinates):
        ra = c.get("ra")
        dec = c.get("dec")
        label = c.get("label", f"point_{idx}")

        if ra is None or dec is None:
            continue

        ra_plot = ra
        dec_plot = dec
        if coord_unit == "rad":
            ra_plot = np.rad2deg(ra_plot)
            dec_plot = np.rad2deg(dec_plot)

        marker_color = default_color_list[idx % len(default_color_list)]

        world_transform = ax.get_transform("world") if hasattr(ax, "get_transform") else None

        if world_transform is not None:
            ax.plot(
                ra_plot,
                dec_plot,
                transform=world_transform,
                marker="+",
                markersize=14,
                markeredgewidth=2.4,
                color=marker_color,
                linestyle="None",
                zorder=marker_z,
                clip_on=False,
            )

            text_transform = world_transform + mtransforms.ScaledTranslation(
                dx_pt / 72.0, dy_pt / 72.0, fig.dpi_scale_trans
            )

            ax.text(
                ra_plot,
                dec_plot,
                str(label),
                transform=text_transform,
                fontsize=10,
                color=marker_color,
                ha="left",
                va="top",
                zorder=text_z,
                clip_on=False,
            )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="+",
                    color=marker_color,
                    linestyle="None",
                    markersize=12,
                    markeredgewidth=2.2,
                    label=str(label),
                )
            )

    # Add legend at bottom
    fig.subplots_adjust(bottom=0.22)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(max(len(legend_handles), 1), 4),
        frameon=False,
        fontsize=9,
    )

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)

def _build_output_name(
    skymap_file_path: str,
    ra: float,
    dec: float,
    coord_unit: str,
) -> str:
    event = _infer_event_name(skymap_file_path)
    tag = f"{event}_ra{ra:.4f}_dec{dec:.4f}_{coord_unit}.png"
    return _sanitize_filename(tag)

def _create_transparent_cmap(cmap_name: str, threshold: float = 0.05):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 256))
    colors[: int(threshold * 256), -1] = 0
    return mcolors.ListedColormap(colors)

def _render_multiple_skymaps_with_em_marker(
    gw_skymaps: List[Dict[str, str]],
    em_coord: Dict[str, Any],
    output_png: str,
    coord_unit: Literal["rad", "deg"] = "deg",
    cmap_names: Optional[List[str]] = None,
    threshold: float = 0.05,
    alpha: float = 0.7,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from astropy.io import fits
    from astropy.visualization.wcsaxes import WCSAxes
    import ligo.skymap.plot.allsky as allsky
    from ligo.skymap.io import read_sky_map

    if cmap_names is None:
        cmap_names = [
            "Blues",
            "Purples",
            "Greens",
            "Oranges",
            "Greys",
            "PuBu",
            "PuRd",
            "YlOrRd",
            "YlGnBu",
        ]

    probs: List[Any] = []
    labels: List[str] = []

    for item in gw_skymaps:
        prob, _hdr = read_sky_map(item["path"])
        probs.append(prob)
        labels.append(item["label"])

    fig = plt.figure(figsize=(10, 5.8))
    ax = plt.axes(projection="astro degrees mollweide")
    ax.grid()

    n_maps = len(probs)
    legend_handles = []

    for i, (prob, gw_label) in enumerate(zip(probs, labels)):
        if i == n_maps - 1:
            display_cmap_name = "Reds"
            ax.imshow_hpx(
                prob,
                cmap=display_cmap_name,
                alpha=0.7,
                zorder=500,
            )
        else:
            base_cmap_name = cmap_names[i % len(cmap_names)]
            display_cmap_name = base_cmap_name
            cmap = _create_transparent_cmap(base_cmap_name, threshold=threshold)
            ax.imshow_hpx(
                prob,
                cmap=cmap,
                alpha=alpha,
                zorder=1000,
            )

        cmap_obj = plt.get_cmap(display_cmap_name)
        legend_color = cmap_obj(0.75)
        legend_handles.append(
            mpatches.Patch(
                facecolor=legend_color,
                edgecolor="none",
                label=str(gw_label),
            )
        )

    ra = em_coord.get("ra")
    dec = em_coord.get("dec")
    em_label = em_coord.get("label", "EM")

    if ra is not None and dec is not None:
        ra_plot = ra
        dec_plot = dec
        if coord_unit == "rad":
            ra_plot = np.rad2deg(ra_plot)
            dec_plot = np.rad2deg(dec_plot)

        world_transform = ax.get_transform("world") if hasattr(ax, "get_transform") else None

        marker_z = 5000
        text_z = 5001
        dx_pt, dy_pt = 4, -4

        if world_transform is not None:
            ax.plot(
                ra_plot,
                dec_plot,
                transform=world_transform,
                marker="+",
                markersize=14,
                markeredgewidth=2.4,
                color="#4E74B4",
                linestyle="None",
                zorder=marker_z,
                clip_on=False,
            )

            text_transform = world_transform + mtransforms.ScaledTranslation(
                dx_pt / 72.0,
                dy_pt / 72.0,
                fig.dpi_scale_trans,
            )
            ax.text(
                ra_plot,
                dec_plot,
                str(em_label),
                transform=text_transform,
                fontsize=10,
                color="#4E74B4",
                ha="left",
                va="top",
                zorder=text_z,
                clip_on=False,
            )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="+",
                    color="#4E74B4",
                    linestyle="None",
                    markersize=12,
                    markeredgewidth=2.2,
                    label=f"EM: {em_label}",
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

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

def _compute_distance_statistics(
    skymap_path: str,
    ra: float,
    dec: float,
    distance: Optional[float] = None,
    coord_unit: Literal["rad", "deg"] = "deg",
    nest: bool = False,
    n_distance_bins: int = 1000,
) -> Dict[str, Any]:
    """
    Compute distance-related statistics for a given coordinate on a GW skymap.

    Parameters
    ----------
    skymap_path : str
        Path to the 3D skymap FITS file.
    ra : float
        Right ascension of the coordinate.
    dec : float
        Declination of the coordinate.
    distance : float, optional
        If provided, compute statistics for this specific distance value.
    coord_unit : "rad" | "deg"
        Unit of input coordinates.
    nest : bool
        HEALPix nesting order. Default is False (RING order).
    n_distance_bins : int
        Number of bins for distance grid.

    Returns
    -------
    dict
        {
            "ra": float,
            "dec": float,
            "distance_grid": np.ndarray,
            "pdf": np.ndarray,
            "mean_distance": float,
            "median_distance": float,
            "std_distance": float,
            "distance_90_lower": float,  # 5th percentile
            "distance_90_upper": float,  # 95th percentile
            "target_distance": Optional[float],
            "target_cdf": Optional[float],
            "target_symmetric_score": Optional[float],
        }
    """
    if coord_unit == "deg":
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
    else:
        ra_rad = ra
        dec_rad = dec

    (prob, distmu, distsigma, distnorm), meta = read_sky_map(skymap_path, distances=True)

    nside = hp.npix2nside(len(prob))
    theta = 0.5 * np.pi - dec_rad
    phi = ra_rad
    ipix = hp.ang2pix(nside, theta, phi, nest=nest)

    r_max = marginal_ppf(0.999, prob, distmu, distsigma, distnorm)
    r = np.linspace(0, r_max, n_distance_bins)

    pdf = conditional_pdf(r, distmu[ipix], distsigma[ipix], distnorm[ipix])

    norm = np.trapz(pdf, r)
    if norm <= 0:
        raise ValueError("PDF normalization failed: integral is not positive.")
    pdf_normalized = pdf / norm

    cdf = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (pdf_normalized[1:] + pdf_normalized[:-1]) * np.diff(r))
    ])
    cdf = cdf / cdf[-1]

    mean_distance = float(np.trapz(r * pdf_normalized, r))
    median_distance = float(np.interp(0.5, cdf, r))
    std_distance = float(np.sqrt(np.trapz((r - mean_distance) ** 2 * pdf_normalized, r)))

    distance_90_lower = float(np.interp(0.05, cdf, r))
    distance_90_upper = float(np.interp(0.95, cdf, r))

    result: Dict[str, Any] = {
        "ra": ra,
        "dec": dec,
        "distance_grid": r,
        "pdf": pdf_normalized,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "std_distance": std_distance,
        "distance_90_lower": distance_90_lower,
        "distance_90_upper": distance_90_upper,
        "target_distance": None,
        "target_cdf": None,
        "target_symmetric_score": None,
    }

    if distance is not None:
        F_d0 = np.interp(distance, r, cdf, left=0.0, right=1.0)
        t = 2.0 * min(F_d0, 1.0 - F_d0)
        result["target_distance"] = distance
        result["target_cdf"] = float(F_d0)
        result["target_symmetric_score"] = float(t)

    return result

def _render_distance_distribution(
    distance_grid: np.ndarray,
    pdf: np.ndarray,
    ra: float,
    dec: float,
    output_png: str,
    target_distance: Optional[float] = None,
    title: Optional[str] = None,
    coord_unit: Literal["rad", "deg"] = "deg",
) -> str:
    """
    Render the distance distribution plot.

    Parameters
    ----------
    distance_grid : np.ndarray
        Grid of distance values.
    pdf : np.ndarray
        Probability density function values.
    ra : float
        Right ascension (for title).
    dec : float
        Declination (for title).
    output_png : str
        Output file path.
    target_distance : float, optional
        If provided, mark this distance on the plot.
    title : str, optional
        Custom title for the plot.
    coord_unit : "rad" | "deg"
        Unit of coordinates (for display).

    Returns
    -------
    str
        Path to the saved plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ra_display = ra
    dec_display = dec
    if coord_unit == "rad":
        ra_display = np.rad2deg(ra)
        dec_display = np.rad2deg(dec)

    # Compute CDF for median and percentile calculations
    cdf = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(distance_grid))
    ])
    cdf = cdf / cdf[-1]

    # Compute median distance from CDF
    median_distance = float(np.interp(0.5, cdf, distance_grid))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(distance_grid, pdf, lw=2, label="Conditional distance PDF", color="#4E74B4")
    ax.fill_between(distance_grid, pdf, 0, color="#4E74B4", alpha=0.4)
    ax.axvline(
        median_distance,
        ls="--",
        lw=1.5,
        label=f"Median = {median_distance:.1f} Mpc",
        color="gray",
        alpha=0.7,
    )

    if target_distance is not None:
        ax.axvline(
            target_distance,
            ls="--",
            lw=2,
            label=f"Target = {target_distance:.1f} Mpc",
            color="#EB6E68",
        )

    ax.set_xlabel("Luminosity distance [Mpc]", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_ylim(bottom=0)

    if title is None:
        ax.set_title(
            f"p(d_L | RA={ra_display:.1f}{chr(176)}, Dec={dec_display:.1f}{chr(176)})"
        )
    else:
        ax.set_title(title)

    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)