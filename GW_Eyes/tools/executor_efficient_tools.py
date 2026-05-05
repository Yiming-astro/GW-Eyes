from __future__ import annotations

import heapq
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.coordinates import Angle

from mcp.server.fastmcp import FastMCP

from GW_Eyes.tools.executor_utils import (
    _compute_credible_level,
    _load_index,
)
from GW_Eyes.src.config import get_em_csv_paths, get_gw_index_file, get_output_path

# Default index path (adjust if you keep it elsewhere)
DEFAULT_INDEX_PATH = get_gw_index_file()
DEFAULT_CACHE_PATH = get_output_path()
DEFAULT_SNE_PATHS = get_em_csv_paths()

mcp = FastMCP("executor_efficient_tools")

@mcp.tool()
def search_electromagnetic_counterpart_for_gw(
    gw_skymap_file: str,
    time_before_days: float = 0,
    time_after_days: float = 7,
    cred_2d_threshold: float = 0.5,
    distance_z_threshold: float = 1,
    expect_nums: int = 3,
) -> Dict[str, Any]:
    """
    Search for electromagnetic counterparts of a gravitational-wave event.

    This is a high-performance tool optimized for computational intensive scenarios.
    It focuses purely on efficient filtering and returns structured data without visualization.

    This tool efficiently filters EM candidates through three layers:
    1. Time window filter: EM events within [gw_trigger - time_before_days, gw_trigger + time_after_days]
    2. 2D spatial credibility: EM coordinates must have credible_level <= cred_2d_threshold on the GW skymap
    3. Distance consistency: EM distance/redshift must have z_score <= distance_z_threshold

    If the user does not specifically request multiple waveforms, you may use any one of the skymaps.
    Do not use all of them, as this helps reduce computational load.

    If the user does not specify a particular time window (e.g., time_before_days or time_after_days),
    use the default time range. Within this range, even if no corresponding counterpart is found,
    do not expand the search window to run this tool again unless the user requests it.
    This helps reduce computational load.

    Parameters
    ----------
    gw_skymap_file : str
        Path to the GW skymap FITS file (e.g., from query_skymaps).
    time_before_days : float, default 0
        Number of days before the GW trigger time to search for EM events.
    time_after_days : float, default 7
        Number of days after the GW trigger time to search for EM events.
    cred_2d_threshold : float, default 0.5
        Maximum credible level for 2D spatial consistency. Only EM events with
        credible_level <= this threshold pass the 2D filter.
    distance_z_threshold : float, default 1
        z_score >= this threshold pass the distance filter.
    expect_nums : int, default 3
        Maximum number of candidates to return.

    Returns
    -------
    Dict[str, Any]
        {
            "status": "ok" | "error",
            "gw_skymap_file": str,
            "gw_trigger_time": str | None,
            "time_window": {"before_days": float, "after_days": float},
            "thresholds": {"2d_credit": float, "distance_credit": float},
            # "candidates" means the event that passes the threshold
            "candidates": [
                {
                    "name": str,
                    "ra": float,
                    "dec": float,
                    "redshift": float | None,
                    "distance_mpc": float | None,
                    "date": str,
                    "cred_2d_level": float,
                    "distance_zscore": float | None,
                    "distance_statistics": dict | None,
                },
                ...
            ],
            # "cache_events" means the event not pass the threshold but maybe needed
            "cache_events": [
                {
                    "name": str,
                    "ra": float,
                    "dec": float,
                    "redshift": float | None,
                    "distance_mpc": float | None,
                    "date": str,
                    "cred_2d_level": float,
                    "distance_zscore": float | None,
                    "distance_statistics": dict | None,
                },
                ...
            ],
            "total_candidates": int,
            "total_cache_events": int,
            "message": str
        }
    """
    from ligo.skymap.distance import conditional_pdf, marginal_ppf
    from ligo.skymap.io import read_sky_map
    import healpy as hp

    skymap_path = Path(gw_skymap_file)

    if not skymap_path.is_file():
        return {
            "status": "error",
            "gw_skymap_file": gw_skymap_file,
            "gw_trigger_time": None,
            "time_window": None,
            "thresholds": None,
            "candidates": [],
            "cache_events": [],
            "total_candidates": 0,
            "total_cache_events": 0,
            "message": f"Skymap file not found: {gw_skymap_file}",
        }

    # Extract GW event info from skymap filename
    event_name_match = re.search(r"(GW\d{6}(?:_\d{6})?)", skymap_path.name)
    if not event_name_match:
        return {
            "status": "error",
            "gw_skymap_file": gw_skymap_file,
            "gw_trigger_time": None,
            "time_window": None,
            "thresholds": None,
            "candidates": [],
            "cache_events": [],
            "total_candidates": 0,
            "total_cache_events": 0,
            "message": "Could not extract GW event name from skymap filename",
        }

    gw_short_name = event_name_match.group(1).split("_")[0]

    # Calculate GW trigger time from event name
    try:
        year = "20" + gw_short_name[2:4]
        month = gw_short_name[4:6]
        day = gw_short_name[6:8]
        gw_trigger_time_str = f"{year}/{month}/{day}"
        gw_trigger_time = datetime.strptime(gw_trigger_time_str, "%Y/%m/%d")
    except Exception as e:
        return {
            "status": "error",
            "gw_skymap_file": gw_skymap_file,
            "gw_trigger_time": None,
            "time_window": None,
            "thresholds": None,
            "candidates": [],
            "cache_events": [],
            "total_candidates": 0,
            "total_cache_events": 0,
            "message": f"Failed to parse GW trigger time: {e}",
        }

    # Step 1: Time window filter - collect all EM events within the time window
    time_filtered_events = []

    for em_file_path in DEFAULT_SNE_PATHS:
        em_path = Path(em_file_path)
        if not em_path.is_file():
            continue

        try:
            df = pd.read_csv(em_path)
        except Exception:
            continue

        # Determine which date column to use
        maxdate_series = pd.to_datetime(df["maxdate"], format="%Y/%m/%d", errors="coerce") if "maxdate" in df.columns else pd.Series(pd.NaT, index=df.index)
        discoverdate_series = pd.to_datetime(df["discoverdate"], format="%Y/%m/%d", errors="coerce") if "discoverdate" in df.columns else pd.Series(pd.NaT, index=df.index)
        effective_date = maxdate_series.fillna(discoverdate_series)

        start_time = gw_trigger_time - timedelta(days=time_before_days)
        end_time = gw_trigger_time + timedelta(days=time_after_days)

        filtered = df[
            (effective_date >= start_time) &
            (effective_date <= end_time) &
            (df["ra"].notna()) &
            (df["dec"].notna())
        ].copy()
        filtered["effective_date"] = effective_date.loc[filtered.index]

        if not filtered.empty:
            for _, row in filtered.iterrows():
                # Convert HMS/DMS format strings to decimal degrees
                ra_deg = float(Angle(row["ra"], unit="hour").deg)
                dec_deg = float(Angle(row["dec"], unit="deg").deg)

                time_filtered_events.append({
                    "name": row["name"],
                    "ra": ra_deg,
                    "dec": dec_deg,
                    "redshift": row.get("redshift"),
                    "date": row["effective_date"],
                    "source_file": em_file_path,
                })

    if not time_filtered_events:
        return {
            "status": "ok",
            "gw_skymap_file": gw_skymap_file,
            "gw_trigger_time": gw_trigger_time_str,
            "time_window": {"before_days": time_before_days, "after_days": time_after_days},
            "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
            "candidates": [],
            "cache_events": [],
            "total_candidates": 0,
            "total_cache_events": 0,
            "message": "No EM events found within the specified time window",
        }

    # Track number of events that passed Step 1 (time window filter)
    n_time_filtered = len(time_filtered_events)

    # Step 2: 2D spatial credibility filter
    spatial_filtered_events = []
    fallback_heap = []

    for event in time_filtered_events:
        credible_level = _compute_credible_level(
            skymap_file_path=str(skymap_path),
            ra=event["ra"],
            dec=event["dec"],
            coord_unit="deg",
            nest=True,
        )
        event["cred_2d_level"] = credible_level
        event["passed_2d"] = credible_level <= cred_2d_threshold  # Track if passed 2D test
        # print('event=', event['name'], ' credible_level=', credible_level)
        if credible_level <= cred_2d_threshold:
            spatial_filtered_events.append(event)
        else:
            if len(fallback_heap) < expect_nums:
                heapq.heappush(fallback_heap, (-credible_level, id(event), event))
            else:
                if credible_level < -fallback_heap[0][0]:
                    heapq.heappushpop(fallback_heap, (-credible_level, id(event), event))

    spatial_cache = [heapq.heappop(fallback_heap)[2] for _ in range(len(fallback_heap))]
    spatial_cache.reverse()

    # Track number of events that actually passed Step 2 filter
    n_spatial_passed = len(spatial_filtered_events)

    # Combine all events that need distance computation:
    # - Events that passed 2D test (spatial_filtered_events) -> potential candidates
    # - Events that didn't pass 2D test but are best among failed (spatial_cache) -> cache only
    all_events_for_distance = spatial_filtered_events + spatial_cache

    # If no events at all, return empty
    if not all_events_for_distance:
        return {
            "status": "ok",
            "gw_skymap_file": gw_skymap_file,
            "gw_trigger_time": gw_trigger_time_str,
            "time_window": {"before_days": time_before_days, "after_days": time_after_days},
            "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
            "candidates": [],
            "cache_events": [],
            "total_candidates": 0,
            "total_cache_events": 0,
            "message": f"No EM events passed 2D spatial credibility filter (threshold <= {cred_2d_threshold}) and no cache available",
        }

    # Step 3: Distance consistency filter
    print('Step 3')
    # candidates_list: passed BOTH 2D and distance tests -> these go to candidates_output
    # cache_from_distance: passed 2D but failed distance test -> cache_events
    # cache_from_2d: failed 2D test (from spatial_cache) -> always cache_events regardless of distance result
    candidates_list = []
    cache_from_distance = []
    cache_from_2d = []

    (prob, distmu, distsigma, distnorm), _ = read_sky_map(str(skymap_path), distances=True)
    nside = hp.npix2nside(len(prob))

    for event in all_events_for_distance:
        passed_2d = event.get("passed_2d", False)

        if event.get("redshift") is None or pd.isna(event["redshift"]):
            event["distance_zscore"] = None
            event["distance_mpc"] = None
            event["distance_statistics"] = None
            # No redshift means we can't do distance test, treat as passed for distance
            # But still need to check if passed_2d
            if passed_2d:
                candidates_list.append(event)
            else:
                cache_from_2d.append(event)
            continue

        try:
            z_array = np.atleast_1d(event["redshift"])
            dL = cosmo.luminosity_distance(z_array)
            target_distance = float(dL.to(u.Mpc).value[0])

            ra_rad = np.deg2rad(event["ra"])
            dec_rad = np.deg2rad(event["dec"])
            theta = 0.5 * np.pi - dec_rad
            phi = ra_rad
            ipix = hp.ang2pix(nside, theta, phi, nest=False)

            mu_pix = distmu[ipix]
            sigma_pix = distsigma[ipix]

            event["distance_mpc"] = target_distance
            event["distance_statistics"] = {
                "mean_distance": float(mu_pix) if np.isfinite(mu_pix) else None,
                "std_distance": float(sigma_pix) if np.isfinite(sigma_pix) else None,
            }

            if np.isfinite(mu_pix) and np.isfinite(sigma_pix) and sigma_pix > 0:
                zscore = abs(target_distance - mu_pix) / sigma_pix
                event["distance_zscore"] = float(zscore)
                passed_distance = zscore <= distance_z_threshold

                # Classification based on both tests
                if passed_2d and passed_distance:
                    # Passed BOTH tests -> candidate
                    candidates_list.append(event)
                elif passed_2d and not passed_distance:
                    # Passed 2D but failed distance -> cache
                    cache_from_distance.append(event)
                else:
                    # Failed 2D (regardless of distance) -> cache
                    cache_from_2d.append(event)
            else:
                event["distance_zscore"] = None
                # Can't compute distance, treat as passed for distance
                if passed_2d:
                    candidates_list.append(event)
                else:
                    cache_from_2d.append(event)

        except Exception:
            continue

    # Track number of events that passed all tests
    n_distance_passed = len(candidates_list)

    # Sort by 2D credible level (best first) and then by distance zscore (best first, smaller is better)
    def sort_key(e):
        cred = e.get("cred_2d_level", 1.0)
        dist_score = e.get("distance_zscore")
        # Use positive distance zscore so smaller is better
        dist_rank = dist_score if dist_score is not None else float('inf')
        return (cred, dist_rank)

    # Sort candidates and cache lists
    candidates_list.sort(key=sort_key)
    cache_from_distance.sort(key=sort_key)
    cache_from_2d.sort(key=sort_key)

    # Format candidates output (events that passed BOTH 2D and distance tests)
    candidates_output = []
    for event in candidates_list[:expect_nums]:
        candidates_output.append({
            "name": event["name"],
            "ra": event["ra"],
            "dec": event["dec"],
            "redshift": event.get("redshift"),
            "distance_mpc": event.get("distance_mpc"),
            "date": str(event["date"]),
            "cred_2d_level": event.get("cred_2d_level"),
            "distance_zscore": event.get("distance_zscore"),
            "distance_statistics": event.get("distance_statistics"),
        })

    # Format cache_events output (events that didn't pass at least one test)
    # Combine cache_from_distance (passed 2D, failed distance) and cache_from_2d (failed 2D)
    # Sort the combined cache and keep top expect_nums
    combined_cache = cache_from_distance + cache_from_2d
    combined_cache.sort(key=sort_key)
    cache_output = []
    for event in combined_cache[:expect_nums]:
        cache_output.append({
            "name": event["name"],
            "ra": event["ra"],
            "dec": event["dec"],
            "redshift": event.get("redshift"),
            "distance_mpc": event.get("distance_mpc"),
            "date": str(event["date"]),
            "cred_2d_level": event.get("cred_2d_level"),
            "distance_zscore": event.get("distance_zscore"),
            "distance_statistics": event.get("distance_statistics"),
        })

    n_candidates = len(candidates_output)
    n_cache = len(cache_output)

    # Build informative message
    message_parts = []
    message_parts.append(f"Step 1 (time window): {n_time_filtered} event(s)")
    message_parts.append(f"Step 2 (2D spatial): {n_spatial_passed} event(s)")
    message_parts.append(f"Step 3 (distance): {n_distance_passed} event(s) passed both tests")

    message = f"Found {n_candidates} EM candidate(s) that passed BOTH 2D and distance tests. " + "; ".join(message_parts) + "."

    if n_cache > 0:
        message += f" Additionally, {n_cache} event(s) in cache_events did not pass one or both threshold tests."

    if n_candidates == 0 and n_cache > 0:
        message += " No events passed all filtering steps; check cache_events for best available options."

    return {
        "status": "ok",
        "gw_skymap_file": gw_skymap_file,
        "gw_trigger_time": gw_trigger_time_str,
        "time_window": {"before_days": time_before_days, "after_days": time_after_days},
        "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
        "candidates": candidates_output,
        "cache_events": cache_output,
        "total_candidates": n_candidates,
        "total_cache_events": n_cache,
        "message": message,
    }

@mcp.tool()
def search_gw_counterpart_for_electromagnetic_event(
    event_time: str,
    coordinate: Dict[str, Any],
    redshift: float | None = None,
    time_before_days: float = 7,
    time_after_days: float = 0,
    cred_2d_threshold: float = 0.5,
    distance_z_threshold: float = 1,
) -> Dict[str, Any]:
    """
    Search for gravitational-wave counterparts of an electromagnetic event.

    This tool efficiently filters GW candidates through three layers:
    1. Time window filter: GW events within [em_event_time - time_before_days, em_event_time + time_after_days]
    2. 2D spatial credibility: GW skymap must have credible_level <= cred_2d_threshold at EM coordinates
    3. Distance consistency: GW distance must have z_score <= distance_z_threshold compared to EM redshift

    Parameters
    ----------
    event_time : str
        The electromagnetic event trigger time (format: "YYYY/MM/DD" or "YYYY-MM-DD").
    coordinate : Dict[str, Any]
        Sky coordinate dictionary with keys:
        - "ra": float (right ascension)
        - "dec": float (declination)
    redshift : float | None, default None
        The redshift of the electromagnetic event. Optional, used for distance consistency check.
    time_before_days : float, default 7
        Number of days before the EM event time to search for GW events.
    time_after_days : float, default 0
        Number of days after the EM event time to search for GW events.
    cred_2d_threshold : float, default 0.5
        Maximum credible level for 2D spatial consistency. Only GW events with
        credible_level <= this threshold pass the 2D filter.
    distance_z_threshold : float, default 1
        z_score <= this threshold pass the distance filter.

    Returns
    -------
    Dict[str, Any]
        {
            "status": "ok" | "error",
            "em_event_time": str,
            "em_coordinate": {"ra": float, "dec": float},
            "em_redshift": float | None,
            "time_window": {"before_days": float, "after_days": float},
            "thresholds": {"2d_credit": float, "distance_credit": float},
            "gw_candidates_time_filtering": [
                {
                    "gw_event_name": str,
                    "gw_trigger_time": str,
                    "skymap_file": str,
                },
                ...
            ],
            "candidates": [
                {
                    "gw_event_name": str,
                    "gw_trigger_time": str,
                    "skymap_file": str,
                    "cred_2d_level": float,
                    "distance_zscore": float | None,
                    "distance_statistics": dict | None,
                },
                ...
            ],
            "total_candidates": int,
            "message": str
        }
    """
    # Parse event_time (support both "YYYY/MM/DD" and "YYYY-MM-DD" formats)
    try:
        event_time = event_time.replace("-", "/")
        em_trigger_time = datetime.strptime(event_time, "%Y/%m/%d")
        em_event_time_str = event_time
    except Exception as e:
        return {
            "status": "error",
            "em_event_time": event_time,
            "em_coordinate": coordinate,
            "em_redshift": redshift,
            "time_window": {"before_days": time_before_days, "after_days": time_after_days},
            "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
            "gw_candidates_time_filtering": [],
            "candidates": [],
            "total_candidates": 0,
            "message": f"Failed to parse event_time: {e}",
        }

    # Step 1: Time window filter - find GW events within [em_trigger_time - time_before_days, em_trigger_time + time_after_days]
    start_time = em_trigger_time - timedelta(days=time_before_days)
    end_time = em_trigger_time + timedelta(days=time_after_days)

    idx_path = Path(DEFAULT_INDEX_PATH)
    if not idx_path.is_file():
        return {
            "status": "error",
            "em_event_time": em_event_time_str,
            "em_coordinate": coordinate,
            "em_redshift": redshift,
            "time_window": {"before_days": time_before_days, "after_days": time_after_days},
            "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
            "gw_candidates_time_filtering": [],
            "candidates": [],
            "total_candidates": 0,
            "message": f"GW index file not found: {DEFAULT_INDEX_PATH}",
        }

    records = _load_index(idx_path)
    gw_candidates_time_filtering = []

    for r in records:
        short_name = r.get("short_name")
        full_name = r.get("full_name")
        skymap_path = r.get("path")

        if not short_name or not full_name or not skymap_path:
            continue

        if not short_name.startswith("GW") or len(short_name) < 8:
            continue

        # Calculate GW trigger time from short_name (format: GWYYMMDD)
        try:
            year = "20" + short_name[2:4]
            month = short_name[4:6]
            day = short_name[6:8]
            gw_trigger_time = datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d")
            gw_trigger_time_str = f"{year}/{month}/{day}"
        except Exception:
            continue

        if start_time <= gw_trigger_time <= end_time:
            gw_candidates_time_filtering.append({
                "gw_event_name": full_name,
                "gw_trigger_time": gw_trigger_time_str,
                "skymap_file": skymap_path,
            })

    # Step 2: 2D spatial credibility filter
    ra = coordinate.get("ra")
    dec = coordinate.get("dec")

    if ra is None or dec is None:
        return {
            "status": "error",
            "em_event_time": em_event_time_str,
            "em_coordinate": coordinate,
            "em_redshift": redshift,
            "time_window": {"before_days": time_before_days, "after_days": time_after_days},
            "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
            "gw_candidates_time_filtering": gw_candidates_time_filtering,
            "candidates": [],
            "total_candidates": 0,
            "message": "Invalid coordinate: ra and dec are required.",
        }

    spatial_filtered_candidates = []
    for gw_candidate in gw_candidates_time_filtering:
        skymap_file = gw_candidate["skymap_file"]
        skymap_path = Path(skymap_file)

        if not skymap_path.is_file():
            continue

        credible_level = _compute_credible_level(
            skymap_file_path=str(skymap_path),
            ra=ra,
            dec=dec,
            coord_unit="deg",
            nest=True,
        )

        if credible_level <= cred_2d_threshold:
            gw_candidate["cred_2d_level"] = credible_level
            spatial_filtered_candidates.append(gw_candidate)

    # Step 3: Distance consistency filter (TODO)

    # If redshift is None, skip Step 3
    if redshift is None:
        return {
            "status": "ok",
            "em_event_time": em_event_time_str,
            "em_coordinate": coordinate,
            "em_redshift": redshift,
            "time_window": {"before_days": time_before_days, "after_days": time_after_days},
            "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
            "gw_candidates_time_filtering": gw_candidates_time_filtering,
            "candidates": spatial_filtered_candidates,
            "total_candidates": len(spatial_filtered_candidates),
            "message": f"Step 1: {len(gw_candidates_time_filtering)} GW candidate(s); Step 2: {len(spatial_filtered_candidates)} passed 2D filter. Step 3 skipped: EM event redshift is None.",
        }

    # Step 3: Distance consistency filter
    from ligo.skymap.io import read_sky_map
    import healpy as hp

    distance_filtered_candidates = []
    # Calculate target distance from redshift
    z_array = np.atleast_1d(redshift)
    dL = cosmo.luminosity_distance(z_array)
    target_distance = float(dL.to(u.Mpc).value[0])

    # Get pixel index for EM coordinate
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    theta = 0.5 * np.pi - dec_rad
    phi = ra_rad

    for gw_candidate in spatial_filtered_candidates:
        skymap_path = Path(gw_candidate["skymap_file"])
        if not skymap_path.is_file():
            continue

        try:
            (prob, distmu, distsigma, distnorm), _ = read_sky_map(str(skymap_path), distances=True)
            nside = hp.npix2nside(len(prob))
            ipix = hp.ang2pix(nside, theta, phi, nest=False)

            mu_pix = distmu[ipix]
            sigma_pix = distsigma[ipix]

            gw_candidate["distance_mpc"] = target_distance
            gw_candidate["distance_statistics"] = {
                "mean_distance": float(mu_pix) if np.isfinite(mu_pix) else None,
                "std_distance": float(sigma_pix) if np.isfinite(sigma_pix) else None,
            }

            if np.isfinite(mu_pix) and np.isfinite(sigma_pix) and sigma_pix > 0:
                zscore = abs(target_distance - mu_pix) / sigma_pix
                gw_candidate["distance_zscore"] = float(zscore)

                if zscore <= distance_z_threshold:
                    distance_filtered_candidates.append(gw_candidate)
            else:
                gw_candidate["distance_zscore"] = None
                distance_filtered_candidates.append(gw_candidate)

        except Exception:
            continue

    return {
        "status": "ok",
        "em_event_time": em_event_time_str,
        "em_coordinate": coordinate,
        "em_redshift": redshift,
        "time_window": {"before_days": time_before_days, "after_days": time_after_days},
        "thresholds": {"2d_credit": cred_2d_threshold, "distance_credit": distance_z_threshold},
        "gw_candidates_time_filtering": gw_candidates_time_filtering,
        "candidates": distance_filtered_candidates,
        "total_candidates": len(distance_filtered_candidates),
        "message": f"Step 1: {len(gw_candidates_time_filtering)} GW candidate(s); Step 2: {len(spatial_filtered_candidates)} passed 2D filter; Step 3: {len(distance_filtered_candidates)} passed distance filter.",
    }


if __name__ == "__main__":
    mcp.run()
