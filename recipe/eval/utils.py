from typing import List, Dict, Any
import numpy as np
import healpy as hp
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
from ligo.skymap.io import read_sky_map
from ligo.skymap.distance import marginal_ppf


def sample_events_from_credible_region(
    skymap_file_path: str,
    candidate_num: int,
    confuse_num: int,
    unrelated_num: int,
    cred_2d_threshold: float,
    distance_credit_threshold: float,
    nest: bool = True,
    weighted_valid: bool = False,
    weighted_unrelated: bool = False
) -> Dict[str, List[Dict[str, float]]]:
    (prob, distmu, distsigma, distnorm), _ = read_sky_map(
        skymap_file_path, nest=nest, distances=True
    )
    prob = prob / prob.sum()

    order = np.argsort(prob)[::-1]
    cumsum = np.cumsum(prob[order])
    credible = np.empty_like(prob)
    credible[order] = cumsum

    candidate_valid_pixels = np.where(credible < 0.05)[0]
    valid_pixels = np.where(credible < cred_2d_threshold-0.1)[0]
    unrelated_pixels = np.where(credible > cred_2d_threshold+0.1)[0]

    if len(candidate_valid_pixels) == 0:
        raise ValueError("No pixels found with credible level below 0.05.")
    if len(valid_pixels) == 0:
        raise ValueError("No pixels found with credible level below threshold.")
    if len(unrelated_pixels) == 0:
        raise ValueError("No pixels found with credible level above threshold.")

    def _sample_pixels(pixels, n_sample, weighted):
        if weighted:
            weights = prob[pixels]
            weights = weights / weights.sum()
            return np.random.choice(pixels, size=n_sample, replace=True, p=weights)
        return np.random.choice(pixels, size=n_sample, replace=True)
    
    nside = hp.get_nside(prob)

    r_max = float(marginal_ppf(0.99, prob, distmu, distsigma, distnorm))
    r_grid = np.linspace(0.0, r_max, 512)

    def _sample_distance(ipix: int, label: bool) -> float:
        if label == 'unrelated':
            return float(np.random.choice(r_grid))

        mean = float(distmu[ipix])
        std = float(distsigma[ipix])

        print('Label: '+label)
        print('Mean distance:' + str(mean))
        print('STD distance:' + str(std))

        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            return max(float(np.mean(r_grid)), 1e-6)

        if label == 'candidate':
            mask = np.abs(r_grid - mean) <= 0.1 * std
            candidates = r_grid[mask]
            if len(candidates) == 0:
                distance = mean
            else:
                distance = float(np.random.choice(candidates))
        else:
            mask = np.abs(r_grid - mean) >= 3 * std
            if np.any(mask):
                candidates = r_grid[mask]
                if len(candidates) == 0:
                    raise ValueError(f"No valid distance support found for pixel {ipix}.")
                distance = float(np.random.choice(candidates))
            else:
                distance = mean + 3 * std

        print('Choosen distance:' + str(distance) )
        return max(distance, 1e-6)
    
    def _build_events(sampled_pixels, distance_label: str):
        events = []
        theta, phi = hp.pix2ang(nside, sampled_pixels, nest=nest)
        ra_list = np.rad2deg(phi)
        dec_list = np.rad2deg(0.5 * np.pi - theta)

        for ipix, ra, dec in zip(sampled_pixels, ra_list, dec_list):
            distance = _sample_distance(ipix, label=distance_label)
            distance = np.clip(distance, 1e-3, 1e8)
            redshift = float(z_at_value(cosmo.luminosity_distance, distance * u.Mpc))
            events.append({
                "ra": float(ra),
                "dec": float(dec),
                "redshift": redshift,
            })
        return events

    candidate_pixels = _sample_pixels(candidate_valid_pixels, candidate_num, weighted_valid)
    confuse_pixels = _sample_pixels(valid_pixels, confuse_num, weighted_valid)
    unrelated_pixels_sampled = _sample_pixels(
        unrelated_pixels, unrelated_num, weighted_unrelated
    )

    return {
        "candidate": _build_events(candidate_pixels, distance_label='candidate'),
        "confuse": _build_events(confuse_pixels, distance_label='confuse'),
        "unrelated": _build_events(unrelated_pixels_sampled, distance_label='unrelated'),
    }