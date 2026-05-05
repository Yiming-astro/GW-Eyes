import argparse
import os
import requests
from pathlib import Path
from tqdm import tqdm

from recipe.download_gw_data.utils import (
    postprocess_GWTC4_skymap,
    postprocess_GWTC3_skymap,
    postprocess_GWTC2p1_skymap,
    postprocess_GW170817_skymap
)

# DEFAULT_PATH
DEFAULT_SAVE_DIR = Path("GW_Eyes/data/gwpe/tmp")

DEFAULT_URL_LINKS = {
    "GWTC3": {
        "url": "https://zenodo.org/records/8177023",
        "skymap_filename": "IGWN-GWTC3p0-v2-PESkyLocalizations.tar.gz",
    },
    "GWTC4": {
        "url": "https://zenodo.org/records/17014085",
        "skymap_filename": "IGWN-GWTC4p0-1a206db3d_721-Archived_Skymaps.tar.gz",
    },
    "GWTC2p1": {
        "url": "https://zenodo.org/records/6513631",
        "skymap_filename": "IGWN-GWTC2p1-v2-PESkyMaps.tar.gz",
    },
}

GW170817_URL = "https://dcc.ligo.org/public/0146/G1701985/001/LALInference_v2.fits.gz"


def _download_catalog(catalog_label: str) -> None:
    """Download a specific GWTC catalog."""
    print(f'Downloading skymap from {catalog_label}.')
    url, fname = fetch_url(catalog_label)
    download_from_zenodo(url, DEFAULT_SAVE_DIR / fname)


def _download_gw170817() -> None:
    """Download GW170817 skymap."""
    print('Downloading skymap of GW170817')
    fname = Path(GW170817_URL).name
    download_from_links(GW170817_URL, DEFAULT_SAVE_DIR / fname)


# Catalog configuration: maps catalog names to their download and postprocess functions
CATALOG_CONFIG = {
    "gwtc4": {
        "download_func": lambda: _download_catalog("GWTC4"),
        "postprocess_func": postprocess_GWTC4_skymap,
    },
    "gwtc3": {
        "download_func": lambda: _download_catalog("GWTC3"),
        "postprocess_func": postprocess_GWTC3_skymap,
    },
    "gwtc2p1": {
        "download_func": lambda: _download_catalog("GWTC2p1"),
        "postprocess_func": postprocess_GWTC2p1_skymap,
    },
    "gw170817": {
        "download_func": _download_gw170817,
        "postprocess_func": postprocess_GW170817_skymap,
    },
}

ALL_CATALOGS = list(CATALOG_CONFIG.keys())


def fetch_url(catalog_label: str):
    url = DEFAULT_URL_LINKS[catalog_label]["url"]
    skymap_name = DEFAULT_URL_LINKS[catalog_label]["skymap_filename"]
    download_url = f"{url}/files/{skymap_name}?download=1"
    return download_url, skymap_name


def download_from_zenodo(download_url: str, save_path: Path) -> None:

    if save_path.exists() and save_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {save_path}")
        return

    print(f"Downloading: {download_url}")
    try:
        response = requests.get(
            download_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; zenodo-check/1.0)",
            },
            timeout=30,
            stream=True,
        )

        if response.status_code == 200:
            file_size = int(response.headers.get("content-length", 0))

            with open(save_path, "wb") as f:
                if file_size > 0:
                    print(f"Download started ({file_size / (1024**2):.2f} MB)...")
                    with tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=save_path.name,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    print("Download started (unknown size)...")
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            print(f"Download complete: {save_path}")
        else:
            print(f"Download failed, status code: {response.status_code}")

    except Exception as e:
        print("Download failed:", repr(e))

def download_from_links(download_url: str, save_path: Path) -> None:

    if save_path.exists() and save_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {save_path}")
        return

    print(f"Downloading: {download_url}")
    try:
        response = requests.get(
            download_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; zenodo-check/1.0)",
            },
            timeout=30,
            stream=True,
        )

        if response.status_code == 200:
            file_size = int(response.headers.get("content-length", 0))

            with open(save_path, "wb") as f:
                if file_size > 0:
                    print(f"Download started ({file_size / (1024**2):.2f} MB)...")
                    with tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=save_path.name,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    print("Download started (unknown size)...")
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            print(f"Download complete: {save_path}")
        else:
            print(f"Download failed, status code: {response.status_code}")

    except Exception as e:
        print("Download failed:", repr(e))

def download_gw_skymap(catalogs=None) -> None:
    """
    Download GW skymap files from specified catalogs.

    Args:
        catalogs: List of catalog names to download. If None, download all catalogs.
    """
    if catalogs is None:
        catalogs = ALL_CATALOGS

    DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for catalog in catalogs:
        config = CATALOG_CONFIG[catalog]
        config["download_func"]()


def postprocess_gw_skymap(catalogs=None) -> None:
    """
    Postprocess GW skymap files from specified catalogs.

    Args:
        catalogs: List of catalog names to postprocess. If None, postprocess all catalogs.
    """
    if catalogs is None:
        catalogs = ALL_CATALOGS

    for catalog in catalogs:
        config = CATALOG_CONFIG[catalog]
        config["postprocess_func"]()


def main():
    parser = argparse.ArgumentParser(
        description="Download and/or postprocess GWTC skymap data from Zenodo."
    )
    parser.add_argument(
        "--catalogs",
        type=str,
        nargs="+",
        default=ALL_CATALOGS,
        help=f"Catalogs to download and/or postprocess. Choices: {ALL_CATALOGS}. "
             f"Default: all catalogs ({ALL_CATALOGS}). "
             f"Example: --catalogs gwtc4 gw170817",
    )
    parser.add_argument(
        "--only-processing",
        action="store_true",
        help="If set, only run postprocessing without downloading. "
             "Skip download step and only extract/process existing archives.",
    )

    args = parser.parse_args()

    # Normalize catalog names to lowercase
    catalogs = [c.lower() for c in args.catalogs]

    # Validate catalog names
    for catalog in catalogs:
        if catalog not in ALL_CATALOGS:
            print(f"Error: Unknown catalog '{catalog}'. Valid choices: {ALL_CATALOGS}")
            return

    if not args.only_processing:
        download_gw_skymap(catalogs=catalogs)

    postprocess_gw_skymap(catalogs=catalogs)


if __name__ == "__main__":
    main()