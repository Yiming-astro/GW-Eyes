import re
import csv
import requests
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from typing import Literal, Any, Dict, Optional, List, Iterator, Tuple


from GW_Eyes.src.config import get_gw_index_file, get_output_path


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

# DEFAULT_PATH
DEFAULT_INDEX_PATH = get_gw_index_file()
DEFAULT_CACHE_PATH = get_output_path()
DEFAULT_GWTC4_RAW_PATH = "GW_Eyes/data/gwpe/tmp/IGWN-GWTC4p0-1a206db3d_721-Archived_Skymaps.tar.gz"
DEFAULT_GWTC4_TARGET_PATH = "GW_Eyes/data/gwpe/GWTC-4"
DEFAULT_GWTC3_RAW_PATH = Path("GW_Eyes/data/gwpe/tmp/IGWN-GWTC3p0-v2-PESkyLocalizations.tar.gz")
DEFAULT_GWTC3_TARGET_PATH = Path("GW_Eyes/data/gwpe/GWTC-3")
DEFAULT_GWTC2P1_RAW_PATH = Path("GW_Eyes/data/gwpe/tmp/IGWN-GWTC2p1-v2-PESkyMaps.tar.gz")
DEFAULT_GWTC2P1_TARGET_PATH = Path("GW_Eyes/data/gwpe/GWTC-2p1")
DEFAULT_GW170817_RAW_PATH = Path("GW_Eyes/data/gwpe/tmp/LALInference_v2.fits.gz")
DEFAULT_GW170817_TARGET_PATH = Path("GW_Eyes/data/gwpe/GW170817")

DEFAULT_CIRCULARS_CSV_PATH = Path("GW_Eyes/data/sne/circulars.csv")

# Extra event to append after OSC catalog download completes
EXTRA_EVENTS = [
    {
        "name": "AT2017gfo",
        "discoverdate": "2017/08/17",
        "maxdate": "",
        "ra": "13:09:48.082",
        "dec": "-23:22:53.28",
        "redshift": "0.009783",
    }
]


# ============== Skymap Postprocessing Functions ==============

def postprocess_GWTC4_skymap() -> bool:
    """
    Extract/unzip/postprocess the GWTC-4 archived skymaps tarball into GW_Eyes/data/gwpe/GWTC-4,
    """
    root_tar = Path(DEFAULT_GWTC4_RAW_PATH)
    target_dir = Path(DEFAULT_GWTC4_TARGET_PATH)
    skymap_dir = target_dir / "parameter_estimation" / "skymaps"
    global_index_path = Path(DEFAULT_INDEX_PATH)

    def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
        base = path.resolve()
        for member in tar.getmembers():
            member_path = (path / member.name).resolve()
            if base not in member_path.parents and member_path != base:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(path=path)

    if not root_tar.is_file():
        raise FileNotFoundError(f"Missing archive: {root_tar}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(root_tar, "r:gz") as tar:
        _safe_extract(tar, target_dir)

    if not skymap_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {skymap_dir}")

    # Decompress *.fits.gz -> *.fits (atomic replace), then delete gz
    for gz_path in sorted(skymap_dir.glob("*.fits.gz")):
        fits_path = gz_path.with_suffix("")
        tmp_path = fits_path.with_suffix(fits_path.suffix + ".tmp")

        with gzip.open(gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        tmp_path.replace(fits_path)
        gz_path.unlink()

    # Build per-directory index and also update global index
    records = _build_GWTC4_skymap_index(skymap_dir, index_name="index.jsonl")
    _append_global_index(global_index_path, records)

    return True


def _build_GWTC4_skymap_index(directory: Path, index_name: str = "index.jsonl") -> list[dict]:
    """
    Build GWTC-4 skymap index from *.fits in the given directory and write it to directory/index.jsonl.

    Returns:
        records (list of dict)
    """
    base_path = Path(directory)
    index_path = base_path / index_name

    records: list[dict] = []

    suffix = "_Skymap_PEDataRelease.fits"

    for fits_file in sorted(base_path.glob("*.fits")):
        name = fits_file.name
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

    with open(index_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return records


def postprocess_GWTC3_skymap() -> bool:
    if not DEFAULT_GWTC3_RAW_PATH.is_file():
        raise FileNotFoundError(f"Missing archive: {DEFAULT_GWTC3_RAW_PATH}")

    DEFAULT_GWTC3_TARGET_PATH.mkdir(parents=True, exist_ok=True)

    def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
        base = path.resolve()
        for m in tar.getmembers():
            p = (path / m.name).resolve()
            if base not in p.parents and p != base:
                raise RuntimeError(f"Unsafe path in tar archive: {m.name}")
        tar.extractall(path=path)

    with tarfile.open(DEFAULT_GWTC3_RAW_PATH, "r:gz") as tar:
        _safe_extract(tar, DEFAULT_GWTC3_TARGET_PATH)

    records = _build_GWTC3_skymap_index(DEFAULT_GWTC3_TARGET_PATH, index_name="index.jsonl")
    _append_global_index(Path("GW_Eyes/data/gwpe/index.jsonl"), records)
    return True


def _build_GWTC3_skymap_index(directory: Path, index_name: str = "index.jsonl") -> list[dict]:
    """
    Build GWTC-3 skymap index from *.fits in the given directory and write it to directory/index.jsonl.
    """
    base_path = Path(directory)
    index_path = base_path / index_name

    records: list[dict] = []

    suffix = ".fits"

    for fits_file in sorted(base_path.rglob("*.fits")):
        name = fits_file.name
        if not name.endswith(suffix) or ":" not in name or "-GW" not in name:
            continue

        stem = name[: -len(suffix)]
        left, waveform = stem.rsplit(":", 1)

        catalog_left, gw_rest = left.rsplit("-GW", 1)
        catalog_label = catalog_left
        full_blob = "GW" + gw_rest

        full_name = full_blob.split("_PEDataRelease", 1)[0].split("_PEDDataRelease", 1)[0]
        short_name = full_name.split("_", 1)[0]

        record = {
            "catalog_label": catalog_label,
            "short_name": short_name,
            "full_name": full_name,
            "waveform": waveform,
            "path": str(fits_file),
        }
        records.append(record)

    with open(index_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return records


def postprocess_GWTC2p1_skymap() -> bool:
    if not DEFAULT_GWTC2P1_RAW_PATH.is_file():
        raise FileNotFoundError(f"Missing archive: {DEFAULT_GWTC2P1_RAW_PATH}")

    DEFAULT_GWTC2P1_TARGET_PATH.mkdir(parents=True, exist_ok=True)

    def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
        base = path.resolve()
        for m in tar.getmembers():
            p = (path / m.name).resolve()
            if base not in p.parents and p != base:
                raise RuntimeError(f"Unsafe path in tar archive: {m.name}")
        tar.extractall(path=path)

    with tarfile.open(DEFAULT_GWTC2P1_RAW_PATH, "r:gz") as tar:
        _safe_extract(tar, DEFAULT_GWTC2P1_TARGET_PATH)

    records = _build_GWTC2p1_skymap_index(DEFAULT_GWTC2P1_TARGET_PATH, index_name="index.jsonl")
    _append_global_index(Path("GW_Eyes/data/gwpe/index.jsonl"), records)
    return True


def _build_GWTC2p1_skymap_index(directory: Path, index_name: str = "index.jsonl") -> list[dict]:
    """
    Build GWTC-2.1 skymap index from *.fits (layout matches GWTC-3 style).
    """
    base_path = Path(directory)
    index_path = base_path / index_name

    records: list[dict] = []
    suffix = ".fits"

    for fits_file in sorted(base_path.rglob("*.fits")):
        name = fits_file.name
        if not name.endswith(suffix) or ":" not in name or "-GW" not in name:
            continue

        stem = name[: -len(suffix)]
        left, waveform = stem.rsplit(":", 1)

        catalog_left, gw_rest = left.rsplit("-GW", 1)
        catalog_label = catalog_left
        full_blob = "GW" + gw_rest

        full_name = full_blob.split("_PEDataRelease", 1)[0].split("_PEDDataRelease", 1)[0]
        short_name = full_name.split("_", 1)[0]

        record = {
            "catalog_label": catalog_label,
            "short_name": short_name,
            "full_name": full_name,
            "waveform": waveform,
            "path": str(fits_file),
        }
        records.append(record)

    with open(index_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return records


def postprocess_GW170817_skymap() -> bool:
    """
    Decompress GW170817 skymap fits.gz into GW_Eyes/data/gwpe/GW170817,
    then append a single record into GW_Eyes/data/gwpe/index.jsonl.
    """
    if not DEFAULT_GW170817_RAW_PATH.is_file():
        raise FileNotFoundError(f"Missing file: {DEFAULT_GW170817_RAW_PATH}")

    DEFAULT_GW170817_TARGET_PATH.mkdir(parents=True, exist_ok=True)

    fits_path = DEFAULT_GW170817_TARGET_PATH / DEFAULT_GW170817_RAW_PATH.with_suffix("").name
    tmp_path = fits_path.with_suffix(fits_path.suffix + ".tmp")

    with gzip.open(DEFAULT_GW170817_RAW_PATH, "rb") as f_in, open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    tmp_path.replace(fits_path)

    record = {
        "catalog_label": "GW170817",
        "short_name": "GW170817",
        "full_name": "GW170817",
        "waveform": "",
        "path": str(fits_path),
    }

    _append_global_index(Path("GW_Eyes/data/gwpe/index.jsonl"), [record])
    return True


def _append_global_index(global_index_path: Path, new_records: list[dict]) -> None:
    """
    Append new skymap records into GW_Eyes/data/gwpe/index.jsonl.
    Skip records that already exist (exact duplicate check on all fields).
    """
    global_index_path = Path(global_index_path)
    global_index_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing records for deduplication
    existing_records = []
    if global_index_path.exists():
        with open(global_index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))

    # Check for duplicates and append only new records
    with open(global_index_path, "a", encoding="utf-8") as f:
        for r in new_records:
            if r not in existing_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                existing_records.append(r)


# ============== Download Functions ==============

def fetch_url(catalog_label: str):
    url = DEFAULT_URL_LINKS[catalog_label]["url"]
    skymap_name = DEFAULT_URL_LINKS[catalog_label]["skymap_filename"]
    download_url = f"{url}/files/{skymap_name}?download=1"
    return download_url, skymap_name


def download_from_zenodo(download_url: str, save_path: Path, use_tqdm: bool = True) -> None:
    """
    Download a file from Zenodo.

    Parameters
    ----------
    download_url : str
        The URL to download from
    save_path : Path
        The path to save the file to
    use_tqdm : bool, optional
        Whether to show progress bar (default True). Set to False for MCP tool calls.
    """

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
    except Exception as e:
        raise RuntimeError(f"Request failed for {download_url}: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Zenodo download failed: {download_url} (status {response.status_code})"
        )

    file_size = int(response.headers.get("content-length", 0))
    if file_size > 0:
        print(f"Download started ({file_size / (1024**2):.2f} MB)...")
    else:
        print("Download started (unknown size)...")

    with open(save_path, "wb") as f:
        if file_size > 0 and use_tqdm:
            from tqdm import tqdm
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
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Download complete: {save_path}")


def download_from_links(download_url: str, save_path: Path, use_tqdm: bool = True) -> None:
    """
    Download a file from a direct link.

    Parameters
    ----------
    download_url : str
        The URL to download from
    save_path : Path
        The path to save the file to
    use_tqdm : bool, optional
        Whether to show progress bar (default True). Set to False for MCP tool calls.
    """

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
    except Exception as e:
        raise RuntimeError(f"Request failed for {download_url}: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Direct download failed: {download_url} (status {response.status_code})"
        )

    file_size = int(response.headers.get("content-length", 0))
    if file_size > 0:
        print(f"Download started ({file_size / (1024**2):.2f} MB)...")
    else:
        print("Download started (unknown size)...")

    with open(save_path, "wb") as f:
        if file_size > 0 and use_tqdm:
            from tqdm import tqdm
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
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Download complete: {save_path}")


def _download_catalog(catalog_label: str, use_tqdm: bool = True) -> None:
    """Download a specific GWTC catalog."""
    print(f'Downloading skymap from {catalog_label}.')
    url, fname = fetch_url(catalog_label)
    download_from_zenodo(url, DEFAULT_SAVE_DIR / fname, use_tqdm=use_tqdm)


def _download_gw170817(use_tqdm: bool = True) -> None:
    """Download GW170817 skymap."""
    print('Downloading skymap of GW170817')
    fname = Path(GW170817_URL).name
    download_from_links(GW170817_URL, DEFAULT_SAVE_DIR / fname, use_tqdm=use_tqdm)


# Catalog configuration: maps catalog names to their download and postprocess functions
CATALOG_CONFIG = {
    "gwtc4": {
        "download_func": lambda use_tqdm=True: _download_catalog("GWTC4", use_tqdm=use_tqdm),
        "postprocess_func": postprocess_GWTC4_skymap,
    },
    "gwtc3": {
        "download_func": lambda use_tqdm=True: _download_catalog("GWTC3", use_tqdm=use_tqdm),
        "postprocess_func": postprocess_GWTC3_skymap,
    },
    "gwtc2p1": {
        "download_func": lambda use_tqdm=True: _download_catalog("GWTC2p1", use_tqdm=use_tqdm),
        "postprocess_func": postprocess_GWTC2p1_skymap,
    },
    "gw170817": {
        "download_func": lambda use_tqdm=True: _download_gw170817(use_tqdm=use_tqdm),
        "postprocess_func": postprocess_GW170817_skymap,
    },
}

ALL_CATALOGS = list(CATALOG_CONFIG.keys())


def download_gw_skymap(catalogs=None, use_tqdm: bool = True) -> None:
    """
    Download GW skymap files from specified catalogs.

    Args:
        catalogs: List of catalog names to download. If None, download all catalogs.
        use_tqdm: Whether to show progress bar (default True). Set to False for MCP tool calls.
    """
    if catalogs is None:
        catalogs = ALL_CATALOGS

    DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for catalog in catalogs:
        config = CATALOG_CONFIG[catalog]
        config["download_func"](use_tqdm=use_tqdm)


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


def download_gw_skymap_and_postprocess(
    catalogs=None,
    only_processing: bool = False,
    use_tqdm: bool = True
) -> bool:
    """
    Download and/or postprocess GW skymap files.

    Parameters
    ----------
    catalogs : list, optional
        List of catalog names to download/postprocess. If None, process all catalogs.
        Choices: ['gwtc4', 'gwtc3', 'gwtc2p1', 'gw170817']
    only_processing : bool, optional
        If True, only run postprocessing without downloading (default False).
        Skip download step and only extract/process existing archives.
    use_tqdm : bool, optional
        Whether to show progress bar (default True). Set to False for MCP tool calls.

    Returns
    -------
    bool
        True if successful
    """
    if catalogs is None:
        catalogs = ALL_CATALOGS

    if not only_processing:
        download_gw_skymap(catalogs=catalogs, use_tqdm=use_tqdm)

    postprocess_gw_skymap(catalogs=catalogs)

    return True


# ============== OSC Catalog Download Functions ==============

DATE_RE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")


def download_file(url: str, out_path: str, timeout: int = 120) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_first_value(obj: Any) -> Optional[str]:
    if obj is None:
        return None

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "value" in item:
                v = item.get("value")
                if v not in ("", None):
                    return str(v).strip()
            elif isinstance(item, str) and item.strip():
                return item.strip()
        return None

    if isinstance(obj, dict):
        v = obj.get("value")
        return str(v).strip() if v not in ("", None) else None

    if isinstance(obj, (str, int, float)):
        s = str(obj).strip()
        return s if s else None

    return None


def normalize_date_yyyymmdd(raw: Any) -> Optional[str]:
    if not raw:
        return None
    m = DATE_RE.search(str(raw))
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{int(y):04d}/{int(mo):02d}/{int(d):02d}"


def get_year(date_yyyymmdd: Optional[str]) -> Optional[int]:
    if not date_yyyymmdd:
        return None
    try:
        return int(date_yyyymmdd.split("/")[0])
    except Exception:
        return None


def iterate_events(data: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if isinstance(data, list):
        for ev in data:
            if not isinstance(ev, dict):
                continue
            name = ev.get("name")
            if not name:
                alias = ev.get("alias")
                if isinstance(alias, list) and alias:
                    name = alias[0]
            if name:
                yield name, ev

    elif isinstance(data, dict):
        for name, ev in data.items():
            if isinstance(ev, dict) and name:
                yield name, ev


def write_filtered_csv(
    data: Any,
    out_csv: str,
    year_cut: int,
) -> Tuple[int, int, int]:

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["name", "discoverdate", "maxdate", "ra", "dec", "redshift"]

    kept = 0
    skipped_no_date = 0
    skipped_old = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, ev in iterate_events(data):
            discover_raw = pick_first_value(ev.get("discoverdate")) or pick_first_value(
                ev.get("discoverydate")
            )
            max_raw = pick_first_value(ev.get("maxdate"))

            discover = normalize_date_yyyymmdd(discover_raw)
            maxdate = normalize_date_yyyymmdd(max_raw)

            if not discover and not maxdate:
                skipped_no_date += 1
                continue

            y_discover = get_year(discover)
            y_max = get_year(maxdate)

            valid_year = False
            if y_discover is not None and y_discover >= year_cut:
                valid_year = True
            if y_max is not None and y_max >= year_cut:
                valid_year = True

            if not valid_year:
                skipped_old += 1
                continue

            ra = pick_first_value(ev.get("ra")) or ""
            dec = pick_first_value(ev.get("dec")) or ""
            redshift = pick_first_value(ev.get("redshift")) or ""

            writer.writerow(
                {
                    "name": name,
                    "discoverdate": discover or "",
                    "maxdate": maxdate or "",
                    "ra": ra,
                    "dec": dec,
                    "redshift": redshift,
                }
            )
            kept += 1

    return kept, skipped_no_date, skipped_old


def append_extra_events_to_csv(out_csv: str, extra_events: list[dict]) -> int:
    """Append extra events to an existing CSV file.

    Parameters
    ----------
    out_csv : str
        Path to the CSV file.
    extra_events : list[dict]
        List of extra events to append.

    Returns
    -------
    int
        Number of events appended.
    """
    out_csv = Path(out_csv)
    if not out_csv.exists():
        return 0

    fieldnames = ["name", "discoverdate", "maxdate", "ra", "dec", "redshift"]
    appended = 0

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for ev in extra_events:
            row = {k: ev.get(k, "") for k in fieldnames}
            writer.writerow(row)
            appended += 1

    return appended


def download_json_data(url: str) -> Dict[str, Any]:
    """Helper function to download and load JSON data from a URL."""
    response = requests.get(url)
    response.raise_for_status()  # Will raise an exception for non-200 responses
    return response.json()


def write_data_to_csv(name: str, discoverdate: str = "", maxdate: str = "", ra: str = "", dec: str = "", redshift: str = "", info_source: str = "") -> str:
    """Helper function to write structured data to a CSV file."""
    try:
        # Prepare data to write
        data = {
            "name": name,
            "discoverdate": discoverdate,
            "maxdate": maxdate,
            "ra": ra,
            "dec": dec,
            "redshift": redshift,
            "info_source": info_source
        }

        # Open the file and write to it
        with open(DEFAULT_CIRCULARS_CSV_PATH, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["name", "discoverdate", "maxdate", "ra", "dec", "redshift", "info_source"])
            # Write the header if the file is empty
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        return f"Successfully written data to {DEFAULT_CIRCULARS_CSV_PATH}"

    except Exception as e:
        return f"Error writing to CSV: {str(e)}"
