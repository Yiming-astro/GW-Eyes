import gzip
import json
import shutil
import tarfile
from GW_Eyes.src.config import get_gw_index_file, get_output_path

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
    _append_global_index(DEFAULT_INDEX_PATH, records)
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
    _append_global_index(DEFAULT_INDEX_PATH, records)
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

    _append_global_index(DEFAULT_INDEX_PATH, [record])
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
