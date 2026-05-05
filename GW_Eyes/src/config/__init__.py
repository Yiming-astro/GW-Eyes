"""Configuration module for GW-Eyes."""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

_CONFIG_DIR = Path(__file__).parent
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


def _load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_em_csv_paths() -> List[Path]:
    """
    Load electromagnetic CSV file paths from config.yaml configuration.

    Returns
    -------
    List[Path]
        List of paths to EM CSV files.

    Notes
    -----
    Users can modify GW_Eyes/src/config/config.yaml to add or remove CSV files.
    The em_csv_paths field can be a single string or a list of strings.
    """
    config = _load_config()
    paths = config.get("em_csv_paths", ["GW_Eyes/data/sne/SNE.csv"])

    # Handle both single path (string) and multiple paths (list)
    if isinstance(paths, str):
        paths = [paths]
    elif not isinstance(paths, list):
        paths = ["GW_Eyes/data/sne/SNE.csv"]

    return [Path(p) for p in paths] if paths else [Path("GW_Eyes/data/sne/SNE.csv")]


def get_gw_index_file() -> Path:
    """
    Get the GW index file path from config.yaml configuration.

    Returns
    -------
    Path
        Path to the GW index file. Default is "GW_Eyes/data/gwpe/index.jsonl".
    """
    config = _load_config()
    index_file = config.get("gw_index_file", "GW_Eyes/data/gwpe/index.jsonl")
    return Path(index_file)


def get_output_path() -> Path:
    """
    Get the output cache directory path from config.yaml configuration.

    Returns
    -------
    Path
        Path to the output cache directory. Default is "GW_Eyes/cache".
    """
    config = _load_config()
    output_path = config.get("output_path", "GW_Eyes/cache")
    return Path(output_path)
