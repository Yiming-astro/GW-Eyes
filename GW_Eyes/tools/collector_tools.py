from __future__ import annotations

import csv
import json
import requests
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Literal, Any, Dict, Optional, List

from GW_Eyes.tools.collector_utils import (
    download_file,
    load_json,
    write_filtered_csv,
    download_gw_skymap_and_postprocess,
    download_json_data,
    write_data_to_csv,
    append_extra_events_to_csv,
    EXTRA_EVENTS,
)
from GW_Eyes.src.config import get_em_csv_paths, get_gw_index_file, get_output_path

# Default index path (adjust if you keep it elsewhere)
DEFAULT_INDEX_PATH = get_gw_index_file()
DEFAULT_CACHE_PATH = get_output_path()
DEFAULT_SNE_PATHS = get_em_csv_paths()
DEFAULT_CIRCULARS_CSV_PATH = Path("GW_Eyes/data/sne/circulars.csv")


mcp = FastMCP("collector_tools")

@mcp.tool()
def download_skymap(
    catalogs: Optional[list[str]] = None,
    only_processing: bool = False
) -> str:
    """
    Download and/or postprocess GWTC (Gravitational-Wave Transient Catalog) skymap data.

    This tool supports flexible download and postprocessing of GW skymap data from
    multiple catalogs (GWTC-4, GWTC-3, GWTC-2.1, GW170817).

    Parameters
    ----------
    catalogs : list[str], optional
        List of catalog names to download/postprocess.
        Choices: ['gwtc4', 'gwtc3', 'gwtc2p1', 'gw170817']
        If None, all catalogs will be processed.
    only_processing : bool, optional
        If True, only run postprocessing without downloading (default False).
        Skip download step and only extract/process existing archives.

    Returns
    -------
    str
        Status message.

    Examples
    --------
    # Download all catalogs
    download_skymap()

    # Only process GWTC-4
    download_skymap(catalogs=['gwtc4'], only_processing=True)

    # Download specific catalogs
    download_skymap(catalogs=['gwtc4', 'gw170817'])
    """
    # Note: use_tqdm=False for MCP tool calls since progress cannot be returned
    download_gw_skymap_and_postprocess(
        catalogs=catalogs,
        only_processing=only_processing,
        use_tqdm=False
    )
    return "Download and postprocessing completed successfully."

@mcp.tool()
def download_osc_event_catalog() -> str:
    """
    Download electromagnetic event catalog data from website.
    It comes from "Open Supernova Catalog"
    https://github.com/astrocatalogs/supernovae

    Returns
    -------
    bool
        Status message.
    """
    CATALOG_URL = "https://raw.githubusercontent.com/astrocatalogs/supernovae/master/output/catalog.json"
    OUT_JSON = Path("GW_Eyes/data/sne/tmp/catalog.json")
    OUT_CSV = Path("GW_Eyes/data/sne/SNE.csv")
    YEAR_CUT = 2015

    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        return "SNE.csv already exists. Please check whether the data has been downloaded before."
    
    download_file(CATALOG_URL, OUT_JSON)
    data = load_json(OUT_JSON)
    kept, skipped_no_date, skipped_old = write_filtered_csv(data, OUT_CSV, YEAR_CUT)
    extra = append_extra_events_to_csv(OUT_CSV, EXTRA_EVENTS)
    return (
        f"Done. kept={kept}, skipped_no_date={skipped_no_date}, skipped_old={skipped_old}"
        f"Saved: {OUT_CSV}"
    )

@mcp.tool()
def fetch_circular_data(circular_id: int) -> Dict[str, Any]:
    """
    Fetch the raw JSON data for a specific GCN circular using its ID.

    Parameters
    ----------
    circular_id : int
        The circular ID to fetch from the URL.

    Returns
    -------
    dict
        The raw JSON data from the circular.
    """
    url = f"https://gcn.nasa.gov/circulars/{circular_id}.json"
    
    try:
        # Fetch the JSON data
        data = download_json_data(url)
        return data
    except Exception as e:
        return {"error": f"Failed to fetch circular data: {str(e)}"}

@mcp.tool()
def write_circular_to_csv(name: str, discoverdate: str = "", maxdate: str = "", ra: str = "", dec: str = "", redshift: str = "", info_source: str = "") -> str:
    """
    Write structured circular data to a CSV file.

    Parameters
    ----------
    name : str
        The name of the event (GRB, Supernova, etc.)
        Typically a combination of letters and numbers (e.g., 'AT2017gfo', 'GRB 260208A'). It should be concise and unique.
    discoverdate : str, optional
        The discovery date of the event
        It should be in the format 'yyyy/mm/dd', for example '2023/05/18'.
        If the JSONL body does not contain a clearly reported date, use the GCN circular's report date as the discovery date.
    maxdate : str, optional
        The maximum date of the event
        It should be in the format 'yyyy/mm/dd', for example '2023/05/18'.
        Leave empty if not available.
    ra : str
        The right ascension of the event
        It should be in the format 'hh:mm:ss.sss', for example '13:09:48.082'.
    dec : str
        The declination of the event
        It should be in the format '±dd:mm:ss.sss', for example '-23:22:53.28'.
    redshift : str
        The redshift of the event
        If no redshift is available, it should be left blank.
    info_source : str, optional
        The source information for tracing, e.g., GCN circular number.
        Format: 'GCN-xxxxx' where xxxxx is the circular ID.

    Returns
    -------
    str
        Status message indicating success or failure
    """
    return write_data_to_csv(name, discoverdate, maxdate, ra, dec, redshift, info_source)


if __name__ == "__main__":
    mcp.run()