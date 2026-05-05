import csv
import json
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from pathlib import Path
import requests

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


# Extra event to append after all events are written
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


def append_extra_events(out_csv: str, events: List[Dict[str, Any]]) -> int:
    """Append extra events to an existing CSV file.

    Parameters
    ----------
    out_csv : str
        Path to the CSV file.
    events : List[Dict[str, Any]]
        List of event dictionaries with keys: name, discoverdate, maxdate, ra, dec, redshift.

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
        for ev in events:
            writer.writerow({k: ev.get(k, "") for k in fieldnames})
            appended += 1

    return appended