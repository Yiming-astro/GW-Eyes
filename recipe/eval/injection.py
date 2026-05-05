from pathlib import Path
import json
from typing import List, Dict
from datetime import datetime, timedelta
import random
import string
import csv
import numpy as np

from recipe.eval.utils import sample_events_from_credible_region

DEFAULT_INDEX_PATH = Path("GW_Eyes/data/gwpe/index.jsonl")
OUTPUT_WITH_ANSWER = Path("recipe/eval/eval_data/em_catalogs/injection_w_answer.csv")
OUTPUT_PUBLIC = Path("recipe/eval/eval_data/em_catalogs/injection.csv")
SUMMARY_TSV_PATH = Path("recipe/eval/eval_data/summary.tsv")

def extract_waveform_from_path(path: str, gw_event: str) -> str:
    """Extract waveform name from skymap file path.

    Example path:
    .../IGWN-GWTC4p0-1a206db3d_721-GW230518_125908-IMRPhenomNSBH:LowSpin_Skymap_PEDataRelease.fits

    Extracts: IMRPhenomNSBH:LowSpin (between '{gw_event}-' and '_Skymap')
    """
    filename = Path(path).stem  # Get filename without extension
    # Find the pattern: {gw_event}-{waveform}_Skymap
    marker = f"{gw_event}-"
    if marker in filename and "_Skymap" in filename:
        after_gw_event = filename.split(marker, 1)[-1]
        waveform = after_gw_event.split("_Skymap")[0]
        return waveform
    return ""

def load_candidate_gw_list(
    candidate_num: int,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> List[Dict[str, str]]:
    if candidate_num <= 0:
        return []

    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    candidate_GW_list: List[Dict[str, str]] = []
    seen_full_names = set()
    blocked_events = {}

    with index_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_num} in {index_path}: {e}"
                ) from e

            full_name = record.get("full_name")
            short_name = record.get("short_name")
            path = record.get("path")

            if full_name is None or short_name is None or path is None:
                continue
            if full_name in blocked_events:
                continue
            if full_name in seen_full_names:
                continue

            seen_full_names.add(full_name)
            candidate_GW_list.append(
                {
                    "short_name": short_name,
                    "full_name": full_name,
                    "path": path,
                }
            )
            if len(candidate_GW_list) >= candidate_num:
                break

    return candidate_GW_list

def generate_injection_candidate(
    candidate_num: int = 1,
    confuse_num: int = 2,
    unrelated_num: int = 10,
    time_before_days: float = 0,
    time_after_days: float = 7,
    cred_2d_threshold: float = 0.5,
    distance_credit_threshold: float = 0.3,
    seed: int = 42,
):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load all GW events from index
    gw_list = load_candidate_gw_list(candidate_num=candidate_num)
    if not gw_list:
        raise ValueError("No GW events loaded from index.")

    # Process all GW events and collect all event catalogs
    all_event_dict_list = []

    for gw in gw_list:
        skymap_path = Path(gw["path"])
        print(skymap_path)
        if not skymap_path.is_file():
            print(f"Warning: Skymap file not found: {skymap_path}, skipping...")
            continue

        # Get the trigger time of GW event
        try:
            year = "20" + gw['short_name'][2:4]
            month = gw['short_name'][4:6]
            day = gw['short_name'][6:8]
            gw_trigger_time_str = f"{year}/{month}/{day}"
            gw_trigger_time = datetime.strptime(gw_trigger_time_str, "%Y/%m/%d")
        except Exception as e:
            print(f"Warning: Failed to parse GW trigger time for {gw['short_name']}: {e}, skipping...")
            continue

        # Extract waveform from skymap path
        waveform = extract_waveform_from_path(gw["path"], gw["full_name"])

        # Generate the event name and time (1 candidate per GW event)
        labels = ["candidate"] * 1 + ["confuse"] * confuse_num + ["unrelated"] * unrelated_num

        start_date = gw_trigger_time - timedelta(days=time_before_days)
        end_date = gw_trigger_time + timedelta(days=time_after_days)
        day_span = (end_date.date() - start_date.date()).days

        event_dict_list = []
        for label in labels:
            offset_days = random.randint(0, day_span)
            discover_date = (start_date + timedelta(days=offset_days)).date()
            suffix = "".join(random.choices(string.ascii_lowercase, k=3))
            name = f"INJ{discover_date.strftime('%y%m%d')}{suffix}"

            event_dict_list.append({
                "name": name,
                "discoverdate": discover_date.isoformat(),
                "maxdate": None,
                "ra": None,
                "dec": None,
                "redshift": None,
                "info_source": None,
                "gw_event": gw["full_name"],
                "waveform": waveform,
                "label": label,
            })

        # Get the RA, Dec, and redshift from credible region sampling
        sampled_events = sample_events_from_credible_region(
            skymap_file_path=str(skymap_path),
            candidate_num=1,
            confuse_num=confuse_num,
            unrelated_num=unrelated_num,
            cred_2d_threshold=cred_2d_threshold,
            distance_credit_threshold=distance_credit_threshold,
        )

        sampled_idx = {"candidate": 0, "confuse": 0, "unrelated": 0}

        for event in event_dict_list:
            label = event["label"]
            sampled = sampled_events[label][sampled_idx[label]]
            sampled_idx[label] += 1

            event["ra"] = sampled["ra"]
            event["dec"] = sampled["dec"]
            event["redshift"] = sampled["redshift"]

        all_event_dict_list.extend(event_dict_list)

    return all_event_dict_list

def save_as_csv(event_dict_list):

    def _format_date(value):
        if value in (None, ""):
            return ""
        if isinstance(value, datetime):
            return value.strftime("%Y/%m/%d")
        try:
            return datetime.fromisoformat(str(value)).strftime("%Y/%m/%d")
        except ValueError:
            return str(value).replace("-", "/")

    def _format_ra_deg_to_hms(ra_deg: float) -> str:
        total_seconds = (ra_deg % 360.0) / 15.0 * 3600.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _format_dec_deg_to_dms(dec_deg: float) -> str:
        sign = "+" if dec_deg >= 0 else "-"
        abs_deg = abs(dec_deg)
        total_seconds = abs_deg * 3600.0
        degrees = int(total_seconds // 3600)
        arcmin = int((total_seconds % 3600) // 60)
        arcsec = total_seconds % 60
        return f"{sign}{degrees:02d}:{arcmin:02d}:{arcsec:06.3f}"

    public_fields = [
        "name",
        "discoverdate",
        "maxdate",
        "ra",
        "dec",
        "redshift",
        "info_source",
    ]
    answer_fields = public_fields + ["gw_event", "waveform", "label"]

    public_rows = []
    answer_rows = []

    for event in event_dict_list:
        row = {
            "name": event.get("name", ""),
            "discoverdate": _format_date(event.get("discoverdate")),
            "maxdate": _format_date(event.get("maxdate")),
            "ra": _format_ra_deg_to_hms(float(event["ra"])) if event.get("ra") is not None else "",
            "dec": _format_dec_deg_to_dms(float(event["dec"])) if event.get("dec") is not None else "",
            "redshift": event.get("redshift", "") if event.get("redshift") is not None else "",
            "info_source": event.get("info_source", "") if event.get("info_source") is not None else "",
        }

        public_rows.append(row)

        answer_row = dict(row)
        answer_row["gw_event"] = event.get("gw_event", "")
        answer_row["waveform"] = event.get("waveform", "")
        answer_row["label"] = event.get("label", "")
        answer_rows.append(answer_row)

    with OUTPUT_PUBLIC.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fields)
        writer.writeheader()
        writer.writerows(public_rows)

    with OUTPUT_WITH_ANSWER.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=answer_fields)
        writer.writeheader()
        writer.writerows(answer_rows)

    # Generate summary.tsv for test.sh: GW_EVENT <tab> waveform <tab> counterpart
    SUMMARY_TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_TSV_PATH.open("w", encoding="utf-8", newline="") as f:
        f.write("GW_EVENT\twaveform\tcounterpart\n")
        for event in event_dict_list:
            if event.get("label") == "candidate":
                f.write(f"{event.get('gw_event', '')}\t{event.get('waveform', '')}\t{event.get('name', '')}\n")

    return {
        "public_path": str(OUTPUT_PUBLIC),
        "answer_path": str(OUTPUT_WITH_ANSWER),
        "summary_path": str(SUMMARY_TSV_PATH),
    }


if __name__ == "__main__":
    result = generate_injection_candidate(
        candidate_num=50,  # Number of GW events to process
        confuse_num=2,
        unrelated_num=10,
        time_before_days= 0,
        time_after_days= 7,
        cred_2d_threshold=0.5,
        distance_credit_threshold=0.3,
    )
    save_result = save_as_csv(result)
    print(f"Saved {len(result)} events to:")
    print(f"  Public: {save_result['public_path']}")
    print(f"  With answer: {save_result['answer_path']}")
    print(f"  Summary TSV: {save_result['summary_path']}")