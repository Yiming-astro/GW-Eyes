from pathlib import Path

from recipe.download_osc_data.utils import (
    download_file,
    load_json,
    write_filtered_csv,
    append_extra_events,
    EXTRA_EVENTS,
)

CATALOG_URL = "https://raw.githubusercontent.com/astrocatalogs/supernovae/master/output/catalog.json"
OUT_JSON = Path("GW_Eyes/data/sne/tmp/catalog.json")
OUT_CSV = Path("GW_Eyes/data/sne/SNE.csv")
YEAR_CUT = 2015

def main() -> None:
    print("Downloading catalog.json ...")
    download_file(CATALOG_URL, OUT_JSON)
    print(f"Saved: {OUT_JSON}")

    print("Loading JSON ...")
    data = load_json(OUT_JSON)

    print("Writing CSV ...")
    kept, skipped_no_date, skipped_old = write_filtered_csv(data, OUT_CSV, YEAR_CUT)

    print("Appending extra events ...")
    extra = append_extra_events(OUT_CSV, EXTRA_EVENTS)

    print(
        f"Done. kept={kept}, skipped_no_date={skipped_no_date}, skipped_old={skipped_old}, extra={extra}"
    )
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()