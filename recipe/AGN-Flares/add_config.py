import argparse
from pathlib import Path

def append_csv_path(catalog):
    """
    Append the CSV path to config.yaml em_csv_paths if not already present.
    Preserves comments and formatting by only modifying the specific lines.
    """
    # Map catalog to file
    catalog_map = {
        "AGNFRC": "AGNFRC-GWEyes.csv",
        "AGNFCC": "AGNFCC-GWEyes.csv"
    }

    if catalog not in catalog_map:
        print(f"Unknown catalog: {catalog}. Must be AGNFRC or AGNFCC.")
        return

    csv_filename = catalog_map[catalog]
    # Path to add to yaml (relative to GW-Eyes-local)
    relative_path = f"recipe/AGN-Flares/data/{csv_filename}"

    config_file = Path("../../GW_Eyes/src/config/config.yaml")

    # Ensure config file exists
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if not config_file.exists():
        # Initialize with default structure if config doesn't exist
        config_file.write_text("""# EM CSV Configuration

# Paths to electromagnetic CSV files (can be a single path or list of paths)
em_csv_paths:
  - "GW_Eyes/data/sne/SNE.csv"

# GW index file path
gw_index_file: "GW_Eyes/data/gwpe/index.jsonl"

# Output cache directory
output_path: "GW_Eyes/cache"
""")

    # Read existing content
    with open(config_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Check if path already exists and find em_csv_paths section
    path_exists = False
    em_csv_found = False
    last_csv_path_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("em_csv_paths:"):
            em_csv_found = True
            continue
        # Check if this line is a list item under em_csv_paths
        if em_csv_found and stripped.startswith("- "):
            # Extract the path value (remove "- " and quotes)
            path_value = stripped[2:].strip().strip('"').strip("'")
            if path_value == relative_path:
                path_exists = True
                break
            # Track the last list item index
            last_csv_path_idx = i
        # If we hit another top-level key (no leading spaces and contains :), stop tracking
        elif em_csv_found and not stripped.startswith("#") and stripped and not stripped.startswith(" ") and ":" in stripped:
            break

    # Append if not exists
    if path_exists:
        print(f"{relative_path} already exists in {config_file}, skipping.")
    else:
        # Find where to insert (after the last em_csv_paths item, or after em_csv_paths: if empty)
        insert_idx = last_csv_path_idx + 1 if last_csv_path_idx >= 0 else None

        if insert_idx is not None:
            # Insert new path as list item with proper indentation (2 spaces)
            new_line = f'  - "{relative_path}"\n'
            lines.insert(insert_idx, new_line)
        else:
            # em_csv_paths section not found, add it after the first line or at the beginning
            # Find em_csv_paths: line and add after it
            for i, line in enumerate(lines):
                if line.strip().startswith("em_csv_paths:"):
                    lines.insert(i + 1, f'  - "{relative_path}"\n')
                    break

        with open(config_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Appended {relative_path} to {config_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default='AGNFRC', help="AGNFRC or AGNFCC")
    args = parser.parse_args()

    append_csv_path(args.catalog)