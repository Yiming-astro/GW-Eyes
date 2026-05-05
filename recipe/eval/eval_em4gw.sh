#!/bin/bash

EVENT_FILE="recipe/eval/eval_data/summary_em4gw.tsv"
OUT_DIR="recipe/eval/eval_data/em4gw"
LIMIT=10   # 0 means no limit

mkdir -p "$OUT_DIR"

count=0

while IFS=$'\t' read -r EM_EVENT GW_EVENT; do
  [[ -z "$EM_EVENT" ]] && continue

  if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
    break
  fi

  OUT_FILE="${OUT_DIR}/${EM_EVENT}.jsonl"

  # Skip if already exists
  if [[ -f "$OUT_FILE" ]]; then
    echo "Skipping ${EM_EVENT} (already exists)"
    continue
  fi

  PROMPT="Please search for the gravitational-wave counterpart of ${EM_EVENT} based on the provided dataset. Please complete this task as efficiently as possible, and no visualization is needed. If there is a high-confidence gravitational-wave counterpart event (as long as all the tests are passed), output Counterpart: <name of the gravitational-wave full name> on the last line. If not, output Counterpart: None."

  echo "Running ${EM_EVENT} (gold: ${GW_EVENT}) ..."

  python -m GW_Eyes.src.run_agent \
    --mode single \
    --prompt "$PROMPT" \
    --no-debug > "$OUT_FILE" 2>&1

  count=$((count + 1))
done < <(tail -n +2 "$EVENT_FILE")

echo "Finished ${count} run(s)."