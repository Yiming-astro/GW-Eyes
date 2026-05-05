#!/bin/bash

EVENT_FILE="recipe/eval/eval_data/summary_gw4em.tsv"
OUT_DIR="recipe/eval/eval_data/gw4em"
LIMIT=10   # 0 means no limit

mkdir -p "$OUT_DIR"

count=0

while IFS=$'\t' read -r GW_EVENT WAVEFORM GOLD_COUNTERPART; do
  [[ -z "$GW_EVENT" ]] && continue

  if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
    break
  fi

  OUT_FILE="${OUT_DIR}/${GW_EVENT}.jsonl"

  # Skip if already exists
  if [[ -f "$OUT_FILE" ]]; then
    echo "Skipping ${GW_EVENT} (already exists)"
    continue
  fi

  PROMPT="Please search for the electromagnetic counterpart of ${GW_EVENT} using the ${WAVEFORM} waveform based on the provided dataset. Please complete this task as efficiently as possible, and no visualization is needed. If there is a high-confidence electromagnetic counterpart event (as long as all the tests are passed), output Counterpart: <name of the electromagnetic counterpart> on the last line. If not, output Counterpart: None."

  echo "Running ${GW_EVENT} (waveform: ${WAVEFORM}, gold: ${GOLD_COUNTERPART}) ..."

  python -m GW_Eyes.src.run_agent \
    --mode single \
    --prompt "$PROMPT" \
    --no-debug > "$OUT_FILE" 2>&1

  count=$((count + 1))
done < <(tail -n +2 "$EVENT_FILE")

echo "Finished ${count} run(s)."