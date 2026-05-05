#!/bin/bash

GW_EVENT="GW230529_181500"
WAVEFORM="IMRPhenomNSBH"
OUT_FILE="recipe/eval/eval_data/gw4em/${GW_EVENT}.jsonl"

mkdir -p recipe/eval/eval_data/gw4em

PROMPT="Please search for the electromagnetic counterpart of ${GW_EVENT} using the ${WAVEFORM} waveform based on the provided dataset. Please complete this task as efficiently as possible, and no visualization is needed. If there is a high-confidence electromagnetic counterpart event (as long as all the tests are passed), output Counterpart: <name of the electromagnetic counterpart> on the last line. If not, output Counterpart: None."

python -m GW_Eyes.src.run_agent \
  --mode single \
  --prompt "$PROMPT" \
  --no-debug > "$OUT_FILE" 2>&1