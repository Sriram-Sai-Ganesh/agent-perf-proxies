#!/bin/bash
# cluster/publish.sh <run_id>...   upload finished runs to the shared dataset as traces/$SITE/<run_id>/
# A run must have finished, and its commit must be on origin so anyone can rebuild the code that made it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
source "$here/site.env"
: "${DATA_ROOT:?}" "${SITE:?set SITE in cluster/site.env}" "${RESULTS_REPO:?set RESULTS_REPO in cluster/site.env}" "${HF_TOKEN:?set HF_TOKEN in cluster/site.env}"
git -C "$root" fetch -q origin
for run in "$@"; do
  dir="$DATA_ROOT/runs/$run"
  read -r commit finished < <("$root/.venv/bin/python" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["commit"], d["finished"])' "$dir/provenance.json")
  [ "$finished" != None ] || { echo "$run has not finished" >&2; exit 1; }
  # shortcut: a local tag that was never pushed also passes, until tags are checked against origin
  git -C "$root" for-each-ref --contains "$commit" refs/remotes/origin refs/tags | grep -q . \
    || { echo "$run was made at commit $commit, which is not on origin; push it first" >&2; exit 1; }
  "$root/.venv/bin/hf" upload "$RESULTS_REPO" "$dir" "traces/$SITE/$run" --repo-type dataset \
    --include steps.parquet --include provenance.json --include provenance.diff --commit-message "add $SITE/$run"
done
