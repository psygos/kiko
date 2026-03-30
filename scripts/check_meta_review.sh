#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REVIEW_DIR="$ROOT_DIR/reviews/meta"

required_headings=(
  "# Meta Review"
  "## Commit Goal"
  "## Current Repo Starting Point"
  "## Previous Invariants"
  "## New Invariants Claimed"
  "## Touched Files"
  "## New Or Changed Metrics"
  "## New Or Changed Solver Outputs"
  "## Tests Added"
  "## Tests Run"
  "## Known Risks Or Deferred Follow-Ups"
  "## Findings"
  "## Invariant Verdict"
  "## Metric Verdict"
  "## Test Verdict"
  "## Merge Decision"
)

review_files=()
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    review_files+=("$arg")
  done
else
  while IFS= read -r -d '' file; do
    review_files+=("$file")
  done < <(find "$REVIEW_DIR" -maxdepth 1 -type f -name '*.md' ! -name 'README.md' ! -name 'TEMPLATE.md' -print0 | sort -z)
fi

if [ "${#review_files[@]}" -eq 0 ]; then
  echo "no review artifacts found in $REVIEW_DIR" >&2
  exit 1
fi

status=0
for file in "${review_files[@]}"; do
  base=$(basename "$file")
  if [[ ! "$base" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}-[A-Za-z0-9._-]+(-v[0-9]+)?\.md$ ]]; then
    echo "invalid review artifact filename: $base" >&2
    status=1
    continue
  fi

  for heading in "${required_headings[@]}"; do
    if ! grep -Fxq "$heading" "$file"; then
      echo "missing heading '$heading' in $file" >&2
      status=1
    fi
  done

done

exit "$status"
