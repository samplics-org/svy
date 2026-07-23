#!/usr/bin/env bash
# Usage: bash audit_type_ignores.sh [--fix]
set -euo pipefail

SRC="./src/svy"
FIX=${1:-""}

echo "=== All type: ignore comments ==="
grep -rn "# type: ignore" "$SRC" | sort

echo ""
echo "Total: $(grep -rc '# type: ignore' "$SRC" | awk -F: '$2>0{sum+=$2} END{print sum}')"

if [[ "$FIX" != "--fix" ]]; then
    echo ""
    echo "Re-run with --fix to remove them all, then run: uv run ty check $SRC"
    exit 0
fi

echo ""
echo "=== Removing all type: ignore comments ==="
# Strip both blanket and bracketed forms, preserving the rest of the line
find "$SRC" -name "*.py" -exec sed -i \
    -e 's/  # type: ignore\[[^]]*\]//g' \
    -e 's/  # type: ignore//g' \
    {} +

echo "Done. Now run: uv run ty check $SRC"
echo "Any new errors = comments that were load-bearing and need a proper fix instead."
