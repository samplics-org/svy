#!/usr/bin/env bash
# Run from packages/svy-io root
# Preview first with: bash cleanup_svy_io.sh --dry-run
# Apply with:         bash cleanup_svy_io.sh

set -euo pipefail

# Guard: this script rm -rf's relative paths, so refuse to run anywhere
# other than the svy-io package root.
if [[ ! -d python/svy_io || ! -d native/svyreadstat_rs ]]; then
    echo "error: run this from packages/svy-io (python/svy_io not found in CWD)" >&2
    exit 1
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

remove() {
    if $DRY_RUN; then
        echo "  [dry-run] would remove: $1"
    else
        rm -rf "$1" && echo "  removed: $1"
    fi
}

echo "=== svy-io cleanup ==="
echo ""

echo "--- compiled artifacts in python/ ---"
find python -name "*.so" -o -name "*.pyd" | while read f; do remove "$f"; done

echo ""
echo "--- dist/ wheels ---"
remove dist

echo ""
echo "--- __pycache__ ---"
find . -type d -name "__pycache__" | while read d; do remove "$d"; done

echo ""
echo "--- pytest cache ---"
find . -type d -name ".pytest_cache" | while read d; do remove "$d"; done

echo ""
echo "--- Rust build artifacts ---"
remove native/readstat-sys/target
remove native/svyreadstat_rs/target

echo ""
echo "--- ReadStat autotools build artifacts ---"
remove ReadStat/autom4te.cache
remove ReadStat/config.log
remove ReadStat/config.status
remove ReadStat/config.sub
remove ReadStat/config.guess
remove ReadStat/config.rpath
remove ReadStat/libtool
remove ReadStat/ltmain.sh
remove ReadStat/libreadstat.la
find ReadStat -name "*.lo" -o -name "*.la" -o -name "*.o" | while read f; do remove "$f"; done
find ReadStat -name "Makefile" -not -name "Makefile.am" -not -name "Makefile.in" | while read f; do remove "$f"; done

echo ""
echo "=== done ==="
