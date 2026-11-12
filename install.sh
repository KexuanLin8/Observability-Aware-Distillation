#!/usr/bin/env bash
# Official HAIC at the commit we used + our overlay and patch, in ./HAIC.
#   bash install.sh [target_dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-HAIC}"
COMMIT=e262500
if [ ! -d "$DST/.git" ]; then
  git clone https://github.com/ldt29/HAIC.git "$DST"
fi
cd "$DST"
git checkout "$COMMIT"
git lfs pull || echo "WARNING: git lfs pull failed; the USD assets must be real files, not LFS pointers"
git apply --whitespace=nowarn "$HERE/patches/haic-$COMMIT.patch"
cp -r "$HERE/overlay/." .
echo "HAIC $COMMIT + overlay ready in $(pwd)"
