#!/usr/bin/env bash
# Fetch mtg-kernel at the pinned commit into mtgkernel/.deps/mtg-kernel (git-ignored) and apply our patches.
#
#   bash mtgkernel/setup.sh                                          # clone from GitHub
#   MK_SRC=/path/to/local/mtg-kernel bash mtgkernel/setup.sh   # from a local clone (fast)
#
# Then: cd mtgkernel && cargo build --release   (toolchain 1.94.1 via rust-toolchain.toml)
#
# Idempotent: an existing clone is reset to the pinned commit and re-patched (local edits in .deps are lost).
# mtg-kernel's build.rs needs a git checkout (it binds the build to HEAD and records clean/dirty); it accepts
# the patched, dirty tree (the binding then says clean = false), so the patch is applied to the worktree
# and not committed.
set -euo pipefail

MK_COMMIT=a4e1474b492f2f14ce1ed9ef8e40309fb70e4e8e
MK_SRC="${MK_SRC:-https://github.com/jackmaiorino/mtg-kernel.git}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEP="$HERE/.deps/mtg-kernel"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi | cut -d' ' -f1; }

mkdir -p "$HERE/.deps"
if [ ! -d "$DEP/.git" ]; then
    echo "cloning mtg-kernel from $MK_SRC"
    git clone --quiet --no-checkout "$MK_SRC" "$DEP"
fi
cd "$DEP"
# already set up (same commit, same patches)? Leave the files alone so cargo does not rebuild mtg-kernel.
want="$(cat "$HERE"/patches/*.patch | sha)"
if [ "$(git rev-parse HEAD 2>/dev/null || true)" = "$MK_COMMIT" ] && [ -f "$HERE/.deps/applied" ] \
    && [ "$(cat "$HERE/.deps/applied")" = "$want $(git diff | sha)" ]; then
    echo "mtg-kernel already at $MK_COMMIT with the current patches"
    exit 0
fi
rm -f "$HERE/.deps/applied"
if ! git cat-file -e "$MK_COMMIT^{commit}" 2>/dev/null; then
    echo "fetching $MK_COMMIT from $MK_SRC"
    git fetch --quiet "$MK_SRC" "$MK_COMMIT" || git fetch --quiet origin
fi
# the pinned commit, detached, with a clean worktree
git checkout --quiet --force --detach "$MK_COMMIT"
git reset --quiet --hard "$MK_COMMIT"
git clean --quiet -fdx -e target
for p in "$HERE"/patches/*.patch; do
    echo "applying $(basename "$p")"
    git apply --check "$p"
    git apply "$p"
done
head="$(git rev-parse HEAD)"
if [ "$head" != "$MK_COMMIT" ]; then
    echo "error: HEAD is $head, expected $MK_COMMIT" >&2
    exit 1
fi
echo "$want $(git diff | sha)" > "$HERE/.deps/applied"
echo "mtg-kernel at $head (patched: $(git diff --stat | tail -1 | sed 's/^ *//'))"
