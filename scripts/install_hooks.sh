#!/bin/bash
# Install the repo's git hooks into .git/hooks/.
# Idempotent: re-running overwrites the installed hooks.
set -e

repo_root=$(git rev-parse --show-toplevel)
src="$repo_root/scripts/hooks"
dst="$repo_root/.git/hooks"

if [[ ! -d "$src" ]]; then
    echo "no hooks dir at $src" >&2
    exit 1
fi

mkdir -p "$dst"
for h in "$src"/*; do
    name=$(basename "$h")
    cp "$h" "$dst/$name"
    chmod +x "$dst/$name"
    echo "installed $name -> $dst/$name"
done
