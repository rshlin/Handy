#!/usr/bin/env bash
# Rebase the fork branch onto an upstream release tag and run the fork's tests.
# Usage: scripts/fork/rebase.sh v0.9.9
set -euo pipefail

tag="${1:?usage: $0 <upstream tag>}"
branch=rshlin/remote-whisper

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree not clean" >&2
  exit 1
fi

git fetch upstream --tags
git rev-parse -q --verify "refs/tags/$tag" >/dev/null || { echo "error: no tag $tag" >&2; exit 1; }
git checkout "$branch"

if ! git rebase "$tag"; then
  echo
  echo "Conflicts rebasing $branch onto $tag:"
  git diff --name-only --diff-filter=U | sed 's/^/  /'
  echo
  echo "Resolve them (the fork only adds hook lines, see FORK.md), then"
  echo "  git add <files> && git rebase --continue"
  echo "or give up with: git rebase --abort"
  exit 1
fi

echo
echo "Upstream files changed by the fork (should be the hook lines only):"
git diff "$tag" --stat -- . ':!FORK.md' ':!scripts/fork' ':!src-tauri/src/managers/transcription/remote_whisper.rs'

echo
(cd src-tauri && CMAKE_POLICY_VERSION_MINIMUM=3.5 cargo test --lib remote_whisper)

echo
echo "Rebased $branch onto $tag. Push with: git push --force-with-lease origin $branch"
