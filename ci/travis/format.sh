#!/bin/bash

# Script for YAPF formatting. Adapted from https://github.com/ray-project/ray/blob/master/ci/travis/format.sh.

YAPF_FLAGS=(
    '--style' ".style.yapf"
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=()

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" -- "$@"
}

# Format all files, and print the diff to `stdout` for Travis.
format_all() {
    yapf --diff "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}"
}

# This flag formats individual files. `--files` *must* be the first command line
# arg to use this option.
if [[ "$1" == '--files' ]]; then
    format "${@:2}"
    # If `--all` is passed, then any further arguments are ignored and the
    # entire Python directory is formatted.
elif [[ "$1" == '--all' ]]; then
    format_all
fi
