#!/bin/sh
# Generate one SAR dataset size.
#
# Usage: sar_run_datagen.sh <datagen> <input_dir> <out_dir>
#
# The generator reads Parameters.txt and Targets.txt from the working
# directory and names its four outputs on the command line, so the input
# directory is entered and the outputs are written out by absolute path.
#
# Outputs land in a private directory and are moved into place afterwards.
# The result is a deterministic function of the two inputs, so two builds
# generating the same size concurrently produce identical bytes; the move
# is what keeps a reader from ever opening a half-written file.
set -e

datagen="$1"
input_dir="$2"
out_dir="$3"

mkdir -p "$out_dir"
staging="$out_dir/.staging.$$"
mkdir -p "$staging"
trap 'rm -rf "$staging"' EXIT INT TERM

cd "$input_dir"
"$datagen" "$staging/Data.bin" "$staging/PlatformPosition.bin" \
           "$staging/PulseTransmissionTime.bin" "$staging/Radar.txt"

for f in Data.bin PlatformPosition.bin PulseTransmissionTime.bin Radar.txt; do
    mv -f "$staging/$f" "$out_dir/$f"
done
