#!/bin/bash
# Copy the track data the harness needs (wall rays, distance along the track)
# from the Trinity-RL/Tracks repository into tracks/: interpolate_track.py and
# <map>/points-offset.csv for every map in maps.yml.
#
#   scripts/sync_tracks.sh [path/to/Tracks]     (default: ../Tracks)
#
# Commit the result. The harness reads tracks/ from this checkout, so what's
# committed here is exactly what every evaluation uses.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-../Tracks}"
[ -f "$SRC/interpolate_track.py" ] || { echo "No Tracks repo at $SRC"; exit 1; }

rm -rf tracks
mkdir -p tracks
cp "$SRC/interpolate_track.py" tracks/
for map in $(python3 -c "import yaml; print(' '.join(m['file'].removesuffix('.Challenge.Gbx') for m in yaml.safe_load(open('maps.yml'))['maps']))"); do
    if [ -f "$SRC/$map/points-offset.csv" ]; then
        mkdir -p "tracks/$map"
        cp "$SRC/$map/points-offset.csv" "tracks/$map/"
        echo "  $map"
    else
        echo "  $map: no track data in $SRC (rays will be (None, None) on it)"
    fi
done
echo "Tracks repo commit: $(git -C "$SRC" rev-parse HEAD)" > tracks/SOURCE
echo "Synced from $SRC into tracks/"
