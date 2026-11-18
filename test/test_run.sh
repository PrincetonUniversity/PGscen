#!/usr/bin/env bash

set -euo pipefail

REPODIR=$( dirname "$0" )
OUTDIR=$REPODIR/temp

python -m unittest discover -s "$REPODIR" -p 'test_console_scripts.py' -v

# Clean up and create output directory
rm -rf $OUTDIR
mkdir -p $OUTDIR

# echo "--- Running pgscen-load test (CSV output) ---"
# pgscen-load 2017-05-27 1 -o $OUTDIR -n 10 --test
# test -d "$OUTDIR/20170527/load"

# echo "--- Running pgscen-solar test (pickle output) ---"
# pgscen-solar 2017-05-27 1 -o $OUTDIR -n 10 --test -p
# test -f "$OUTDIR/scens_2017-05-27.p.gz"

# echo "--- Running pgscen-wind test (CSV output) ---"
# pgscen-wind 2018-05-02 1 -o $OUTDIR -n 10 --test
# test -d "$OUTDIR/20180502/wind"

# echo "--- Running pgscen-load-solar test (pickle output) ---"
# pgscen-load-solar 2017-03-27 1 -o $OUTDIR -n 10 --test -p
# test -f "$OUTDIR/scens_2017-03-27.p.gz"

# echo "--- Running pgscen command (separate load/solar, CSV) ---"
# pgscen 2018-04-10 1 -o $OUTDIR -n 10 --test
# test -d "$OUTDIR/20180410/load"
# test -d "$OUTDIR/20180410/solar"
# test -d "$OUTDIR/20180410/wind"

echo "--- Running pgscen command (joint load/solar, pickle) ---"
pgscen 2018-05-11 1 -o $OUTDIR -n 10 --test --joint -p
test -f "$OUTDIR/scens_2018-05-11.p.gz"

echo "--- All tests passed! ---"

# Clean up the output directory
rm -r $OUTDIR

echo "Cleanup complete."
