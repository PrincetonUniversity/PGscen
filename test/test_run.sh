#!/bin/usr/env bash

REPODIR=$( dirname "$0" )
OUTDIR=$REPODIR/temp

pgscen-load 2018-03-02 2 -o $OUTDIR --test
pgscen-solar 2017-05-27 1 -o $OUTDIR --test
pgscen-load-solar 2017-03-27 2 -o $OUTDIR --test
pgscen-load 2018-03-02 4 -o $OUTDIR

pgscen-wind 2018-05-02 3 -o $OUTDIR --test
python $REPODIR/wind_test.py $OUTDIR/wind

rm -r $OUTDIR
