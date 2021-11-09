#!/bin/usr/env bash

OUTDIR=$( dirname "$0" )/temp

pgscen-load 2018-03-02 2 -o $OUTDIR --test
pgscen-wind 2018-05-02 3 -o $OUTDIR --test
pgscen-solar 2017-05-27 1 -o $OUTDIR --test
pgscen-load-solar 2017-03-27 2 -o $OUTDIR --test

rm -r $OUTDIR

