#! /bin/bash

FILES=data/raw/*/*.txt
mkdir -p data/clean
mkdir -p data/merged
for f in $FILES
do
    echo "Processing "$f" file..."
    bname="$(basename -- "$f")"
    dirname="$(dirname "$f")"
    starttime="$(cat "$dirname/starttime")"
    echo "  Start time: $starttime"
    python preprocessing/preprocess.py \
           --ti "$dirname/telespor.csv" \
           --to data/clean/telespor.$bname \
           --ci $f \
           --co data/clean/capture.$bname \
           --allout data/merged/$bname \
           --starttime "$starttime"
done
