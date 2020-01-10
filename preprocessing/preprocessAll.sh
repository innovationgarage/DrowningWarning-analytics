#! /bin/bash

FILES=data/raw/*/*.txt
mkdir -p data/clean
mkdir -p data/merged
for f in $FILES
do
    echo "Processing "$f" file..."
    filename="$(basename -- "$f")"
    basename="$(basename -- "$f" .txt)"
    dirname="$(dirname "$f")"
    if ! [ -e "$dirname/$basename.starttime" ]; then
        echo "  Missing start time"
    else
        starttime="$(cat "$dirname/$basename.starttime")"
        echo "  Start time: $starttime"
        python preprocessing/preprocess.py \
               --ti "$dirname/telespor.csv" \
               --to data/clean/telespor.$filename \
               --ci $f \
               --co data/clean/capture.$filename \
               --allout data/merged/$filename \
               --starttime "$starttime"
    fi
done
