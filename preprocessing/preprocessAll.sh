FILES=data/raw/first/*.txt
mkdir -p data/clean
mkdir -p data/merged
for f in $FILES
do
    echo "Processing "$f" file..."
    bname=$(basename -- "$f")
    python preprocessing/preprocess.py --ti data/raw/first/telespor.csv --to data/clean/telespor.$bname --ci $f --co data/clean/capture.$bname --allout data/merged/$bname --starttime "2019-10-11 10:17:00"
done
#cp data/first/capture_246058.txt.merged.csv all_data_1.csv

# FILES=data/second/*.txt
# for f in $FILES
# do
#     echo "Processing "$f" file..."
#     python preprocess.py --ti data/second/telespor.csv --to data/second/telespor_clean.csv --ci $f --co $f.clean.csv --allout $f.merged.csv --starttime "2019-10-22 16:34:00" 
# done
# cp data/second/capture_864481.txt.merged.csv all_data_2.csv
