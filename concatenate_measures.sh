MEASURES_SUMMARY_FILE=measures_summary.csv
echo TestFile,segSNR,fw-segSNR,ESR,ESR+DC+prefilter,DI,ODG > $MEASURES_SUMMARY_FILE
for file in `find . -type f -name '*measures.csv' -print | sort`
do
    echo "$file,`cat $file | sed -n 2p`" >>  $MEASURES_SUMMARY_FILE
done
