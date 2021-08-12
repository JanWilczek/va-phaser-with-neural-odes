MEASURES_SUMMARY_FILE=measures_summary.csv
'segSNR,fw-segSNR,ESR,ESR+DC+prefilter,DI,ODG' > $MEASURES_SUMMARY_FILE
find . -type f -name '*.measures.csv' -print | cat | sed -n 2p >>  $MEASURES_SUMMARY_FILE
