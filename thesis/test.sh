#!/usr/bin/env bash

retval=0

if grep -q "undefined references" $1; then
    echo "There were undefined references found"
    grep "LaTeX Warning: Citation" $1
    grep "LaTeX Warning: Reference" $1
    retval=1
fi

files_bom=$(find . -name "*.txt" | xargs -I filename grep -l $'^\xEF\xBB\xBF' filename)
if [ "$files_bom" != "" ]; then
    echo "Files with BOM detected"
    echo "$files_bom"
    retval=1
fi

files_non_utf=$(find . -name "*.tex" | xargs -I filename file --mime filename | grep -v 'us-ascii\|utf-8\|x-empty')
if [ "$files_non_utf" != "" ]; then
    echo "Files with not UTF-8 detected"
    echo "$files_non_utf"
    retval=1
fi

exit $retval
