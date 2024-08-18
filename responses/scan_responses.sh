#!/bin/bash
#
# Script name: scan_responses.sh
#
# Description:
#    This script calls a python script question_row_compare.py which compares two fields in the same row of a csv file.
#    It is passed the name of the csv file and the row number to start comparing rows. It will continue comparing
#    rows incrementally when a space bar is pressed or quit with q is pressed. max_rows is currently set to the length
#    of the response csv file
#
# Usage: scan_responses.sh <csv file> <starting row number>
#    

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <csv file> <start row>"
	exit 1
fi

if ! [ "$2" -eq "$2" ] 2> /dev/null; then
	echo  "Error: Argument must be an integer."
	exit 1
fi

start=$2
max_rows=1111

for i in $(seq $start $max_rows)
do
    python ../python/question_row_compare.py $1 --row=$i --field_name1=Response_Llama3_1-8B-Instruct --field_name2=Response_Llama3_1-8B-Instruct-AMD-python | more
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
       read -n1 -s -r -p $'Press space bar to continue, q to exit...\n' input
       if [ "$input" = " " ]; then
          continue 
       elif [ "$input" == 'q' ]; then
          break
       fi
    else exit $exit_code
    fi
done
