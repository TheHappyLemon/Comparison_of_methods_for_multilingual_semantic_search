#!/bin/bash

output_file="true_counts_FLAT.csv"

# Empty the file
> "$output_file"
echo "file_name;successful_finds" > "$output_file"

# Loop recursively through csv files in numerical order and count 'True' in each
for file in $(find . -type f -name "*.csv" ! -name "true_counts.csv" | sort -V); do
    count=$(grep -o 'True' "$file" | wc -l)
    echo "\"$file\";$count" >> "$output_file"
done
