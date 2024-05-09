#!/bin/bash

output_file="true_counts.csv"

# Empty the file
> "$output_file"
echo "file_name;successful_finds" > "$output_file"

# Loop recurisvely through csv files and count 'True' in each
for file in $(find . -type f -name "*.csv" ! -name "true_counts.csv"); do
    count=$(grep -o 'True' "$file" | wc -l)
    echo "\"$file\";$count" >> "$output_file"
done


