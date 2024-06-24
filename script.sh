#!/bin/bash

# Check if the input file exists
if [ ! -f "input.txt" ]; then
    echo "Error: input.txt not found."
    exit 1
fi

# Check if the flookup binary (english.bin) exists
if [ ! -f "english.bin" ]; then
    echo "Error: english.bin not found."
    exit 1
fi

# Loop through each line in input.txt
while IFS= read -r line; do
    # Ignore empty lines
    if [ -n "$line" ]; then
        # Apply the command
        echo "$line" | flookup english.bin
    fi
done < "input.txt"
