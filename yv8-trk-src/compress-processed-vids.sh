#!/bin/bash

# Specify the root directory
root_directory="ID707S-202303231212to202303281043/100_compressed_vid"

# Iterate through subdirectories and print file names
for subdir in "$root_directory"/*; do
    if [ -d "$subdir" ]; then
        for file in "$subdir"/*; do
            if [ -f "$file" ]; then
                case "$file" in
                    *.mp4)
                        echo "MP4 file in $subdir: $(basename "$file")"
                        echo "$(basename "$file" .mp4)-comp.mp4"
                        echo "$file"
                        echo "$subdir/$(basename "$file" .mp4)-comp.mp4"
                        ffmpeg -i "$file" -preset slow -crf 32 "$subdir/$(basename "$file" .mp4)-comp.mp4" -loglevel warning
                        rm "$file"
                        ;;
                esac
            fi
        done
    fi
done