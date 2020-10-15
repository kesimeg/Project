#!/bin/bash
#FILES="./video_deneme"
##read -p 'file_name: ' FILES 
##for f in *.jpeg ; do echo "$f"; done
output="../audio_files"
a=$FILES/*.flv 
echo $a
#for file in $FILES/*.flv; do
for file in *.flv;do
    destination="$output/$file";
    ##mkdir -p "$destination";
    ##ffmpeg -i "$file" "$destination/$file.wav";
    ffmpeg -i "$file" "$output/"${file%".flv"}.wav"";
    #echo $destination
done

