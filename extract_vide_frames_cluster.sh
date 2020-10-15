#!/bin/bash
#FILES="./video_deneme"
##read -p 'file_name: ' FILES 
##for f in *.jpeg ; do echo "$f"; done
output="../data/video_frames"
FILES="../data/VideoFlash"
a=$FILES/*.flv 
echo $a
#for file in $FILES/*.flv; do
for file in $a;do
    basename "$file"
    f="$(basename -- $file)"
    destination="$output/$f";
    #basename "$destination"
    #f="$(basename -- $destination)"
    #echo "$destination"
    mkdir -p "$destination";
    ffmpeg -i "$file" "$destination/image-%d.png";
    echo $destination
done

