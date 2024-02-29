# Run in civil server
# conda activate base
export VIDEO_DIR=ID761F-202303311612to202304041112/*
FILES=$(ls $VIDEO_DIR/*.mp4|tr " " "?")

for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  ffmpeg -i $f -preset slow -filter:v fps=fps=30 -crf 30 ID761F-202303311612to202304041112/compressed_vid/$(basename "$f")
  $f
  echo Done.. $(basename "$f")
  echo 
done
