# python tracking_vehicle_counting-writeCSV.py mode="pedict" model=yolov8x.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/7.count-motion-vehicle/408E-03170024.MOV conf=0.5 show=True agnostic_nms=0.4 classes=[0,1,2,3,5,7] line_thickness=1

# conda activate yolov8.0-track
# sh tracking-veh-runall.sh 

#yolov8s-uml-fl-mit.pt
#best-therm-50ep.pt
#yolov8x.pt
# tracking_vehicle_counting-writeCSV-MidPoint-PurpleCyan-v6.py
# tracking_vehicle_counting-writeCSV-MidPoint-PurpleCyan-NoXlim-v7.py
# ffmpeg -i out-F4-S-toll.mp4 -vf scale=-1:1080 -preset slow -crf 32 out-F4-S-toll-comp.mp4
#export VIDEO_DIR=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/7.count-motion-vehicle
#export VIDEO_DIR=/media/zubin/SSD/ID373-202303211345to202303231030/100EK113
#export VIDEO_DIR=/media/zubin/SSD/ID374E-202303211235to202303231053/100EK113
#export VIDEO_DIR=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.BATCH_3/ID437-202304071246to202304111118/100_compressed_vid
#export VIDEO_DIR=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.BATCH_2/ID690E-202303311443to202304041147/100_compressed_vid
#export VIDEO_DIR=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.LowLight/ID1/1
#export VIDEO_DIR=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.Therm/ID1/1
export VIDEO_DIR=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.UML/ID1/1
# ID707N-202303231150to202303281038
# ID690W-202303311457to202304041142
# ID707S-202303231212to202303281043
# 9.BATCH_3 ID443N-202304041238to202304071222
# 9.BATCH_1 ID409-202303101755to202303160945
FILES=$(ls $VIDEO_DIR/*.mp4|tr " " "?")

for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  #python tracking_vehicle_counting-writeCSV-v3.py mode="pedict" model=yolov8x.pt conf=0.51 show=False agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=2 source="$f" #vid_stride=3 
  python tracking_vehicle_counting-writeCSV-v4-THERM.py mode="pedict" model=best-therm-50ep.pt conf=0.45 show=True agnostic_nms=0.4 line_thickness=12 source="$f" 
  #source=$(basename "$f")
  echo $f

  #cp $f mask_rcnn_orpn_r50_fpn_1x_isaid_on_AWS-FStNB-v1/compressed_videos/$(basename "$f")
  echo Done.. $(basename "$f")
  echo 
done
