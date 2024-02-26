#03210751.MOV
import datetime
import cv2

import os

# Path to the file
path = r"03210751.MOV"
path = r"/media/zubin/SSD2/ITEVideos-3rd-4-12-2023/ID410N-202304041337-to-202304071200/100_compressed_vid/04040002.MOV"

# file modification timestamp of a file
m_time = os.path.getmtime(path)
# convert timestamp into DateTime object
dt_m = datetime.datetime.fromtimestamp(m_time)
print('Modified on:\t\t', dt_m)
print('Added 4 hrs on:\t\t', dt_m+ datetime.timedelta(hours=4))
print('Subtract 10 sec:\t', dt_m+ datetime.timedelta(hours=4) - datetime.timedelta(seconds=10))

data = cv2.VideoCapture(path)
# count the number of frames
frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
fps_vid = data.get(cv2.CAP_PROP_FPS)
print('\nFPS:\t\t\t', fps_vid)
print('Video duration:\t\t', frames/fps_vid)
print('Event at half duration:\t', frames/(2*fps_vid))

#Video begin time
time_of_recording = dt_m + datetime.timedelta(hours=4) - datetime.timedelta(seconds=round(frames / fps_vid))
print('Time of record begin:\t', time_of_recording)
f_no = int(frames/2)
curr_time = time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid))
print('Current time:\t\t', curr_time)


# file creation timestamp in float
c_time = os.path.getctime(path)
# convert creation timestamp into DateTime object
#dt_c = datetime.datetime.fromtimestamp(c_time)#zubin
#print('Created on:', dt_c)#zubin
