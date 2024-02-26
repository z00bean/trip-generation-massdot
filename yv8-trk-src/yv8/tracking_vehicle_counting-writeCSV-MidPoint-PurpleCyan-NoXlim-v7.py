# v5 Similar to v4, but with purple. Purple at edge, with yellow just above it.
# Ideal for 1 way traffic, when video begins with vehicle already in the edge of the video frame.
# Vehicle in purple and vehicle doesn't move much in one axis (x-axis), i.e. vehicle waits in one region of purple and disappears => Vehicle exits.
# Vehicle on purple then veicle on yellow => vehicle entering.
# If purple in all along x-axis at the bottom, and the vehicle travels all along, it means it is a vehicle on the main road.
# Edges of frame will be black and not purple. To try and ignore horizntal traffic.
# If vehicle ever touches CYAN, ignore for purple. PURPLE to CYAN : DO NOT COUNT.
#Count vehicle after it spends 1.5 seconds in purple.

# ENTERING: RED -> Green

# pip install ultralytics==8.0.0

# conda activate yolov8.0-track

#usps
# python tracking_vehicle_counting-writeCSV-MidPoint-Purple-v5.py mode="pedict" model=yolov8x.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.BATCH_1/ID373-202303211345to202303231030/100_compressed_vid/03210003.MOV conf=0.55 show=True agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=1
#374E-03210007.MOV

#4.kids
# python tracking_vehicle_counting-writeCSV-MidPoint-Purple-v5.py mode="pedict" model=yolov8s.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/9.BATCH_1/ID373-202303211345to202303231030/100_compressed_vid/03210003.MOV conf=0.55 show=True agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=1


import datetime
import os
#CHANGE 1 #new file for each location
#line_file = 'line-points/374E-100.txt'
#CHANGE 2 #Create new folder and change name of file
csv_file_event = "" # '/media/zubin/SSD/csv-det/374E/374E-data.txt'
#csv_file_event = 'xxx.txt'
#if os.path.exists(csv_file_event):
#    os.remove(csv_file_event)

file_name_vid = "" #to be assigned in main()
mask_file_loc = "" #to be assigned in main()
mask_file = 0 # Mask must have colors from the list: [Black(0), Red(1), Green(2), Blue(3), Yellow(4), White(5)]
mask_colors = [] # [Black, Yellow, White, Red, Green, Blue]; this can be a set. Black/0 will always be present.


fps_vid = -1
time_of_recording = "0."# datetime.time()


trackID_list = [] # list of all tracked objects till current time, vehicles in white mask.
listVehCounted = [] # List of vehicles counted
removed_trkIDlst = []

classes_of_interest=[0,1,2,3,5,7] 

import hydra
import torch
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

data_deque = {}
dict_counted = {} # For vehicles which have changed color twice, i.e. are in 2 dict, or in 1 dict and its track ended
dict_green = {}
dict_red = {}
dict_purple = {}
dict_yellow = {}

dict_purp_X = {}
dict_purp_Y = {}

deepsort = None

object_counter = {} #Direction1 (Leaving) | LEAVING: Green -> RED
object_counter1 = {} #Direction2 (Entering) | ENTERING: RED -> Green
object_counter2 = {} #Direction3, for later

#line = [(100, 500), (1050, 500)]
#line = [(24, 519),(930, 500)] #B1-ID373; make it small for other locations, like 1 px long
line = [(1, 1),(3, 3)] # point line
'''
tt_line = [] #only used temporarily to hold line text file content.
with open(line_file) as f: #zubin
    tt_line = [line.strip() for line in f]
    p1 = (tt_line)[0].split(',')
    line.append((int((p1[0])), int(p1[1])))
    p2 = (tt_line)[1].split(',')
    line.append((int((p2[0])), int(p2[1])))
'''
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=0.5,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True) #relpaced n_init=cfg_deep.DEEPSORT.N_INIT, with n_init=5, # replaced cfg_deep.DEEPSORT.MAX_IOU_DISTANCE with 0.4 
##########################################################################################

def getColorName(r_val, g_val, b_val):
    if r_val== 255 and g_val == 255 and b_val == 255:
        return "white"
    if r_val== 0 and g_val == 0 and b_val == 0:
        return "black"
    if r_val== 255 and g_val == 0 and b_val == 0:
        return "red"
    if r_val== 0 and g_val == 255 and b_val == 0:
        return "green"
    if r_val== 0 and g_val == 0 and b_val == 255:
        return "blue"
    if r_val== 255 and g_val == 255 and b_val == 0:
        return "yellow"
    if r_val== 100 and g_val == 0 and b_val == 255:
        return "purple"
    if r_val== 0 and g_val == 255 and b_val == 255:
        return "cyan"
    if r_val== 255 and g_val == 150 and b_val == 0:
        return "orange"
    return "unknown_color-"+str(r_val)+", "+str(g_val) +", "+str(b_val)

def getMaskPixelColor(x_pos, y_pos): #x_pos, y_pos are in terms of nd.array: vertical, hor, (720,1280)
    global mask_file
    b,g,r = cv2.split(mask_file)
    #print(x_pos)
    #print(y_pos)
    #print(b.shape)
    #print("\n\t*******"+getColorName(r[x_pos][y_pos], g[x_pos][y_pos], b[x_pos][y_pos])+"********")
    return getColorName(r[x_pos][y_pos], g[x_pos][y_pos], b[x_pos][y_pos])

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA) #Draws rectangle/BBox
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2) #Draw background of veh label

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

#Draw boxes and keep count
'''
Logic for counting: If vehicle changes color twice compute direction. 
Also, if vehicle changes color once and track is lost, compute direction.
'''
# identities = list of track IDs
def draw_boxes(img, f_no, bbox, names,object_id, identities=None, offset=(0, 0)):
    # object_id: Class ID; 
    # identities: track ID
    global trackID_list
    global removed_trkIDlst
    global listVehCounted

    global object_counter
    global object_counter1
    global object_counter2
    global dict_purple

    cv2.line(img, line[0], line[1], (46,162,112), 1)

    height, width, _ = img.shape
    #Draw rectangle black for text background
    img = cv2.rectangle(img, (0, 0), (235,150), (0,0,0), -1) #left top
    img = cv2.rectangle(img, (850, 0), (width, 150), (0,0,0), -1) #right top
    
    print("\n\n\n\n\t\t",dict_purple.keys())
    # Vehicle in color dict appear once and track is lost, compute direction. ENTERING: RED -> Green
    
    # remove tracked point from buffer if object is lost;
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)
    # Also check if lost object was last seen in purple.
    temp1 = dict_purple.copy()
    if len(temp1) > 0:
        for p_key in temp1:
            #print("\n\n\n\n\n",temp1,"\n\n\n\n\n")
            '''
            print("\t\t",p_key)
            print("\n\n\t\t",identities,"\n\n")
            print("\t\t",len(identities))
            '''
            if p_key not in identities and len(identities) > 0:
                #Vehicle left: increment count
                # Check id vehicle travel along X 
                '''
                print("\n\n\n\n EXIT PURPLE EXIT PURPLE ***********")
                print("\n\ndict_purp_Y len:")
                print(len(dict_purp_Y))
                print(max(dict_purp_Y[p_key]))
                print(min(dict_purp_Y[p_key]))
                print("\n\n\n\n\n",abs(max(dict_purp_Y[p_key]) - min(dict_purp_Y[p_key])),"\n\n\n\n\n")
                '''
                #Checking veh didnt travel al the way across the screen, and that it was not a short track (quarter second long at least)
                if abs(max(dict_purp_Y[p_key]) - min(dict_purp_Y[p_key])) < 99900 and len(dict_purp_Y[p_key]) >= 10:#int(fps_vid/10):
                    print("\n\n\n\n\nINSIDE    \t INSIDE  \ INSIDE \n\n\n\n\n")
                    if f_no > fps_vid*5 and len(dict_purp_Y[p_key]) < int(fps_vid/3):
                        dict_purple.pop(p_key)
                        continue
                    with open(csv_file_event, 'a+') as f:
                        #print(time_of_recording)
                        #print(time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid)))
                        curr_time = time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid))
                        f.write(str(curr_time)+","+ max(set(dict_purple[p_key]), key = dict_purple[p_key].count)+","+str(p_key)+",Direction1," + file_name_vid + '\n') #Leaving
                    listVehCounted.append(p_key)
                    if max(set(dict_purple[p_key]), key = dict_purple[p_key].count) not in object_counter: #obj_name is class of object; replaced with dict_purple[p_key]
                        object_counter[max(set(dict_purple[p_key]), key = dict_purple[p_key].count)] = 1
                    else:
                        object_counter[max(set(dict_purple[p_key]), key = dict_purple[p_key].count)] += 1
                    dict_purple.pop(p_key)
                else:
                    dict_purple.pop(p_key)


    #Check if bottom_center is in purple region in the begining of the trajectory.
    #get color changes twice (red to green or vide versa), compute direction.
    #Ignore detection points in black region. (DONE before calling this function.)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge (dot at the bottom)
        #bottom_center = (int((x2+x1)/ 2), int((y2+y2)/2)) #(y2+y2)/2 #x, y : in terms of cv2, vertical is y [0,720).
        center = (int((x2+x1)/ 2), int((y2+y2)/2)) #This is the midpoint. Tracking will be done on this.

        # get ID of object: trackID
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)
        
        #Checking and adding to color dict if bottom centre is on color part:zubin
        #if point on red and point/ID not already in dict_red.
        if getMaskPixelColor(int(center[1]), int(center[0])) == "red" and id not in dict_red.keys() and id not in listVehCounted:# and id not in removed_trkIDlst:
            dict_red[id] = obj_name
            trackID_list.append(id) 
            if id in dict_green.keys():
                #print("\tLEAVING: Green - > Red")
                #xxx = input()
                with open(csv_file_event, 'a+') as f:
                    #print(time_of_recording)
                    #print(time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid)))
                    curr_time = time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid))
                    f.write(str(curr_time)+","+obj_name+","+str(id)+",Direction1," + file_name_vid + '\n') #Leaving
                listVehCounted.append(id)
                if obj_name not in object_counter: #obj_name is class of object
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
        elif getMaskPixelColor(int(center[1]), int(center[0])) == "green" and id not in dict_green.keys() and id not in listVehCounted:# and id not in removed_trkIDlst:
            dict_green[id] = obj_name
            trackID_list.append(id)
            if id in dict_red.keys():
                #print("\tENTERING: RED -> Green")
                #xxx = input()
                with open(csv_file_event, 'a+') as f:
                    #print(time_of_recording)
                    #print(time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid)))
                    curr_time = time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid))
                    f.write(str(curr_time)+","+obj_name+","+str(id)+",Direction2," + file_name_vid + '\n') #Entering
                listVehCounted.append(id)
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        
        if getMaskPixelColor(int(center[1]), int(center[0])) == "yellow" and id not in dict_yellow.keys() and id not in listVehCounted:
            dict_yellow[id] = obj_name
            trackID_list.append(id)
            if id in dict_purple.keys():
                # Removing entry from dict_purple once vehicle is counted for direction.
                dict_purple.pop(id)
                with open(csv_file_event, 'a+') as f:
                    curr_time = time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid))
                    f.write(str(curr_time)+","+obj_name+","+str(id)+",Direction2," + file_name_vid + '\n') #Entering
                listVehCounted.append(id)
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        # vehicle appears in purple and eventually exits:
        elif getMaskPixelColor(int(center[1]), int(center[0])) == "purple" and id not in listVehCounted:
            if id not in dict_purple.keys():
                dict_purple[id] = [obj_name]
            else:
                dict_purple[id].append(obj_name)
            if id not in trackID_list:# listVehCounted:
                trackID_list.append(id)
            if id not in dict_purp_X.keys():
                dict_purp_X[id] = [int(center[1])]
                dict_purp_Y[id] = [int(center[0])]
            else:
                dict_purp_X[id].append(int(center[1]))
                dict_purp_Y[id].append(int(center[0]))
            # If vehicle lingers in purple count it 
            if len(dict_purple[id]) > int(fps_vid*3.5):
                dict_purple.pop(id)
                with open(csv_file_event, 'a+') as f:
                        curr_time = time_of_recording + datetime.timedelta(seconds=round(f_no / fps_vid))
                        f.write(str(curr_time)+","+obj_name+","+str(id)+",Direction1," + file_name_vid + '\n') #Leaving
                listVehCounted.append(id)
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
            
        # Removing vehicle id whichtouch cyan right after touching purple. (But not the othewayround.)
        if getMaskPixelColor(int(center[1]), int(center[0])) == "cyan" and id in dict_purple.keys() and id not in listVehCounted:
            dict_purple.pop(id)
            


        # Add lower center to buffer
        data_deque[id].appendleft(center)
        
        UI_box(box, img, label=label, color=color, line_thickness=2) # Draws BBox, labels and label background.

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails, Dot at the bottom comes from here
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)  #zubin uncomment later
    
    #4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            
            cv2.putText(img, f'Direction2', (width - 300, 35), 0, 1, [0, 0, 180], thickness=1, lineType=cv2.LINE_AA)   #entering main
            cv2.putText(img, cnt_str, (width - 300, 75 + (idx*40)), 0, 1, [0, 0, 180], thickness = 1, lineType = cv2.LINE_AA)   #entering main

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.putText(img, f'Direction1', (11, 35), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)    #leaving main
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)   #leaving main
    return img
    
def draw_only_line_nos(img, names, offset=(0, 0)): #zubin
    cv2.line(img, line[0], line[1], (46,162,112), 1)

    height, width, _ = img.shape
    img = cv2.rectangle(img, (0, 0), (235,150), (0,0,0), -1) #left top
    img = cv2.rectangle(img, (850, 0), (width, 150), (0,0,0), -1) #right top
    #4. Display Count in top right corner
    for idx, (key, value) in enumerate(object_counter1.items()):
        cnt_str = str(key) + ":" +str(value)
        #cv2.line(img, (width - 400,25), (width,25), [45,45,45], 40)
        cv2.putText(img, f'Direction2', (width - 300, 35), 0, 1, [0, 0, 180], thickness=1, lineType=cv2.LINE_AA)   #entering main
        cv2.putText(img, cnt_str, (width - 300, 75 + (idx*40)), 0, 1, [100, 100, 180], thickness = 1, lineType = cv2.LINE_AA)   #entering main

    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" +str(value)
        #cv2.line(img, (20,25), (500,25), [45,45,45], 40)
        cv2.putText(img, f'Direction1', (11, 35), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)       #leaving main
        cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)   #leaving main
    return img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    #Get detection results, pass to deepsort.update and call draw box function.
    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        draw_only_line_nos(im0, self.model.names)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            if cls not in classes_of_interest: #zubin
                    continue #zubin
            #Checking if object bottom mid-point in black region
            #print("\n")
            #print(getMaskPixelColor(int(y_c+int(bbox_h/2)), int(x_c)))
            #print(int(y_c+int(bbox_h/2)))
            #print(int(x_c))
            #print("\n")
            #301.5, 1198.5
            #cv2.circle(img, int(y_c+int(bbox_h/2)), int(x_c), 2, (0,0,0), 12)
            if getMaskPixelColor(int(y_c+int(bbox_h/2)), int(x_c)) == "black": #x, y, interms of cv2. X: along horizontal axis
                #print("\n\t\t***********BLACK**********")
                continue #Will cause error if 0 detections are passed to deepsort.update()
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = [] #zubin: added later
        if len(xywh_bboxs) > 0: #zubin: condition added later
            outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0, frame, bbox_xyxy, self.model.names, object_id,identities)
        else:
            draw_only_line_nos(im0, self.model.names)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()

    global time_of_recording
    global fps_vid
    global file_name_vid
    global mask_file_loc
    global mask_file
    global mask_colors
    global csv_file_event

    cfg.model = cfg.model or "yolov8x.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    cfg.project = os.path.join("/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/OUT_DIR_9", cfg.source.split('/')[-4], cfg.source.split('/')[-3], cfg.source.split('/')[-2]) #"/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/OUT_DIR_9/Batch1"
    cfg.name = cfg.source.split('/')[-1]

    #print(os.path.join(cfg.project, cfg.name))
    #Checking output folder exists ()
    if os.path.exists(os.path.join(cfg.project, cfg.name)):
        print("Output video/folder exists: "+cfg.name +"\nExiting,\n\n")
        return

    csv_file_event = os.path.join("/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/OUT_DIR_9", cfg.source.split('/')[-4], str(cfg.source.split('/')[-3][:-4])+ str("-"+cfg.source.split('/')[-2][:3]+".txt"))
    
    # create video capture object
    data = cv2.VideoCapture(cfg.source)
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_vid = data.get(cv2.CAP_PROP_FPS)

    mask_file_loc = os.path.join(os.path.dirname(cfg.source), "mask.png")
    #print(mask_file_loc)
    if os.path.isfile(mask_file_loc):
        #print(os.path.getsize(mask_file_loc))
        mask_file = cv2.imread(mask_file_loc, flags=cv2.IMREAD_COLOR)
        b,g,r = cv2.split(mask_file)
        number_of_white_pix = 0
        number_of_black_pix = 0
        number_of_red_pix = 0
        number_of_green_pix = 0
        number_of_blue_pix = 0
        number_of_yellow_pix = 0 #Yellow = Red + Green
        #print(r.shape) # (720, 1280)
        x_l_im = r.shape[1] #width of image, 1280, y
        y_l_im = r.shape[0] #height, 720, x
        for i in range(0, y_l_im):
            for j in range(0, x_l_im):
                if "black" not in mask_colors and r[i][j] == 0 and  g[i][j] == 0 and b[i][j] == 0:
                    mask_colors.append("black")
                    number_of_black_pix += 1
                    continue
                if "white" not in mask_colors and r[i][j] == 255 and  g[i][j] == 255 and b[i][j] == 255:
                    mask_colors.append("white")
                    number_of_white_pix += 1
                    continue
                if "red" not in mask_colors and r[i][j] == 255 and  g[i][j] == 0 and b[i][j] == 0:
                    mask_colors.append("red")
                    number_of_red_pix += 1
                    continue
                if "green" not in mask_colors and r[i][j] == 0 and  g[i][j] == 255 and b[i][j] == 0:
                    mask_colors.append("green")
                    number_of_green_pix += 1
                    continue
                if "blue" not in mask_colors and r[i][j] == 0 and  g[i][j] == 0 and b[i][j] == 255:
                    mask_colors.append("blue")
                    number_of_blue_pix += 1
                    continue
                if "yellow" not in mask_colors and r[i][j] == 255 and  g[i][j] == 255 and b[i][j] == 0:
                    mask_colors.append("yellow")
                    number_of_yellow_pix += 1
                    continue
                if "purple" not in mask_colors and r[i][j] == 100 and  g[i][j] == 0 and b[i][j] == 255:
                    mask_colors.append("purple")
                    number_of_yellow_pix += 1
                    continue
                if "cyan" not in mask_colors and r[i][j] == 0 and  g[i][j] == 255 and b[i][j] == 255:
                    mask_colors.append("cyan")
                    number_of_yellow_pix += 1
                    continue
                if "orange" not in mask_colors and r[i][j] == 255 and  g[i][j] == 150 and b[i][j] == 0:
                    mask_colors.append("orange")
                    number_of_yellow_pix += 1
                    continue
        '''
        print(number_of_white_pix)
        print(number_of_black_pix)
        print(number_of_red_pix)
        print(number_of_green_pix)
        print(number_of_blue_pix)
        print(number_of_yellow_pix)
        '''
    #return

    if os.path.isfile(cfg.source) and os.path.getsize(cfg.source) > 0:
        # file modification timestamp of a file
        m_time = os.path.getmtime(cfg.source)
        # convert timestamp into DateTime object
        dt_m = datetime.datetime.fromtimestamp(m_time)
        time_of_recording = dt_m + datetime.timedelta(hours=4) - datetime.timedelta(seconds=round(frames / fps_vid)) #time when video started recording; round(frames / fps) = duration of video
        

        file_name_vid = cfg.source.split("/")[-1] #assigning filename of source video to variable
        predictor = DetectionPredictor(cfg)
        #predictor.save_dir = "/media/zubin/Stuff1/DATA/TRANSPORT/IR-sample-data/OUT_DIR_9/Batch1"
        predictor()


if __name__ == "__main__":
    predict()
