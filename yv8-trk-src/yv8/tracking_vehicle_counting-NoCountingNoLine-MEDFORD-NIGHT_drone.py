# conda activate yolov8.0-track



# ======== THERMAL ================
#
# python tracking_vehicle_counting-NoCountingNoLine-MEDFORD-NIGHT_drone.py mode="pedict" model=/media/zubin/Stuff1/CODE/YOLOv8-DeepSORT/runs/detect/train4/weights/best.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/trans-drone/Night-Medford-img/video-clips/DJI_0927-comp-22-comp-ENHANCED.mp4 conf=0.5 show=True agnostic_nms=0.4 line_thickness=1
# python tracking_vehicle_counting-NoCountingNoLine-MEDFORD-NIGHT_drone.py mode="pedict" model=/media/zubin/Stuff1/CODE/YOLOv8-DeepSORT/runs/detect/train4/weights/best.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/trans-drone/Night-Medford-img/video-clips/DJI_0928-comp-22-comp-ENHANCED conf=0.5 show=True agnostic_nms=0.4 line_thickness=1


'''
python tracking_vehicle_counting-NoCountingNoLine.py mode="pedict" model=best-therm-50ep.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/THERMAL/UML/VIDEOS-ORG-ALL/Day1_Lowell-UCross-Fletcher/Market_Basket-FletcherStreet/20220106_15_26_56_Pro.mp4 conf=0.55 show=True agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=1;python tracking_vehicle_counting-NoCountingNoLine.py mode="pedict" model=best-therm-50ep.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/THERMAL/UML/VIDEOS-ORG-ALL/Day2_Boston-Cambridge-MIT/Spot1/20220212_spot1_1.mp4 conf=0.55 show=True agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=1;python tracking_vehicle_counting-NoCountingNoLine.py mode="pedict" model=best-therm-50ep.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/THERMAL/UML/VIDEOS-ORG-ALL/Day2_Boston-Cambridge-MIT/Spot1/20220212_spot1_2.mp4 conf=0.55 show=True agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=1;python tracking_vehicle_counting-NoCountingNoLine.py mode="pedict" model=best-therm-50ep.pt source=/media/zubin/Stuff1/DATA/TRANSPORT/THERMAL/UML/VIDEOS-ORG-ALL/Day2_Boston-Cambridge-MIT/Spot2/20220212_spot2_3.mp4 conf=0.55 show=True agnostic_nms=0.4 cls=[0,1,2,3,5,7] line_thickness=1
'''

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

deepsort = None

object_counter = {}

object_counter1 = {}

line = [(100, 500), (1050, 500)]
line = []
tt_line = []

    
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=15, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True) #relpaced cfg_deep.DEEPSORT.N_INIT with 15
##########################################################################################
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
    '''
    if "erson" not in label and "car" not in label:
    	return
    '''
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        
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
def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    '''
    #Draw rectangle black for text background
    img = cv2.rectangle(img, (0, 0), (235,150), (0,0,0), -1) #left top
    img = cv2.rectangle(img, (850, 0), (width, 150), (0,0,0), -1) #right top
    '''
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]

        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y1+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)


        #DO NOT DRAW BOXES WHICH HAVE NOT MOVED IN 30 frames: zubin
        #deque([(470, 313), (473, 312), (477, 310), (481, 309), (484, 306)], maxlen=64)
        #if len(data_deque[id]) >= 2:
        #        break
        
        # add center to buffer
        data_deque[id].appendleft(center)
        '''
        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              #cv2.line(img, line[0], line[1], (255, 255, 255), 3)
              if "South" in direction:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
              if "North" in direction:
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        '''
        UI_box(box, img, label=label, color=color, line_thickness=2) #zubin uncomment later
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)  #zubin uncomment later
    
    #4. Display Count in top right corner
        '''
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            
            cv2.putText(img, f'Leaving', (width - 300, 35), 0, 1, [0, 0, 180], thickness=1, lineType=cv2.LINE_AA)   #leaving main
            cv2.putText(img, cnt_str, (width - 300, 75 + (idx*40)), 0, 1, [0, 0, 180], thickness = 1, lineType = cv2.LINE_AA)   #leaving main

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.putText(img, f'Entering', (11, 35), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)    #entering main
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)   #entering main
        '''
    return img
    
def draw_only_line_nos(img, names, offset=(0, 0)): #zubin
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    #img = cv2.rectangle(img, (0, 0), (235,150), (0,0,0), -1) #left top
    #img = cv2.rectangle(img, (850, 0), (width, 150), (0,0,0), -1) #right top
    '''
    #4. Display Count in top right corner
    for idx, (key, value) in enumerate(object_counter1.items()):
        cnt_str = str(key) + ":" +str(value)
        #cv2.line(img, (width - 400,25), (width,25), [45,45,45], 40)
        cv2.putText(img, f'Leaving', (width - 300, 35), 0, 1, [0, 0, 180], thickness=1, lineType=cv2.LINE_AA)   #leaving main
        cv2.putText(img, cnt_str, (width - 300, 75 + (idx*40)), 0, 1, [100, 100, 180], thickness = 1, lineType = cv2.LINE_AA)   #leaving main

    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" +str(value)
        #cv2.line(img, (20,25), (500,25), [45,45,45], 40)
        cv2.putText(img, f'Entering', (11, 35), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)       #entering main
        cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [180, 180, 180], thickness=1, lineType=cv2.LINE_AA)   #entering main
    '''
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
            #if cls not in classes_of_interest: #zubin
            #        continue #zubin
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
        else:
            draw_only_line_nos(im0, self.model.names)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8x.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
