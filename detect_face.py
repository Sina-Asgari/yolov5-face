# -*- coding: UTF-8 -*-
import argparse
from pyexpat import model
import time
from pathlib import Path
from tkinter import N

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from fastapi import FastAPI
import uvicorn
app = FastAPI()

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    n_landmarks = coords.shape[1] // 2
    
    coords[:, [i for i in range(0, n_landmarks*2, 2)]] -= pad[0]  # x padding
    coords[:, [i for i in range(1, n_landmarks*2 + 1, 2)]] -= pad[1]  # y padding
    coords[:, :n_landmarks * 2] /= gain
    #clip_coords(coords, img0_shape)
    for i in range(n_landmarks * 2):
        coords[:, i].clamp_(0, img0_shape[(i+1)%2])  # x1   

    # coords[:, 0].clamp_(0, img0_shape[1])  # x1
    # coords[:, 1].clamp_(0, img0_shape[0])  # y1
    # coords[:, 2].clamp_(0, img0_shape[1])  # x2
    # coords[:, 3].clamp_(0, img0_shape[0])  # y2
    # coords[:, 4].clamp_(0, img0_shape[1])  # x3
    # coords[:, 5].clamp_(0, img0_shape[0])  # y3
    # coords[:, 6].clamp_(0, img0_shape[1])  # x4
    # coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    n_landmarks = len(landmarks) // 2
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)


    for i in range(n_landmarks):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        color = random.randint(0, 255, size=3)
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
        cv2.circle(img, (point_x, point_y), tl+1, color, -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img



def detect_one(model, image_path, device, n_landmarks=5):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres, n_landmarks=n_landmarks)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:5+n_landmarks*2] = scale_coords_landmarks(img.shape[2:], det[:, 5:5+n_landmarks*2], orgimg.shape).round()
            landmarks_ = []
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:5+n_landmarks*2].view(-1).tolist()
                class_num = det[j, 5+n_landmarks*2].cpu().numpy()
                orgimg = show_results(orgimg, xyxy, conf, landmarks, class_num)
                landmarks_.append(landmarks)

    cv2.imwrite('result.jpg', orgimg)
    return landmarks_




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--n_landmark', type=int, default=5, help='number of landmarks')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    # detect_one(model, opt.image, device, n_landmarks=opt.n_landmark)

    @app.get('/')
    async def get_landmarks(img_path: str = opt.image, n_landmarks: int = opt.n_landmark):
        landmarks = detect_one(model, img_path, device, n_landmarks=n_landmarks)
        # return {f"lx_{i//2}":landmarks[i] if i%2==0 else f"ly_{i//2}" for i in range(len(landmarks))}
        lndmrks = {}
        for j in range(len(landmarks)):
            d = {}
            for i in range(len(landmarks[j])):
                if i%2==0:
                    d[f"lx_{i//2}"] = landmarks[j][i]
                else:
                    d[f"ly_{i//2}"] = landmarks[j][i]
            lndmrks[f"face_{j}"] = d
        return lndmrks

    uvicorn.run(app)


