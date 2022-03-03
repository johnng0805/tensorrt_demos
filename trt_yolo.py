"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

import pytesseract

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.crop_image import crop_image
from utils.tesseract_ocr import extract_text


WINDOW_NAME = 'TrtYOLODemo'

IMG_CROP_DIR = r'/home/john/Projects/tensorrt_demos/crop/'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    isCropped = False
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if isCropped == False:
            img = cam.read()
            if img is None:
                break
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            for idx, id in enumerate(clss):
                if id == 0:
                    cropped_img, dilation, thresh = crop_image(
                        img, boxes, idx, mBlur=3, gBlur=(5, 5))
                # os.chdir(IMG_CROP_DIR)
                #cv2.imwrite("img.jpg", cropped_img)
                #cv2.imwrite("img_dilation.jpg", dilation)
                #cv2.imwrite("img_thresh.jpg", thresh)
                    #isCropped = True
        # else:
            #plate = cv2.imread(IMG_CROP_DIR + 'img.jpg')
            #plate_dilation = cv2.imread(IMG_CROP_DIR + 'img_dilation.jpg')
            #plate_thresh = cv2.imread(IMG_CROP_DIR + 'img_thresh.jpg')
            #cv2.imshow('Dilation', plate_dilation)
                    text = extract_text(cropped_img, dilation, thresh)
                    print(text)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            cv2.imshow('cam', img)
            key = cv2.waitKey(1)
            if key == 27:
                break
            #isCropped = False


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    #cls_dict = get_cls_dict(args.category_num)
    cls_dict = {0: 'license-plate', 1: 'vehicle'}
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    # open_window(
    #    WINDOW_NAME, 'Camera TensorRT YOLO Demo',
    #    cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
