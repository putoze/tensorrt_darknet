"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.fitEllipse import find_eye_roi,find_max_contour

WINDOW_NAME = 'TrtYOLODemo'


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
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
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
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        # auto select if the frame is gray or RGB
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        # write my self code
        # (img, text, org, fontFace, fontScale, color, thickness, lineType)
        cv2.putText(img,"Esc: Quit",(cam.img_width-300,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img,"F  : Full Screen",(cam.img_width-300,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        bb_list = []
        for bb, cf, cl in zip(boxes, confs, clss):
            if cl == 0:
                bb_list.append(bb)
        if(len(bb_list) >= 1):
            x_min, y_min, x_max, y_max = bb_list[0][0], bb_list[0][1], bb_list[0][2], bb_list[0][3]
            eye_h1 = y_max-y_min
            eye_w1 = x_max-x_min
            if(len(bb_list) == 2):
                if(bb_list[0][0] > bb_list[1][0]):
                    x1_min, y1_min, x1_max, y1_max = x_min, y_min, x_max, y_max
                    x_min, y_min, x_max, y_max = bb_list[1][0], bb_list[1][1], bb_list[1][2], bb_list[1][3]
                    eye_h1 = y_max-y_min
                    eye_w1 = x_max-x_min
                else:
                    x1_min, y1_min, x1_max, y1_max = bb_list[1][0], bb_list[1][1], bb_list[1][2], bb_list[1][3]
                eye_h2 = y1_max-y1_min
                eye_w2 = x1_max-x1_min
    
                img[0:eye_h1,eye_w1:eye_w1+eye_w1,:] = img[y1_min:y1_min+eye_h1,x1_min:x1_min+eye_w1,:]
            img[0:eye_h1,0:eye_w1,:] = img[y_min:y_max,x_min:x_max,:]
            

        #(Gray,Binary,Morphological,Gaussian blur,Sobel,Canny,Find contours)
        # flag_list = [1,1,1,1,1,1,1]
        # target_img,contours = find_eye_roi(img[0:eye_h1,0:eye_w1,:],flag_list)
        # target_img = find_max_contour(target_img,contours)
        # img[0:eye_h1,0:eye_w1,:] = target_img

        # ---------------------------------------------
        img = vis.draw_bboxes(img, boxes, confs, clss)
        """Draw fps number at down-right corner of the image."""
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
