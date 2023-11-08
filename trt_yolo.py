"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.mtcnn import TrtMtcnn
import numpy as np


from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from process_alg.fitEllipse import find_max_Thresh

WINDOW_NAME = 'TrtYOLODemo'
# coordinate
coordinates = []

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

def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for j in range(5):
            # cv2.putText(img,str(j),(int(ll[j]), int(ll[j+5])),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, (0, 255, 0), 2)
    return img


def affineMatrix_eye(img, boxes, landmarks, scale=2.5):
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        nose = np.array([ll[2],ll[7]], dtype=np.float32)
        left_eye = np.array([ll[0],ll[5]], dtype=np.float32)
        right_eye = np.array([ll[1],ll[6]], dtype=np.float32)
        eye_width = right_eye - left_eye
        angle = np.arctan2(eye_width[1], eye_width[0])
        # print(eye_width)
        center = nose
        alpha = np.cos(angle)
        beta = np.sin(angle)
        w = np.sqrt(np.sum(eye_width**2)) * scale
        w = int(w)
        m =  np.array([[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
            [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]])
        align_eye = cv2.warpAffine(img,m,(w,w))

    return align_eye 

def pupil_fit(img,pupil,flag_list):
    #(Gray,Binary,Morphological,Gaussian blur,Sobel,Canny,Find contours)
    margin = 0
    input_eye_img = img[pupil[1]-margin:pupil[3]+margin,pupil[0]-margin:pupil[2]+margin,:]
    elPupilThresh = find_max_Thresh(input_eye_img,flag_list)
    if elPupilThresh != None:
        # update elPupilThresh into golbal image
        center = (int(elPupilThresh[0][0] + pupil[0]), int(elPupilThresh[0][1] + pupil[1]))
        new_elPupilThresh_left = (center,elPupilThresh[1],elPupilThresh[2])
        cv2.ellipse(img, new_elPupilThresh_left, (0, 255, 0), 2)
        cv2.circle(img, center, 3, (0, 0, 255), -1)
        # resize image into top left corner
        return center
    else:
        return 0

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        coordinates.append((x,y))
        # cv2.circle(image,(x,y),100,(255,0,0), -1)
        print(f"Mouse clicked at ({x}, {y})")

def loop_and_detect(cam, trt_yolo, mtcnn, conf_th, vis):
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

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # Self-define global parameter
    # ---------------------------
    # ------ face imformation ------
    nose_center_point = (0,0)
    mouse_center_point = (0,0)
    left_center = (0,0)
    right_center = (0,0)
    eye_w_roi = 100
    eye_h_roi = 50
    face_roi = 200
    driver_face_roi = [0,cam.img_width,0,cam.img_height]
    #------ put txt ------
    base_txt_height = 35
    gap_txt_height = 35
    len_width = 400
    #------ eye img ------
    right_eye_img = cv2.imread("./test_image/eye/3.png")  
    right_eye_img = cv2.resize(right_eye_img,(eye_w_roi,eye_h_roi))
    left_eye_img = cv2.imread("./test_image/eye/2.png")  
    left_eye_img = cv2.resize(left_eye_img,(eye_w_roi,eye_h_roi))
    
    #------ record/save ------
    save_cnt = 0
    start_record_f = 0
    save_video_cnt = 0
    #------ frame conut ------
    frame_cnt = 0
    # ---------------------------

    print("")
    print("-------------------------------")
    print("------------ Start ------------")
    print("-------------------------------")
    print("")

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

        if coordinates != []:
            print("right_center",left_center)
            print("error right center:",abs(coordinates[0][0]-right_center[0]),abs(coordinates[0][1]-right_center[1]))
            coordinates.pop()

        # end my self code
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

        if(start_record_f):
            out.write(img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            print("")
            print("-------------------------------")
            print("------ See You Next Time ------")
            print("-------------------------------")
            print("")
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

        # write a img when press buttom r
        elif key == ord('s') or key == ord('S'):
            save_path = cam.args.save_img+str(save_cnt)+".jpg"
            cv2.imwrite(save_path,img)
            print("Save img:",save_cnt)
            save_cnt += 1
        elif (key == ord('r') or key == ord('R')) and not start_record_f :
            start_record_f = 1
            save_video_path = cam.args.save_record+str(save_video_cnt)+".avi"
            out = cv2.VideoWriter(save_video_path,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280,722))
            print("Start record")
        elif (key == ord('e') or key == ord('E')) and start_record_f:
            start_record_f = 0
            save_video_cnt += 1
            out.release()
            print("End record")

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
    
    # add MTCNN detector
    mtcnn = TrtMtcnn()

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, mtcnn, args.conf_thresh, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
