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

from imutils import face_utils
import dlib
from threading import Thread

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

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

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

    xml = '../weights/landmark-model/haarcascade_frontalface_default.xml'
    model_name = '../weights/landmark-model/shape_predictor_5_face_landmarks.dat'

    detector = cv2.CascadeClassifier(xml)    #Faster but less accurate
    predictor = dlib.shape_predictor(model_name)

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 30
    YAWN_THRESH = 20
    alarm_status = False
    alarm_status2 = False
    saying = False
    COUNTER = 0

    print("")
    print("-------------------------------")
    print("------------ Start ------------")
    print("-------------------------------")
    print("")

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        gray = cam.read()

        if gray is None:
            break

        # auto select if the frame is gray or RGB
        if len(gray.shape) < 3:
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else :
            img = gray
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        #for rect in rects:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            print(rect)
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye [1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(img, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if alarm_status == False:
                        alarm_status = True
                        t = Thread(target=alarm, args=('wake up sir',))
                        t.deamon = True
                        t.start()

                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0
                alarm_status = False

            if (distance > YAWN_THRESH):
                    cv2.putText(img, "Yawn Alert", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if alarm_status2 == False and saying == False:
                        alarm_status2 = True
                        t = Thread(target=alarm, args=('take some fresh air sir',))
                        t.deamon = True
                        t.start()
            else:
                alarm_status2 = False

            cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # end my self code
        # ---------------------------------------------
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)

        """Draw fps number at down-right corner of the image."""
        toc = time.time()
        fps = 1.0 / (toc - tic)

        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)

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
