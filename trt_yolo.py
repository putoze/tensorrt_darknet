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
from utils.fitEllipse import find_max_Thresh

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

    # Self-define global parameter
    # ---------------------------
    # ------ face imformation ------
    nose_center_point = (0,0)
    mouse_center_point = (0,0)
    left_center = (0,0)
    right_center = (0,0)
    eye_w_roi = 100
    eye_h_roi = 50
    #------ put txt ------
    base_txt_height = 25
    gap_txt_height = 30
    len_width = 400
    #------ eye img ------
    right_eye_img = cv2.imread("./test_image/3.png")  
    right_eye_img = cv2.resize(right_eye_img,(eye_w_roi,eye_h_roi))
    left_eye_img = cv2.imread("./test_image/2.png")  
    left_eye_img = cv2.resize(left_eye_img,(eye_w_roi,eye_h_roi))
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
        # write my self code
        #(img, text, org, fontFace, fontScale, color, thickness, lineType)
        # Write the user guide interface
        cv2.putText(img,"Esc: Quit",(cam.img_width-len_width,base_txt_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img,"F  : Full Screen",(cam.img_width-len_width,base_txt_height+gap_txt_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # ------- Main Algorithm ------
        bb_eye_list = []
        bb_pupil_list = []
        for bb, cf, cl in zip(boxes, confs, clss):
            if cl == 0:
                bb_eye_list.append(bb)
            if cl == 1:
                bb_pupil_list.append(bb)
            if cl == 2:
                nose_center_point = (bb[0]+(bb[2]-bb[0])/2,bb[1]+(bb[3]-bb[1])/2)
            if cl == 3:
                mouse_center_point = (bb[0]+(bb[2]-bb[0])/2,bb[1]+(bb[3]-bb[1])/2)

        # Sorted 
        # bb_eye_list = sorted(bb_eye_list, key=lambda x: x[0])
        # bb_pupil_list = sorted(bb_pupil_list, key=lambda x: x[0])

        # To find the eye roi
        flag_list = [1,1,1,1,1,1,1]
        for i in range(len(bb_eye_list)):
            x_min, y_min, x_max, y_max = bb_eye_list[i][0], \
                bb_eye_list[i][1], bb_eye_list[i][2], bb_eye_list[i][3]
            # center eye roi from yolo-detection
            cen_eye = (x_min+(x_max - x_min)/2,y_min+(y_max - y_min)/2)

            # print("number of eye index:",i)
            # print("center of class nose x:", nose_center_point[0])
            # print("center of class mouse x:", mouse_center_point[0])
            # print("center of class eye x:", cen_eye[0])

            # using nose and mouse center to determine left or right region of eye
            if(cen_eye[0] < nose_center_point[0] or cen_eye[0] < mouse_center_point[0]):
                left_eye_img = img[y_min:y_max,x_min:x_max,:]
                input_left_eye_img = left_eye_img
                # determine input image
                for i in range(len(bb_pupil_list)):
                    x1_min, y1_min, x1_max, y1_max = bb_pupil_list[i][0], \
                        bb_pupil_list[i][1], bb_pupil_list[i][2], bb_pupil_list[i][3]
                    cen_eye = (x1_min+(x1_max - x1_min)/2,y1_min+(y1_max - y1_min)/2)
                    # if center of pupil in the region of eye, then change input image
                    if(cen_eye[0] < x_max and cen_eye[0] > x_min 
                       and cen_eye[1] < y_max and cen_eye[1] > y_min):
                        input_left_eye_img = img[y1_min:y1_max,x1_min:x1_max,:]
                        x_min, y_min, x_max, y_max = x1_min, y1_min, x1_max, y1_max
                
                #(Gray,Binary,Morphological,Gaussian blur,Sobel,Canny,Find contours)
                elPupilThresh_left = find_max_Thresh(input_left_eye_img,flag_list)
                if elPupilThresh_left != None:
                    # update elPupilThresh into golbal image
                    center = (int(elPupilThresh_left[0][0] + x_min), int(elPupilThresh_left[0][1] + y_min))
                    new_elPupilThresh_left = (center,elPupilThresh_left[1],elPupilThresh_left[2])
                    cv2.ellipse(img, new_elPupilThresh_left, (0, 255, 0), 2)
                    cv2.circle(img, center, 3, (0, 0, 255), -1)
                    left_center = center
                # resize image into top left corner
                left_eye_img = cv2.resize(left_eye_img,(eye_w_roi,eye_h_roi))

            elif(cen_eye[0] > nose_center_point[0] or cen_eye[0] > mouse_center_point[0]):
                right_eye_img = img[y_min:y_max,x_min:x_max,:]
                input_right_eye_img = right_eye_img
                # determine input image
                for i in range(len(bb_pupil_list)):
                    x1_min, y1_min, x1_max, y1_max = bb_pupil_list[i][0], \
                        bb_pupil_list[i][1], bb_pupil_list[i][2], bb_pupil_list[i][3]
                    cen_eye = (x1_min+(x1_max - x1_min)/2,y1_min+(y1_max - y1_min)/2)
                    # if center of pupil in the region of eye, then change input image
                    if(cen_eye[0] < x_max and cen_eye[0] > x_min 
                       and cen_eye[1] < y_max and cen_eye[1] > y_min):
                        input_right_eye_img = img[y1_min:y1_max,x1_min:x1_max,:]
                        x_min, y_min, x_max, y_max = x1_min, y1_min, x1_max, y1_max
                
                #(Gray,Binary,Morphological,Gaussian blur,Sobel,Canny,Find contours)
                elPupilThresh_right = find_max_Thresh(input_right_eye_img,flag_list)
                if elPupilThresh_right != None:
                    # update elPupilThresh into golbal image
                    center = (int(elPupilThresh_right[0][0] + x_min), int(elPupilThresh_right[0][1] + y_min))
                    new_elPupilThresh_right = (center,elPupilThresh_right[1],elPupilThresh_right[2])
                    cv2.ellipse(img, new_elPupilThresh_right, (0, 255, 0), 2)
                    cv2.circle(img, center, 3, (0, 0, 255), -1)
                    right_center = center

                # resize image into top left corner
                right_eye_img = cv2.resize(right_eye_img,(eye_w_roi,eye_h_roi))

        # update eye image
        img[0:eye_h_roi,0:eye_w_roi,:] = left_eye_img
        img[0:eye_h_roi,eye_w_roi:2*eye_w_roi,:]  = right_eye_img

        content = "Left-center:("+str(left_center[0])+","+str(left_center[1])+")"
        # update center point
        cv2.putText(img,content,(cam.img_width-len_width,base_txt_height+2*gap_txt_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        content = "Right-center:("+str(right_center[0])+","+str(right_center[1])+")"
        cv2.putText(img,content,(cam.img_width-len_width,base_txt_height+3*gap_txt_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
