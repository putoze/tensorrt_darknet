"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
Cython wrapped TensorRT optimized MTCNN engine.
"""

import time
import argparse

import cv2
import numpy as np
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn


WINDOW_NAME = 'TrtMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
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


def loop_and_detect(cam, mtcnn, minsize):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()

    # Self-define global parameter
    # ---------------------------
    #------ record/save ------
    save_cnt = 0
    start_record_f = 0
    save_video_cnt = 0
    # ---------------------------
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            # print('{} face(s) found'.format(len(dets)))
            if len(dets) != 0:
                align_eye = affineMatrix_eye(img, dets, landmarks)
                align_eye = cv2.resize(align_eye,(100,100))
                # print(align_face,align_face.shape)
                img[0:100,0:100:] = align_eye

            # print('{} face(s) found'.format(len(dets)))
            img = show_faces(img, dets, landmarks)
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
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    mtcnn = TrtMtcnn()

    open_window(
        WINDOW_NAME, 'Camera TensorRT MTCNN Demo for Jetson Nano',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, mtcnn, args.minsize)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
