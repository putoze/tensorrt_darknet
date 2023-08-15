#!/bin/sh
echo ""
echo "Hello, choose the mode you want it~"
echo ------ Tensorrt Demo ------
echo [0]: otocam  yolov3-tiny-mid_eyetracker0808
echo ----------------
echo [1]: webcam  yolov3-tiny-mid_eyetracker0808
echo ----------------
echo [2]: otocam  yolov3-tiny-mid-track-owl
echo ----------------
echo [3]: img  yolov3-tiny-mid-track-owl
echo ----------------
echo -n "Press enter to start it:"

read MY_mode

# [./darknet] --> [./home/lab716/Desktop/Rain/darknet/darknet]
#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「otocam tensorrt demo with yolov3-tiny-mid_eyetracker0808」
    echo ============
    python3 trt_yolo.py \
    -m ./mid-track0808/yolov3-tiny-mid_eyetracker0808 \
    --gstr 1 --save_img ./save_img/save_img \
    --save_record ./save_img/save_record \
    -c 5 -t 0.98 #--width 1280 --height 722
    
fi

#============================================================================ 
if [ $MY_mode -eq 1 ] ; then
    echo ============
    echo 「webcam tensorrt demo with yolov3-tiny-mid_eyetracker0808」
    echo ============
    python3 trt_yolo.py \
    -m ./mid-track0808/yolov3-tiny-mid_eyetracker0808 \
    --usb 8 -c 5 -t 0.98 #--width 1280 --height 722
fi

#============================================================================ 

if [ $MY_mode -eq 2 ] ; then
    echo ============
    echo 「otocam tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============
    python3 trt_yolo.py \
    -m ./mid-track-owl/yolov3-tiny-mid_eyetracker \
    --gstr 1 -c 5 -t 0.98 #--width 1280 --height 722
fi

#============================================================================ 

if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo img tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============

    for filelist in ./test_image/frank/*.png;do
    
    python3 trt_yolo.py \
    -m ./mid-track-owl/yolov3-tiny-mid_eyetracker \
    --image $filelist \
    -t 0.98 #--width 1280 --height 722
    done
fi
#============================================================================ End
echo [===YOLO===] ok!


