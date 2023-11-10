!/bin/sh
echo ""
echo "Hello, choose the env you want it~"
echo [0]: yolov3_tenrt
echo ----------------
echo [1]: yolov7-gaze
echo ----------------
echo [n]: None
echo ----------------
echo -n "Press enter to start it:"

read ENV_Set

#============================================================================
if [ $ENV_Set -eq 0 ] ; then
    source activate
    conda activate yolov3_tenrt

    echo ============
    echo 「Success Enter yolov3_tenrt」
    echo ============ 
fi

#============================================================================
if [ $ENV_Set -eq 1 ] ; then
    source activate
    conda activate yolov7-gaze

    echo ============
    echo 「Success Enter yolov7-gaze」
    echo ============ 
fi

#============================================================================
echo ""
echo "Hello, choose the mode you want it~"
echo ------ Tensorrt Demo ------
echo [0]: otocam  yolov3-tiny-mid_eyetracker0808
echo ----------------
echo [1]: webcam  yolov3-tiny-mid_eyetracker0808
echo ----------------
echo [2]: img  yolov3-tiny-mid_eyetracker0808
echo ----------------
echo [3]: otocam  yolov3-tiny-mid-track-owl
echo ----------------
echo [4]: otocam mtcnn  
echo ----------------
echo [5]: Video yolov3-tiny-mid_eyetracker0808  
echo ----------------
echo [6]: otocam tensorrt demo with yolov3-tiny-mid_eyetracker 9cs 
echo ----------------
echo [7]: otocam tensorrt demo with yolov4-tiny-custom 9cs 
echo ----------------
echo [8]: otocam tensorrt demo with yolov4-tiny-custom 4cs 
echo ----------------
echo [9]: Video tensorrt demo with yolov4-tiny-custom 4cs 
echo ----------------
echo [10]: eval_yolo 5cs
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
    echo img tensorrt demo with yolov3-tiny-mid_eyetracker0808」
    echo ============

    for filelist in ./test_image/frank/*.png;do
    
    python3 trt_yolo.py \
    -m ./mid-track0808/yolov3-tiny-mid_eyetracker0808 \
    --image $filelist \
    -t 0.98 #--width 1280 --height 722
    done
fi

#============================================================================ 

if [ $MY_mode -eq 3 ] ; then
    echo ============
    echo 「otocam tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============
    python3 trt_yolo.py \
    -m ./mid-track-owl/yolov3-tiny-mid_eyetracker \
    --gstr 1 -c 5 -t 0.98 #--width 1280 --height 722
fi

#============================================================================ 

if [ $MY_mode -eq 4 ] ; then
    echo ============
    echo 「otocam mtcnn」
    echo ============
    python3 trt_mtcnn.py --gstr 1

fi

#============================================================================ 

if [ $MY_mode -eq 5 ] ; then
    echo ============
    echo 「Video tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============
    python3 trt_yolo.py \
    -m ./mid-track-owl/yolov3-tiny-mid_eyetracker \
    --video ./../tensor_test.avi -c 5 -t 0.8 #--width 1280 --height 722

fi

#============================================================================ 

if [ $MY_mode -eq 6 ] ; then
    echo ============
    echo 「otocam tensorrt demo with yolov3-tiny-mid_eyetracker 9cs 」
    echo ============
    python3 trt_yolo.py \
    -m ../../weights/darknet/yolov3-tiny-20231005/yolov3-tiny-mid_eyetracker \
    --gstr 1 -c 9 -t 0.75 #--width 1280 --height 722

fi

#============================================================================ 

if [ $MY_mode -eq 7 ] ; then
    echo ============
    echo 「otocam tensorrt demo with yolov3-tiny-mid_eyetracker 9cs 」
    echo ============
    python3 trt_yolo.py \
    -m ../../weights/darknet/yolov4-tiny-20231005/yolov4-tiny-custom \
    --gstr 1 -c 9 -t 0.75 #--width 1280 --height 722

fi

#============================================================================ 

if [ $MY_mode -eq 8 ] ; then
    echo ============
    echo 「otocam tensorrt demo with yolov4-tiny-mid_eyetracker 1011 4cs 」
    echo ============
    python3 trt_yolo.py \
    -m ../../weights/darknet/yolov4-tiny-20231011-4cs/yolov4-tiny-custom \
    --gstr 0 -c 4 -t 0.75 #--width 1280 --height 722

fi


#============================================================================ 

if [ $MY_mode -eq 9 ] ; then
    echo ============
    echo 「Video tensorrt demo with yolov4-tiny-mid_eyetracker 1011 4cs 」
    echo ============
    python3 trt_yolo.py \
    -m ../../weights/darknet/yolov4-tiny-20231011-4cs/yolov4-tiny-custom \
    --video /media/joe/Xavierssd/2023_0816_otocam_datavideo/output12.avi \
    -c 4 -t 0.75 #--width 1280 --height 722

fi

#============================================================================ 


if [ $MY_mode -eq 10 ] ; then
    echo ============
    echo 「map tensorrt demo with yolov3-tiny-mid_eyetracker」
    echo ============
    python3 eval_yolo.py \
    -m ../../weights/darknet/yolov4-tiny-20231106-5cs/yolov4-tiny-custom-5cs \
    -c 5 \
    --non_coco \
    # --annotations custom_dataset_all.json \
    # --imgs_dir /media/joe/Xavierssd/first_years_5cs/data/test 

fi



#============================================================================ End
echo [===YOLO===] ok!


