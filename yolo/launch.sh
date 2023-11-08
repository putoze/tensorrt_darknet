!/bin/sh
echo ""
echo "Hello, choose the env you want it~"
echo [0]: yolov3_tenrt
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

echo ""
echo "Hello, choose the mode you want it~"
echo ------ Demo ------
echo [0]: yolo_to_onnx_to_ten
echo ----------------
echo -n "Press enter to start it:"

read MY_mode

#============================================================================ 

if [ $MY_mode -eq 0 ] ; then
    echo ============
    echo 「yolo_to_ten」
    echo ============

    python3 yolo_to_onnx.py \
    -m ../../weights/darknet/yolov4-tiny-20231106-5cs/yolov4-tiny-custom-5cs

    python3 onnx_to_tensorrt.py \
    -m ../../weights/darknet/yolov4-tiny-20231106-5cs/yolov4-tiny-custom-5cs
fi


#============================================================================ End
echo [===YOLO===] ok!


