#!/bin/sh

echo
echo '************************************'
echo -n "YOLO之TensorRT加速實驗"
echo
echo  "AutoRun： Yolo-->Onnx-->TensorRT (5 min~)"
echo
echo  "請選擇模式1~3:"
echo  "1:全自動轉檔 Yolo-->Onnx-->TensorRT"
echo  "2:測試照片"
echo  "3:啟動攝影機"
echo  -n "請輸入1~3:"
read MY_mode

#============================================================================ auto 
if [ $MY_mode -eq 1 ] ; then
   echo -n "請輸入 Project_Name :"
   read MY_Project_Name
   echo
   echo '************* start ****************'

   echo '================='
   echo '1. yolo to onnx'
   echo '================='
   python3 yolo_to_onnx.py -m input/$MY_Project_Name

   echo -n "已轉換成onnx格式(Open Neural Network Exchange)，按任意鍵繼續..."
   read ok

   echo '================='
   echo '2. onnx to tensorRT'
   echo '================='
   python3 onnx_to_tensorrt.py -m input/$MY_Project_Name -v

   echo -n "已轉換成trt格式，按任意鍵進入DEMO模式(請先接攝影機)..."
   read ok
   echo '================='
   echo '3. test webcam...'
   echo '================='

   cd ..
   python3 trt_yolo.py -m input/$MY_Project_Name --usb 0 -c 1 

   echo '************** end *****************'
fi

#============================================================================ image
if [ $MY_mode -eq 2 ] ; then
    echo '================='
    echo 'test image...'
    echo '================='
    echo -n "請輸入 Project_Name :"
    read MY_Project_Name
    cd ..
    python3 trt_yolo.py -m input/$MY_Project_Name --image yolo/input/test.jpg
fi

#============================================================================ video
if [ $MY_mode -eq 3 ] ; then
    echo '================='
    echo 'test webcam (請先接攝影機)...'
    echo '================='
    echo -n "請輸入 Project_Name :"
    read MY_Project_Name
    cd ..
    python3 trt_yolo.py -m input/$MY_Project_Name --usb 0 
fi
