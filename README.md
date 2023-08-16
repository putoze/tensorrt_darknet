# tensorrt_demos

Reference
----------------- 
[jkjung-avt-tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)

## Requirement

```
appdirs==1.4.4
certifi==2023.7.22
Cython==3.0.0
dataclasses==0.8
decorator==5.1.1
keras==2.10.0
Mako==1.1.6
MarkupSafe==2.0.1
mtcnn==0.1.1
numpy==1.19.4
onnx==1.9.0
opencv-python==4.8.0.76
platformdirs==2.4.0
protobuf==3.19.6
pycuda==2020.1
pytools==2022.1.12
requests==2.31.0
six==1.16.0
tensorrt==8.0.1.6
torchvision==0.12.0
typing==3.7.4.3
typing_extensions==4.1.1
```

## Enviroment

- Xavier AGX JetPack-4.6
- Yolov3-tiny(my own dataset training from AlexAB darknet)
- python 3.6.9
- opencv-python==4.8.0.76
- otocam 250

## Step to run

```
cd Camera Driver_oToCAM250_Ver_0.02
sh command_gstreamer.sh
cd tensorrt-demo
conda activate yolov3_tenrt
./lauch_tensorrt.sh
```
```
"Hello, choose the mode you want it~"
------ Tensorrt Demo ------
[0]: otocam  yolov3-tiny-mid_eyetracker0808
----------------
[1]: webcam  yolov3-tiny-mid_eyetracker0808
----------------
[2]: img  yolov3-tiny-mid_eyetracker0808
----------------
[3]: otocam  yolov3-tiny-mid-track-owl
----------------
[4]: otocam mtcnn  
----------------
"Press enter to start it:"
```
```
buttom R: Start record
buttom E: End record
buttom S: Save img
buttom F: Full screen
buttom ESC: Quit
```

