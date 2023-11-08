import os
import json

# 指定图像和标签文件夹路径
# image_folder = '/media/joe/Xavierssd/first_years_5cs/data/test/images'
# label_folder = '/media/joe/Xavierssd/first_years_5cs/data/test/labels'
# dst = 'custom_dataset_coco_format.json'

# input_h = 1080
# input_w = 1920

image_folder = '/media/joe/Xavierssd/20231011_4cs_dataset/test'
label_folder = '/media/joe/Xavierssd/20231011_4cs_dataset/test'
# dst = 'custom_dataset_4cs_format.json'

input_h = 722
input_w = 1280

image_folder_2 = '/media/joe/Xavierssd/first_years_5cs/data/test/images'
label_folder_2 = '/media/joe/Xavierssd/first_years_5cs/data/test/labels'
dst = 'custom_dataset_all.json'

input_h_2 = 1080
input_w_2 = 1920

# 初始化COCO格式数据结构
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "mask"},
        {"id": 1, "name": "glasses"},
        {"id": 2, "name": "seatbelt"},
        {"id": 3, "name": "phone"},
        {"id": 4, "name": "smoke"}
    ]
}

image_id = 1
annotation_id = 1

def calculate_coordinates(x_center, y_center, box_width, box_height, image_width, image_height):
    x_min = (x_center - box_width / 2) * image_width
    y_min = (y_center - box_height / 2) * image_height
    x_max = (x_center + box_width / 2) * image_width
    y_max = (y_center + box_height / 2) * image_height

    return x_min, y_min, x_max, y_max


for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": input_w,  # Replace with the actual image width
            "height": input_h  # Replace with the actual image height
        }
        coco_data["images"].append(image_info)

        label_file = os.path.join(label_folder, filename.replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split()
                class_id, x_center, y_center, box_width, box_height = map(float, data)
                x_min, y_min, x_max, y_max = calculate_coordinates(x_center, y_center, box_width, box_height, input_w, input_h)
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),  # Use the YOLO class ID
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation_info)
                annotation_id += 1

        image_id += 1

for filename in os.listdir(image_folder_2):
    if filename.endswith('.jpg'):
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": input_w_2,  # Replace with the actual image width
            "height": input_h_2  # Replace with the actual image height
        }
        coco_data["images"].append(image_info)

        label_file = os.path.join(label_folder_2, filename.replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split()
                class_id, x_center, y_center, box_width, box_height = map(float, data)
                x_min, y_min, x_max, y_max = calculate_coordinates(x_center, y_center, box_width, box_height, input_w_2, input_h_2)
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),  # Use the YOLO class ID
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation_info)
                annotation_id += 1

        image_id += 1

# 保存COCO格式数据为JSON文件
with open(dst, 'w') as json_file:
    json.dump(coco_data, json_file)
