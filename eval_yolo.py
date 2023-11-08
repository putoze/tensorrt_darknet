"""eval_yolo.py

This script is for evaluating mAP (accuracy) of YOLO models.
"""


import os
import sys
import json
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar

from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import yolo_cls_to_ssd


VAL_IMGS_DIR = '/media/joe/Xavierssd/first_years_5cs/data/test/images'
VAL_ANNOTATIONS = 'custom_dataset_coco_format.json'

def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLO model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs_dir', type=str, default=VAL_IMGS_DIR,
        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument(
        '--annotations', type=str, default=VAL_ANNOTATIONS,
        help='groundtruth annotations [%s]' % VAL_ANNOTATIONS)
    parser.add_argument(
        '--non_coco', action='store_true',
        help='don\'t do coco class translation [False]')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)


def generate_results(trt_yolo, imgs_dir, jpgs, results_file, non_coco):
    """Run detection on each jpg and write results to file."""
    results = []
    image_id = 1
    for jpg in progressbar(jpgs):
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        boxes, confs, clss = trt_yolo.detect(img, conf_th=1e-2)
        for box, conf, cls in zip(boxes, confs, clss):
            x = float(box[0])
            y = float(box[1])
            w = float(box[2] - box[0] + 1)
            h = float(box[3] - box[1] + 1)
            cls = int(cls)
            cls = cls if non_coco else yolo_cls_to_ssd[cls]
            results.append({'image_id': image_id,
                            'category_id': cls,
                            'bbox': [x, y, w, h],
                            'score': float(conf)})
        image_id += 1

    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))


def main():
    args = parse_args()
    check_args(args)
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (%s.trt) not found!' % args.model)

    results_file = 'results.json'

    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(trt_yolo, args.imgs_dir, jpgs, results_file,
                     non_coco=args.non_coco)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    for category_id, category_name in cocoGt.cats.items():
        print(f'Class {category_name["name"]}:')
        i = cocoGt.getCatIds(category_name["name"])[0]  # 获取类别的索引
        print(f'AP: {cocoEval.stats[i]:.3f}')
        print(f'Recall: {cocoEval.stats[i + len(cocoGt.cats)]:.3f}')


if __name__ == '__main__':
    main()
