#!/usr/bin/env python3
# 功能批量将多个同类mask 转单个json

import datetime
import json
import os
import io
import re
import fnmatch
import json
from PIL import Image
import numpy as np
from PIL import Image
from skimage import measure


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

for i in range(1):
    coco_output = {
        "version": "5.0.1",
        "flags": {},
        "shapes": "",
    }

    class_id = 1
    image = Image.open("/home/cyq/Work/Mask2Former/image/car.png")
    binary_mask = np.asarray(Image.open("/home/cyq/Work/Mask2Former(复件)/demo/car1.jpg")
                             .convert('1')).astype(np.uint8)

    segmentation = binary_mask_to_polygon(binary_mask, tolerance=3)
    # 筛选多余的点集合
    for item in segmentation:
        if (len(item) > 10):
            list1 = []
            for i in range(0, len(item), 2):
                list1.append([item[i], item[i + 1]])

            label = "car"  #
            seg_info = {"label": label, 'points': list1, "group_id": None,
                        "shape_type": "polygon", "flags": {}}
            coco_output["shapes"] = [seg_info]
    coco_output["imagePath"] = "car.png"
    coco_output["imageData"] = None
    coco_output["imageHeight"] = binary_mask.shape[0]
    coco_output["imageWidth"] = binary_mask.shape[1]

    with open("/home/cyq/图片/11/car.json", 'w') as f:
        json.dump(coco_output, f, indent=2)

