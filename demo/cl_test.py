import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys

import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings
import torch
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from torch.nn import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from PIL import Image  # 替换read_image


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True if args.segment_type in ["semantic"] else False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True if args.segment_type in ["instances"] else False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True if args.segment_type in ["panoptic"] else False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output1",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="building",
        help="building or car",
    )

    parser.add_argument(
        "--segment_type",
        type=str,
        default="semantic",
        help="semantic or panoptic or instances",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "experment/coco/model_final_f07440.pkl"],
        nargs=argparse.REMAINDER,
    )
    return parser


def preprocess_for_segment(img):
    original_image = img[:, :, ::-1]
    height, width = original_image.shape[:2]
    aug = T.ResizeShortestEdge([800, 800], 1333)
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    images = [image.to(device)]
    pixel_mean = torch.Tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    pixel_std = torch.Tensor([[[58.3950]], [[57.1200]], [[57.3750]]]).to(device)
    images = [(x - pixel_mean) / pixel_std for x in images]
    images = ImageList.from_tensors(images, 32)
    img_dict = {}
    img_dict['ori_sizes'] = torch.Tensor([int(height), int(width)])
    img_dict['image_sizes'] = torch.as_tensor(images.image_sizes[0])
    img_dict['tensor'] = torch.as_tensor(images.tensor)

    return [img_dict]


if __name__ == "__main__":
    device = torch.device("cuda")
    args = get_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    path_list = glob.glob(os.path.join(args.input, "*"))

    for path in tqdm.tqdm(path_list):
        img = Image.open(path).convert('RGB')
        img = np.asarray(img)
        img = img[:, :, ::-1]
        predictions, visualized_output = demo.run_on_image(img, args.segment_type, args.data_type)  # 官方推理

        if visualized_output is not None:
            visualized_output = ~visualized_output
            visualized_output = cv2.cvtColor(visualized_output, cv2.COLOR_GRAY2BGR)
            out = cv2.add(img, visualized_output)
            cv2.imwrite(os.path.join(args.output, os.path.basename(path)), out)






