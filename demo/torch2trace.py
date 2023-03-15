import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
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
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="car",
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
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    return semseg


def preprocess_for_segment(img):
    original_image = img[:, :, ::-1]
    height, width = original_image.shape[:2]
    # aug = T.ResizeShortestEdge([800, 800], 800)
    # image = aug.get_transform(original_image).apply_image(original_image)
    pil_image = Image.fromarray(original_image)
    pil_image = pil_image.resize((800, 800), 2)
    image = np.asarray(pil_image)
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


def postprocess_for_segment(outputs):
    mask_cls_results = outputs["pred_logits"]
    mask_pred_results = outputs["pred_masks"]
    # upsample masks
    mask_pred_results = F.interpolate(
        mask_pred_results,
        size=(images[0]["tensor"].shape[-2], images[0]["tensor"].shape[-1]),
        mode="bilinear",
        align_corners=False,
    )
    del outputs

    processed_results = []
    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
        height = int(images[0]["ori_sizes"][0].item())
        width = int(images[0]["ori_sizes"][1].item())
        image_size = (images[0]["image_sizes"][0].item(), images[0]["image_sizes"][1].item())
        processed_results.append({})

        if model.sem_seg_postprocess_before_inference:
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            mask_cls_result = mask_cls_result.to(mask_pred_result)

        # semantic segmentation inference
        if model.semantic_on:
            r = retry_if_cuda_oom(semantic_inference)(mask_cls_result, mask_pred_result)
            if not model.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
            processed_results[-1]["sem_seg"] = r

        # panoptic segmentation inference
        # if model.panoptic_on:
        #     panoptic_r = retry_if_cuda_oom(panoptic_inference)(mask_cls_result, mask_pred_result)
        #     processed_results[-1]["panoptic_seg"] = panoptic_r
        #
        # # instance segmentation inference
        # if model.instance_on:
        #     instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result)
        #     processed_results[-1]["instances"] = instance_r
    return processed_results[-1]["sem_seg"]


if __name__ == "__main__":
    device = torch.device("cuda")
    args = get_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    path = "/home/cyq/Work/Mask2Former/image/car.png"
    # path = "/home/cyq/图片/11/111.jpg"
    img = Image.open(path).convert('RGB')
    img = np.asarray(img)
    img = img[:, :, ::-1]
    images = preprocess_for_segment(img)
    mask_cls_result = demo.run_on_image(img)  # 官方推理


    with torch.no_grad():
        # model loading
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model.requires_grad_(False).eval())
        checkpointer.load("/home/cyq/Work/Mask2Former/experment/coco/model_final_f07440.pkl")

        # pytorch inference
        # images = preprocess_for_segment(img)  # 数据预处理
        # features = model.backbone(images[0]["tensor"])  # 简化推理
        # outputs = model.sem_seg_head(features)
        # out1 = postprocess_for_segment(outputs)

        # trace saving and inference
        # traced_script_module = torch.jit.trace(model, ([images]))
        # traced_script_module.save("seg.pt")
        model_new = torch.jit.load("seg.pt", map_location='cuda')
        mask_cls_result1 = model_new(images)
    # print(mask_cls_result.equal(mask_cls_result1))
    print(mask_cls_result[0].equal(mask_cls_result1[0]))
    print(mask_cls_result[1].equal(mask_cls_result1[1]))
    # print(mask_cls_result[2].equal(mask_cls_result1[2]))
    # print(mask_cls_result[3].equal(mask_cls_result1[3]))



