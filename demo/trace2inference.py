import os
import cv2
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
from torch.nn import functional as F
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
stuff_classes = [k["name"] for k in COCO_CATEGORIES]
from skimage import measure
import json

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
def segment2json(img_path, binary_masks, type="car"):
    cur_json_dict = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
    }

    for idx, binary_mask in enumerate(binary_masks):
        print(binary_mask.shape)
        binary_mask = binary_mask.astype(np.uint8)
        segmentation = binary_mask_to_polygon(binary_mask, tolerance=3)
        # 筛选多余的点集合
        for item in segmentation:
            if (len(item) > 10):
                points = []
                for i in range(0, len(item), 2):
                    points.append([item[i], item[i + 1]])
                seg_info = {"label": type, 'points': points, "group_id": idx,
                            "shape_type": "polygon", "flags": {}}
                cur_json_dict["shapes"].append(seg_info)
    cur_json_dict["imagePath"] = os.path.basename(img_path)
    cur_json_dict["imageData"] = None
    cur_json_dict["imageHeight"] = binary_mask.shape[0]
    cur_json_dict["imageWidth"] = binary_mask.shape[1]

    with open('1.json', 'w') as f:
        json.dump(cur_json_dict, f, indent=2)

class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


def overlay_instances(
    boxes=None,
    labels=None,
    masks=None,
):
    num_instances = 0
    if boxes is not None:
        boxes = boxes.detach().numpy()
        num_instances = len(boxes)
    if masks is not None:
        if num_instances:
            assert len(masks) == num_instances
        else:
            num_instances = len(masks)
    if labels is not None:
        assert len(labels) == num_instances
    if num_instances == 0:
        return None

    areas = None
    if boxes is not None:
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    elif masks is not None:
        areas = np.asarray([x.area() for x in masks])

    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        boxes = boxes[sorted_idxs] if boxes is not None else None
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None

    binary_masks = []
    for i in range(num_instances):
        binary_masks.append(masks[i].astype("uint8"))

    return binary_masks

def preprocess_for_segment(original_image):
    height, width = original_image.shape[:2]
    pil_image = Image.fromarray(original_image)
    pil_image = pil_image.resize((800, 800), 2)
    image = np.asarray(pil_image)
    # image = cv2.resize(original_image, (800, 800))
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    pixel_mean = torch.Tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    pixel_std = torch.Tensor([[[58.3950]], [[57.1200]], [[57.3750]]]).to(device)
    image = (image.to(device) - pixel_mean) / pixel_std

    image_size = torch.as_tensor((image.shape[-2], image.shape[-1]), device=device)
    max_size = torch.stack([image_size]).max(0).values
    max_size = (max_size + (32 - 1)).div(32, rounding_mode="floor") * 32
    padding_size = [0, max_size[-1] - image.shape[-1], 0, max_size[-2] - image.shape[-2]]
    batched_imgs = F.pad(image, padding_size, value=0).unsqueeze_(0)

    img_dict = {}
    img_dict['ori_sizes'] = torch.as_tensor((height, width), device=device)
    img_dict['image_sizes'] = image_size
    img_dict['tensor'] = batched_imgs

    return [img_dict]


def postprocess_for_segment(mask_cls_results, mask_pred_results, images):
    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
        height = int(images[0]["ori_sizes"][0].item())
        width = int(images[0]["ori_sizes"][1].item())
        image_size = (images[0]["image_sizes"][0].item(), images[0]["image_sizes"][1].item())

        mask_pred_result = mask_pred_result[:, : image_size[0], : image_size[1]].expand(1, -1, -1, -1)
        mask_pred_result = F.interpolate(mask_pred_result, size=(height, width), mode="bilinear", align_corners=False)[0]
        mask_cls_result = mask_cls_result.to(mask_pred_result)

    return mask_cls_result, mask_pred_result


def semantic_inference(mask_cls, mask_pred):
    mask_cls, mask_pred = postprocess_for_segment(mask_cls, mask_pred, images)
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    sem_seg = semseg.argmax(dim=0).to("cpu")
    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.numpy()
    labels, areas = np.unique(sem_seg, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]
    binary_masks = []
    for label in filter(lambda l: l < len(stuff_classes), labels):
        binary_mask = (sem_seg == label).astype(np.uint8)
        text = stuff_classes[label]
        if "car" in text:
            binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
            binary_masks.append(binary_mask * 255)
    return binary_masks


def instance_inference(mask_cls, mask_pred):
    mask_cls, mask_pred = postprocess_for_segment(mask_cls, mask_pred, images)
    num_classes = 133
    num_queries = 200
    test_topk_per_image = 100

    image_size = mask_pred.shape[-2:]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // num_classes
    mask_pred = mask_pred[topk_indices]

    thing_dataset_id_to_contiguous_id = {}
    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
    if True:
        keep = torch.zeros_like(scores_per_image).bool()
        for i, lab in enumerate(labels_per_image):
            keep[i] = lab in thing_dataset_id_to_contiguous_id.values()

        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]

    result = {}
    result["pred_masks"] = (mask_pred > 0).float()
    result["pred_boxes"] = torch.zeros(mask_pred.size(0), 4)
    mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result["pred_masks"].flatten(1)).sum(1) / (
            result["pred_masks"].flatten(1).sum(1) + 1e-6)
    result["scores"] = scores_per_image * mask_scores_per_image
    result["pred_classes"] = labels_per_image

    return result


def instance_postprocess(result):
    predictions = result
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    boxes = predictions["pred_boxes"] if "pred_boxes" in predictions.keys() else None
    scores = predictions["scores"] if "scores" in predictions.keys() else None
    classes = predictions["pred_classes"].tolist() if "pred_classes" in predictions.keys() else None
    labels = _create_text_labels(classes, scores, thing_classes)

    if "pred_masks" in predictions.keys():
        masks = np.asarray(predictions["pred_masks"].to("cpu"))
        masks = [x.astype("uint8") for x in masks]
    else:
        masks = None

    areas = None
    if boxes is not None:
        boxes = boxes.detach().numpy()
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    elif masks is not None:
        areas = np.asarray([x.area() for x in masks])

    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None

    binary_masks = []
    for i in range(len(masks)):
        name = labels[i]  # 'car 98%'
        if "car" in name[0: name.index(" ")] and int(name[name.index(" ") + 1:-1]) > 90:
            binary_masks.append(masks[i].astype("uint8") * 255)
    return binary_masks


def panoptic_inference(mask_cls, mask_pred):
    mask_cls, mask_pred = postprocess_for_segment(mask_cls, mask_pred, images)
    num_classes = 133
    object_mask_threshold = 0.8
    overlap_threshold = 0.8
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    mask_pred = mask_pred.sigmoid()

    keep = labels.ne(num_classes) & (scores > object_mask_threshold)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]
    cur_mask_cls = mask_cls[keep]
    cur_mask_cls = cur_mask_cls[:, :-1]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    segments_info = []

    thing_dataset_id_to_contiguous_id = {}
    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
    current_segment_id = 0
    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        return panoptic_seg, segments_info
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_dataset_id_to_contiguous_id.values()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info


def panoptic_postprocess(panoptic_seg, segments_info):
    _seg = panoptic_seg.to("cpu")

    _sinfo = {s["id"]: s for s in segments_info}  # seg id -> seg info
    segment_ids, areas = torch.unique(_seg, sorted=True, return_counts=True)
    areas = areas.numpy()
    sorted_idxs = np.argsort(-areas)
    _seg_ids, _seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
    _seg_ids = _seg_ids.tolist()

    binary_masks = []
    for sid in _seg_ids:
        sinfos = _sinfo.get(sid)
        if sinfos is None:
            continue
        if sinfos['category_id'] == 2:
            binary_mask = (_seg == sid).numpy().astype(np.bool)
            binary_mask = binary_mask.astype("uint8")
            binary_masks.append(binary_mask * 255)
    return binary_masks


if __name__ == "__main__":
    device = torch.device("cuda")

    path = "/home/cyq/Work/Mask2Former/image/car.png"
    # path = "/home/cyq/下载/images.jpeg"
    img = Image.open(path).convert('RGB')
    img = np.asarray(img)

    with torch.no_grad():
        # pytorch inference
        images = preprocess_for_segment(img)  # 数据预处理
        # trace saving and inference
        model_new = torch.jit.load("seg.pt", map_location='cuda')
        seg_result = model_new(images)

        """语义分割"""
        seg = semantic_inference(seg_result[0], seg_result[1])
        segment2json(path, seg, type="car")

        """全景分割"""
        panoptic_seg, segments_info = panoptic_inference(seg_result[0], seg_result[1])
        pan = panoptic_postprocess(panoptic_seg, segments_info)

        """实例分割"""
        instances = instance_inference(seg_result[0], seg_result[1])
        ins = instance_postprocess(instances)

        for i in range(len(pan)):
            cv2.imwrite("car_seg.jpg", seg[0])
            cv2.imwrite("car_pan.jpg", pan[i])
            cv2.imwrite("car_ins.jpg", ins[i])






