set -x




python demo/demo.py \
    --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
    --input image/car.png \
    --output output \
    --opts MODEL.WEIGHTS experment/coco/model_final_f07440.pkl