#!/bin/sh

set -x

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=2
python "dataset-tools/wsod_roi_to_npy.py" \
  --image_data_path="raw_data/ads_train_images" \
  --bounding_box_json_path="raw_data/ads_wsod_boxes.json/trainval/" \
  --nmsed_bounding_box_json_path="raw_data/ads_wsod_boxes.json/nmsed/" \
  >> log/wsod_roi_to_npy_trainval.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
python "dataset-tools/wsod_roi_to_npy.py" \
  --image_data_path="raw_data/ads_test_images" \
  --bounding_box_json_path="raw_data/ads_wsod_boxes.json/test/" \
  --nmsed_bounding_box_json_path="raw_data/ads_wsod_boxes.json/nmsed/" \
  --nmsed_roi_image_path="raw_data/ads_wsod_boxes.json/nmsed_rois" \
  > log/wsod_roi_to_npy_test.log 2>&1 &

exit 0
