#!/bin/sh

set -x

number_of_processes=10

for ((i=0;i<${number_of_processes};++i)); do
  python "dataset-tools/create_advise_tfrecord.py" \
    --image_data_path="raw_data/ads_trainval_images" \
    --number_of_processes=${number_of_processes} \
    --process_id=${i} \
    --tfrecord_output_path="data/ads_trainval.record" \
    --proposal_json_path="raw_data/ads_wsod_boxes.json/nmsed" \
    --proposal_feature_npy_path="raw_data/ads_wsod_boxes.json/nmsed_npys" \
    --statement_json_path="raw_data/statement.json" \
    --slogan_json_path="raw_data/slogan.json" \
#     > log/train_tfrecord.log 2>&1 &
#   
#  python "dataset-tools/create_advise_tfrecord.py" \
#    --image_data_path="raw_data/ads_test_images" \
#    --number_of_processes=${number_of_processes} \
#    --process_id=${i} \
#    --tfrecord_output_path="data/ads_test.record" \
#    --proposal_json_path="raw_data/ads_wsod_boxes.json/nmsed" \
#    --proposal_feature_npy_path="raw_data/ads_wsod_boxes.json/nmsed_npys" \
#    --statement_json_path="raw_data/statement.json" \
#    --slogan_json_path="raw_data/slogan.json" \
#    > log/test_tfrecord.log 2>&1 &
#
done

exit 0
