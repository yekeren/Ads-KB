#!/bin/sh

set -o errexit
set -o nounset
set -x

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

# python "dataset-tools/expand_ads_knowledge.py" \
#   --alsologtostderr \
#   --ocr_detection_dir="raw_data/ocr.json/" \
#   --dbpedia_dir="dbpedia-data/"

top_k_list=(1 2 3 5)
for top_k in ${top_k_list[@]}; do
  python "dataset-tools/index_ads_knowledge.py" \
    --dbpedia_dir="dbpedia-data/" \
    --query_to_id_file="sparql_query2id_top${top_k}.txt" \
    --id_to_comment_file="sparql_id2comment_top${top_k}.txt" \
    --top_k_entries="${top_k}"
done
