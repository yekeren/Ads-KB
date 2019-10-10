#!/bin/sh

set -o errexit
set -o nounset
set -x

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME=$1

PIPELINE_CONFIG_PATH="configs.advise/${NAME}.pbtxt"
NAME="${NAME}"
MODEL_DIR="logs/${NAME}"
SAVED_CKPTS_DIR="logs/${NAME}/saved"
RESULT_DIR="advise_results"
MAX_STEPS=20000

mkdir -p ${RESULT_DIR}
mkdir -p "raw_data/visl.${NAME}.json"

export CUDA_VISIBLE_DEVICES=$2
python train/trainer_main.py \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --max_steps=${MAX_STEPS}

#python train/predict_advise.py \
#  --run_once=true \
#  --alsologtostderr \
#  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
#  --model_dir="${MODEL_DIR}" \
#  --input_pattern="/own_files/yekeren/GCN/wsod_reason_test.*" \
#  --eval_log_dir="${MODEL_DIR}/eval_log" \
#  --qa_json_path="raw_data/reason.json" \
#  --json_output_path="raw_data/visl.${NAME}.json" \
#  --prediction_output_path="${RESULT_DIR}/${NAME}.json"

python train/predict_advise.py \
  --run_once=true \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --eval_log_dir="${MODEL_DIR}/eval_log" \
  --qa_json_path="raw_data/qa.json" \
  --input_pattern="output.advise/wsod_kb18519_test.record*" \
  --json_output_path="raw_data/visl.${NAME}.json" \
  --prediction_output_path="${RESULT_DIR}/${NAME}.json"


exit 0
