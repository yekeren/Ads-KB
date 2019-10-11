#!/bin/sh

set -o errexit
set -o nounset
set -x

# Export PYTHONPATH to use tensorflow models.

export PYTHONPATH="`pwd`:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

NAME=$1

PIPELINE_CONFIG_PATH="configs/${NAME}.pbtxt"
MODEL_DIR="logs/${NAME}"
SAVED_CKPTS_DIR="logs/${NAME}/saved"
MAX_STEPS=20000

export CUDA_VISIBLE_DEVICES=$2
python train/trainer_main.py \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --max_steps=${MAX_STEPS}

RESULT_FILE="advise_results/${NAME}/predictions.json"
VISL_DIR="advise_results/${NAME}/visl"

mkdir -p "${VISL_DIR}"
python train/predict_advise.py \
  --run_once=true \
  --alsologtostderr \
  --pipeline_proto="${PIPELINE_CONFIG_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --eval_log_dir="${MODEL_DIR}/eval_log" \
  --qa_json_path="raw_data/statement.json" \
  --input_pattern="/own_files/yekeren/GCN/ads_test*" \
  --json_output_path="${VISL_DIR}" \
  --prediction_output_path="${RESULT_FILE}"


exit 0
