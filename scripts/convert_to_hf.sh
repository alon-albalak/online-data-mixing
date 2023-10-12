METHOD=$1
STEP=$2
MODEL_CONFIG=$3

python3 tools/convert_sequential_to_hf.py \
    --input_dir outputs/$METHOD/global_step${STEP} \
    --config_file $MODEL_CONFIG \
    --output_dir outputs/$METHOD/global_step${STEP}/hf_model