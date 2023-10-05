MODEL_CONFIG=$1
EVAL_CONFIG=$2
NUM_FEWSHOT=${3:-0}
GPU=${4:-0}

# Can't use perplexity-based evaluation tasks with in-context examples
if [ ${NUM_FEWSHOT} -eq 0 ]; then
    EVAL_TASKS="lambada_openai piqa winogrande wsc arc_easy sciq logiqa wikitext openbookqa hendrycksTest-*"
else
    # if using few-shot, then we can't use wikitext
    EVAL_TASKS="lambada_openai piqa winogrande wsc arc_easy sciq logiqa openbookqa hendrycksTest-*"
fi
# Temporarily not using triviaqa because it can't download?
# Not using:
#   webqs (web questions) because our models have very poor performance (0.005 accuracy)
#   squad2 because it leads to: AttributeError: 'SequentialWrapper' object has no attribute 'clear_cache' 

# MODEL_CONFIG should be from the output of the training script
#   For example, gpt-neox/outputs/160m_doremi_280_seed42/global_step100000/configs/160m_doremi_280_seed42.yml
#       It should have additional fields. Eg.:
#         "load": "outputs/160m_doremi_280_seed42",
#         "eval_results_prefix": "outputs/160m_doremi_280_seed42",
# CONFIGS="outputs/160m_doremi_280_seed42/global_step100000/configs/160m_doremi_280_seed42.yml alon_configs/models/eval_160m_1gpu.yml"
python3 tools/create_eval_config.py --config_path ${MODEL_CONFIG} --num_fewshot ${NUM_FEWSHOT}

# if not using num_fewshot, then you can just use the following:
if [ ${NUM_FEWSHOT} -eq 0 ]; then
    # EVAL_MODEL_CONFIG is MODEL_CONFIG with .yml replaced by _eval.yml
    EVAL_MODEL_CONFIG=${MODEL_CONFIG%.yml}_eval.yml
else
    EVAL_MODEL_CONFIG=${MODEL_CONFIG%.yml}_eval_${NUM_FEWSHOT}shot.yml
fi

# Get GPU Config
GPU_CONFIG=alon_configs/gpu/gpu${GPU}.yml


# EVAL_CONFIG should be in the configs folder. See alon_configs/models/eval_160m_1gpu.yml for an example
python ./deepy.py evaluate.py ${EVAL_MODEL_CONFIG} ${EVAL_CONFIG} ${GPU_CONFIG} --eval_tasks ${EVAL_TASKS}