MODEL_CONFIG=$1
EVAL_CONFIG=$2

EVAL_TASKS="lambada_openai piqa winogrande arc_easy sciq wikitext openbookqa"

# MODEL_CONFIG should be from the output of the training script
#   For example, gpt-neox/outputs/160m_doremi_280_seed42/global_step100000/configs/160m_doremi_280_seed42.yml
#       It should have additional fields. Eg.:
#         "load": "outputs/160m_doremi_280_seed42",
#         "eval_results_prefix": "outputs/160m_doremi_280_seed42",
# CONFIGS="outputs/160m_doremi_280_seed42/global_step100000/configs/160m_doremi_280_seed42.yml alon_configs/models/eval_160m_1gpu.yml"

# EVAL_CONFIG should be in the configs folder. See alon_configs/models/eval_160m_1gpu.yml for an example
python ./deepy.py evaluate.py ${MODEL_CONFIG} ${EVAL_CONFIG} --eval_tasks ${EVAL_TASKS}