MODEL_CONFIG=$1
EVAL_CONFIG=$2

EVAL_TASKS="lambada_openai piqa winogrande wsc arc_easy sciq logiqa wikitext openbookqa"
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
python3 tools/create_eval_config.py --config_path ${MODEL_CONFIG}

# EVAL_MODEL_CONFIG is MODEL_CONFIG with .yml replaced by _eval.yml
EVAL_MODEL_CONFIG=${MODEL_CONFIG%.yml}_eval.yml

# EVAL_CONFIG should be in the configs folder. See alon_configs/models/eval_160m_1gpu.yml for an example
python ./deepy.py evaluate.py ${EVAL_MODEL_CONFIG} ${EVAL_CONFIG} --eval_tasks ${EVAL_TASKS}