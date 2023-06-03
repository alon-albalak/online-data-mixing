RUN_NAME="current_run"
python3 deepy.py train.py configs/alon_config_small.yml 2>&1 | tee outputs/${RUN_NAME}.log
