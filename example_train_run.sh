CONFIGS="alon_configs/data/pile_v2.yml alon_configs/models/160.yml alon_configs/init.yml alon_configs/optimizer.yml alon_configs/parallelism.yml"
echo "Running with configs: ${CONFIGS}"

RUN_NAME="current_run"
python3 deepy.py train.py ${CONFIGS} 2>&1 | tee outputs/${RUN_NAME}.log
