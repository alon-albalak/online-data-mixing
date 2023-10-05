RUN_NAME=$1
STEP=$2

# evaluate 0-shot
bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step${STEP}/configs/${RUN_NAME}.yml alon_configs/models/eval_3B_1gpu.yml 0 2>&1 | tee outputs/${RUN_NAME}_${STEP}_eval.log &
# evaluate 1-shot through 5-shot
for i in {1..5}; do
    bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step${STEP}/configs/${RUN_NAME}.yml alon_configs/models/eval_3B_1gpu.yml ${i} ${i} 2>&1 | tee outputs/${RUN_NAME}_${STEP}_${i}shot_eval.log &
done
wait < <(jobs -p)