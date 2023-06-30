domains=$1

for split in train validation test;
do
    SPLIT_PATH="/share/edc/home/alon_albalak/data/pile/${split}/"
    for DATASET_NAME in ${domains};
    do
        DATASET_PATH="${SPLIT_PATH}${DATASET_NAME}.jsonl"
        echo "path: dataset path: ${DATASET_PATH}"
        echo "name: dataset name: ${DATASET_NAME}"
        echo "outputting to: /share/edc/home/alon_albalak/data/pile/preprocessed/${DATASET_NAME}"

        OUTPUT_DIR=/share/edc/home/alon_albalak/data/pile/preprocessed/$split/${DATASET_NAME}
        mkdir -p ${OUTPUT_DIR}

        python tools/preprocess_data.py \
            --input $DATASET_PATH \
            --output-prefix ${OUTPUT_DIR}/${DATASET_NAME} \
            --vocab-file /share/edc/home/alon_albalak/tokenizers/20B_tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --workers 24 \
            --append-eod 2>&1 | tee ${OUTPUT_DIR}.log
    done
done