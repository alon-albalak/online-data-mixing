SHARE_DIR=/share/edc/home/alon_albalak/data/pile/debug
mkdir -p ${SHARE_DIR}

for split in train validation test; do
    mkdir -p ${SHARE_DIR}/$split
    mkdir -p ${SHARE_DIR}/preprocessed/$split
    for DATASET_FILE in $(ls /share/edc/home/alon_albalak/data/pile/$split); do
        DATASET_NAME=${DATASET_FILE::-6}
        echo ${DATASET_FILE}
        echo ${DATASET_NAME}
        head -n 200 /share/edc/home/alon_albalak/data/pile/$split/${DATASET_FILE} > ${SHARE_DIR}/$split/${DATASET_FILE}
        
        OUTPUT_DIR=${SHARE_DIR}/preprocessed/$split/${DATASET_NAME}
        mkdir -p ${OUTPUT_DIR}

        python tools/preprocess_data.py \
            --input ${SHARE_DIR}/$split/${DATASET_FILE} \
            --output-prefix ${OUTPUT_DIR}/${DATASET_NAME} \
            --vocab-file /share/edc/home/alon_albalak/tokenizers/20B_tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --append-eod
            # 2>&1 | tee /share/edc/home/alon_albalak/data/pile/preprocessed/${DATASET_NAME}.log
    done
done