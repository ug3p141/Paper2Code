#!/bin/bash

# export ANTHROPIC_API_KEY=""

CLAUDE_MODEL="claude-sonnet-4-20250514"

#PAPER_NAME="Transformer"
#PDF_PATH="../examples/Transformer.pdf" # .pdf
#PDF_JSON_PATH="../examples/Transformer.json" # .json
#PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
#OUTPUT_DIR="../outputs/Transformer_claude"
#OUTPUT_REPO_DIR="../outputs/Transformer_claude_repo"

PAPER_NAME="CADRecode"
PDF_PATH="../examples/cadrecode.pdf" # .pdf
PDF_JSON_PATH="../examples/cadrecode.json" # .json
PDF_JSON_CLEANED_PATH="../examples/cadrecode_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/cadrecode_claude"
OUTPUT_REPO_DIR="../outputs/cadrecode_claude_repo"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \

echo "------- PaperCoder with Claude -------"


#python ../codes/1_planning_claude_improved.py \
python ../codes/1_planning_claude.py \
    --paper_name $PAPER_NAME \
    --claude_model ${CLAUDE_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}
#    --max_retries 5 \
#    --base_delay 60

python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing_claude.py \
    --paper_name $PAPER_NAME \
    --claude_model ${CLAUDE_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/3_coding_claude.py  \
    --paper_name $PAPER_NAME \
    --claude_model ${CLAUDE_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
