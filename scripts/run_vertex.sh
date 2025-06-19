#!/bin/bash

# Set your Google Cloud project and Vertex AI settings
# export GOOGLE_CLOUD_PROJECT="your-project-id"
# export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

VERTEX_MODEL="claude-3-5-sonnet-v2@20241022"
PROJECT_ID=${GOOGLE_CLOUD_PROJECT}
LOCATION="us-central1"

PAPER_NAME="Transformer"
PDF_PATH="../examples/Transformer.pdf" # .pdf
PDF_JSON_PATH="../examples/Transformer.json" # .json
PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/Transformer_vertex"
OUTPUT_REPO_DIR="../outputs/Transformer_vertex_repo"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo "🚀 Starting PaperCoder with Vertex AI Claude"
echo "📊 Project: $PROJECT_ID"
echo "🌍 Location: $LOCATION"
echo "🤖 Model: $VERTEX_MODEL"
echo "📄 Paper: $PAPER_NAME"

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \

echo "------- PaperCoder with Vertex AI Claude -------"

python ../codes/1_planning_vertex.py \
    --paper_name $PAPER_NAME \
    --vertex_model ${VERTEX_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --project_id ${PROJECT_ID} \
    --location ${LOCATION} \
    --max_retries 5 \
    --base_delay 60

python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing_vertex.py \
    --paper_name $PAPER_NAME \
    --vertex_model ${VERTEX_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --project_id ${PROJECT_ID} \
    --location ${LOCATION} \
    --max_retries 5 \
    --base_delay 60

python ../codes/3_coding_vertex.py  \
    --paper_name $PAPER_NAME \
    --vertex_model ${VERTEX_MODEL} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
    --project_id ${PROJECT_ID} \
    --location ${LOCATION} \
    --max_retries 5 \
    --base_delay 60

echo "🎉 PaperCoder pipeline completed!"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📁 Generated repository: ${OUTPUT_REPO_DIR}"
