#!/bin/bash
set -euo pipefail

# usage function
usage() {
    cat <<'EOF'
Usage: run.sh -k <DASHSCOPE_API_KEY>

Options:
  -k, --key     API key to export as DASHSCOPE_API_KEY
  -h, --help    Show this help text

You may also set the DASHSCOPE_API_KEY environment variable before invoking
this script instead of passing -k/--key.
EOF
}

API_KEY="${DASHSCOPE_API_KEY:-}"

# parse arguments
while (($#)); do
    case "$1" in
        -k|--key)
            API_KEY="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "$API_KEY" ]]; then
    echo "Error: DASHSCOPE_API_KEY is required (pass with -k/--key or pre-set env var)." >&2
    exit 1
fi

export DASHSCOPE_API_KEY="$API_KEY"
trap 'unset -v DASHSCOPE_API_KEY' EXIT

VLM_MODEL_NAME="qwen3-vl-flash"
MODEL_NAME="qwen-plus-2025-09-11"
CODER_MODEL_NAME="qwen3-coder-480b-a35b"

INPUT_PATH="/Users/v1p0r/Documents/DDL_Fall25/term_project/protopy_agent/data/input/latex/arXiv-2510.14980v1"
OUTPUT_PATH="/Users/v1p0r/Documents/DDL_Fall25/term_project/protopy_agent/data/output/arXiv-2510.14980v1"

echo
echo "--------------------------- ProtoPy Agent ---------------------------"
PAPER_NAME="$(basename "$INPUT_PATH")"
printf '\n[%s] Processing paper: %s (model: %s)\n' "$(date +%H:%M:%S)" "$PAPER_NAME" "$MODEL_NAME"

# echo
# echo "************************** 0_preprocessing ***************************"
# python 0_preprocessing.py \
#     --paper_name $PAPER_NAME \
#     --input_path $INPUT_PATH \
#     --output_path $OUTPUT_PATH
# echo

# echo "********************* 1_image_caption_analysis *********************"
# python 1_image_caption_analysis.py \
#     --paper_name $PAPER_NAME \
#     --model_name $VLM_MODEL_NAME \
#     --key $DASHSCOPE_API_KEY \
#     --input_path $INPUT_PATH \
#     --output_path $OUTPUT_PATH
# echo

# echo "*************************** 2_planning *****************************"
# python 2_planning.py \
#     --paper_name $PAPER_NAME \
#     --model_name $MODEL_NAME \
#     --key $DASHSCOPE_API_KEY \
#     --input_path $INPUT_PATH \
#     --output_path $OUTPUT_PATH
# echo

# python 2.1_extract_config.py \
#     --paper_name $PAPER_NAME \
#     --output_dir ${OUTPUT_PATH}

# cp -rp ${OUTPUT_PATH}/planning_config.yaml ${OUTPUT_PATH}/codebase/config.yaml

# echo "*************************** 3_analyzing *****************************"
# python 3_analyzing.py \
#     --paper_name $PAPER_NAME \
#     --model_name $MODEL_NAME \
#     --output_path $OUTPUT_PATH
# echo

echo "*************************** 4_coding *****************************"
python 4_coding.py \
    --paper_name $PAPER_NAME \
    --model_name $MODEL_NAME \
    --output_path $OUTPUT_PATH