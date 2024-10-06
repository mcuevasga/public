#!/bin/bash

# Parse command line options using getopts
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_dir) MODEL_DIR="$2"; shift ;;
        --gguf_file) GGUF_FILE="$2"; shift ;;
        --model_url) MODEL_CLONE_URL="$2"; shift ;;
        --gguf_url) GGUF_FILE_URL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate that all required options are provided
if [ -z "$MODEL_DIR" ] || [ -z "$GGUF_FILE" ] || [ -z "$MODEL_CLONE_URL" ] || [ -z "$GGUF_FILE_URL" ]; then
    echo "Usage: $0 --model_dir <model_dir> --gguf_file <gguf_file> --model_url <model_clone_url> --gguf_url <gguf_file_url>"
    exit 1
fi

# Check if the model directory exists and is not empty
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR")" ]; then
    echo "Cloning the TinyLlama model..."
    git lfs install
    sudo git clone "$MODEL_CLONE_URL" "$MODEL_DIR"
else
    echo "Model directory already exists and is not empty. Skipping clone."
fi

# Check if the .gguf file exists
if [ ! -f "$GGUF_FILE" ]; then
    echo "Downloading the GGUF file..."
    sudo curl -L -o "$GGUF_FILE" "$GGUF_FILE_URL"
else
    echo "GGUF file already exists. Skipping download."
fi

echo "Script execution completed."
