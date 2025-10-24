#!/bin/bash

# Create directories and download files with automatic redirect support
mkdir -p dpr_wiki
wget -L -O dpr_wiki/wiki_dpr.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/dpr_wiki/wiki_dpr.jsonl?download=true

mkdir -p math
wget -L -O math/gsm8k.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/gsm8k.jsonl?download=true
wget -L -O math/math.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/math.jsonl?download=true
wget -L -O math/naturalproofs.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/naturalproofs.jsonl?download=true

for i in {0..15}
do
  wget -L -O math/openwebmath_${i}.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/openwebmath_${i}.jsonl?download=true
done

mkdir -p pes2o
for i in {0..19}
do
  formatted_number=$(printf "%02d" $i)
  wget -L -O pes2o/train-000${formatted_number}-of-00020.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/pes2o/train-000${formatted_number}-of-00020.jsonl?download=true
done

mkdir -p pubmed
wget -L -O pubmed/pubmed.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/pubmed/pubmed.jsonl?download=true

mkdir -p redpajama_v1

# Function to download files from a subdirectory
download_subdir() {
    local SUBDIR=$1
    echo "Downloading from ${SUBDIR}..."
    FILE_LIST=$(curl -sL "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
    for FILE in $FILE_LIST; do
        mkdir -p "$(dirname "$FILE")"
        wget -L -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
    done
}

# Download all redpajama_v1 subdirectories
download_subdir "redpajama_v1/arxiv"
download_subdir "redpajama_v1/book"
download_subdir "redpajama_v1/c4"
download_subdir "redpajama_v1/github"
download_subdir "redpajama_v1/stackexchange"
download_subdir "redpajama_v1/wikipedia"
