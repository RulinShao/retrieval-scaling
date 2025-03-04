mkdir dpr_wiki
wget -O dpr_wiki/wiki_dpr.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/dpr_wiki/wiki_dpr.jsonl?download=true

mkdir math
wget -O math/gsm8k.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/gsm8k.jsonl?download=true
wget -O math/math.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/math.jsonl?download=true
wget -O math/naturalproofs.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/naturalproofs.jsonl?download=true
for i in {0..15}
do
  wget -O math/openwebmath_${i}.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/math/openwebmath_${i}.jsonl?download=true
done

mkdir pes2o
for i in {0..19}
do
  formatted_number=$(printf "%02d" $i)
  wget -O pes2o/train-000${formatted_number}-of-00020.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/pes2o/train-000${formatted_number}-of-00020.jsonl?download=true
done

mkdir pubmed
wget -O pubmed/pubmed.jsonl https://huggingface.co/datasets/rulins/raw_data/resolve/main/pubmed/pubmed.jsonl?download=true

mkdir redpajama_v1
SUBDIR="redpajama_v1/arxiv"
FILE_LIST=$(curl -s "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
for FILE in $FILE_LIST; do
    mkdir -p "$(dirname "$FILE")"  # Create directories if they don't exist
    wget -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
done

SUBDIR="redpajama_v1/book"
FILE_LIST=$(curl -s "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
for FILE in $FILE_LIST; do
    mkdir -p "$(dirname "$FILE")"  # Create directories if they don't exist
    wget -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
done

SUBDIR="redpajama_v1/c4"
FILE_LIST=$(curl -s "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
for FILE in $FILE_LIST; do
    mkdir -p "$(dirname "$FILE")"  # Create directories if they don't exist
    wget -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
done

SUBDIR="redpajama_v1/github"
FILE_LIST=$(curl -s "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
for FILE in $FILE_LIST; do
    mkdir -p "$(dirname "$FILE")"  # Create directories if they don't exist
    wget -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
done

SUBDIR="redpajama_v1/stackexchange"
FILE_LIST=$(curl -s "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
for FILE in $FILE_LIST; do
    mkdir -p "$(dirname "$FILE")"  # Create directories if they don't exist
    wget -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
done

SUBDIR="redpajama_v1/wikipedia"
FILE_LIST=$(curl -s "https://huggingface.co/api/datasets/rulins/raw_data/tree/main/${SUBDIR}" | jq -r '.[] | .path')
for FILE in $FILE_LIST; do
    mkdir -p "$(dirname "$FILE")"  # Create directories if they don't exist
    wget -O "$FILE" "https://huggingface.co/datasets/rulins/raw_data/resolve/main/${FILE}"
done
