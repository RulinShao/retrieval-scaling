# Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore

Code and data for paper "Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore".

README still work in progress. Stay tuned!

## Overview
This repository supports:

1. Easy development and evaluation for retrieval-based language models (LMs). You can define different experiments by simply modifying one yaml configuration file.
2. Conduct efficient datastore scaling study with retrieval-based LMs.

## Installation

Install dependent Python libraries by running the command below.

```bash
git clone https://github.com/RulinShao/retrieval-scaling.git
cd retrieval-scaling
pip install -r requirements.txt
```

## Datastore Release
We release the full MassiveDS datastore and its 10% subsampled version in the HuggingFace Hub:

* **Full MassiveDS**: https://huggingface.co/datasets/rulins/MassiveDS-1.4T
* **10% Subsampled**: https://huggingface.co/datasets/rulins/MassiveDS-140B


# Quick Start
We support running scaling experiments with a unified hydra ymal configuration which is decomposed into stages. For example, you can either use the default configuration file [ric/conf/default.yaml](https://github.com/RulinShao/retrieval-scaling/blob/main/ric/conf/default.yaml) and pass self-defined arguments through command lines, or create a personalized configuration so you do not need to pass at run time. We provide example scripts to use the default configuration yaml file to run datastore construction and evaluation on DPR Wikipedia data.

To begin, we assume you downloaded the DPR Wikipedia data to path `DS_RAW_DATA`

## Example Scripts
### Encode Documents
```bash
datastore_domain=dpr_wiki
datastore_raw_data_path=$DS_RAW_DATA
num_shards=1

PYTHONPATH=.  python ric/main_ric.py \
  --config-name=default \
  tasks.datastore.embedding=true \
  datastore.domain=$datastore_domain \
  datastore.raw_data_path=$datastore_raw_data_path \
  datastore.embedding.num_shards=$num_shards \
  datastore.embedding.shard_ids=[0]
```

### Build Index
```bash
PYTHONPATH=.  python ric/main_ric.py \
  --config-name=default \
  tasks.datastore.index=true \
  datastore.domain=$datastore_domain \
  datastore.raw_data_path=$datastore_raw_data_path \
  datastore.embedding.num_shards=$num_shards \
  datastore.index.index_shard_ids=[[0]]

```

### Search Top-k Given a JSONL File
We show the search for perplexity evaluation for example. Assume the queries are saved in `$EVAL_DOMAIN.jsonl` in `EVAL_DATA_DIR`.

```bash
N_DOCS=100  # number of retrieved documents per query
N_EVAL_SAMPLES=10000  # maximum number of evaluation samples

PYTHONPATH=.  python ric/main_ric.py \
    --config-name largest_default \
    tasks.eval.task_name=perplexity \
    tasks.eval.search=true \
    datastore.domain=$datastore_domain \
    datastore.embedding.num_shards=$num_shards \
    datastore.embedding.shard_ids=[] \
    datastore.index.index_shard_ids=[[0]] \
    evaluation.domain=$EVAL_DOMAIN \
    evaluation.data.eval_data=$EVAL_DATA_DIR/$EVAL_DOMAIN.jsonl \
    evaluation.search.n_docs=$N_DOCS \
    evaluation.data.num_eval_samples=$N_EVAL_SAMPLES
```


### Evaluate Perplexity
```bash
MODEL_NAME=llama2-7b
MODEL=meta-llama/Llama-2-7b-hf
EVAL_FILE=$EVAL_DATA_DIR/$EVAL_DOMAIN.jsonl
LOG_DIR=out
mkdir -p $LOG_DIR
PYTHONPATH=.  python ric/main_ric.py \
    --config-name default \
    tasks.eval.task_name=perplexity \
    tasks.eval.inference=true \
    evaluation.search.merge_multi_source_results=true \
    evaluation.search.n_docs=$N_DOCS \
    evaluation.concate_k=3 \
    evaluation.domain=$EVAL_DOMAIN \
    evaluation.data.eval_data=$EVAL_FILE \
    datastore.domain=merged_0.1_subsampled \
    datastore.embedding.num_shards=8 \
    datastore.index.index_shard_ids=[] \
    evaluation.search.merged_path=${PATH_TO_MERGE_DIR}/${EVAL_DOMAIN}.jsonl \
    evaluation.results_only_log_file=$LOG_DIR/${MODEL_NAME}_ppl.log \
    model.lm_model=$MODEL \
    evaluation.decontamination=true \
    evaluation.contamination_threshold=32 \
    evaluation.decontamination_method=longest
```

# Evaluate Downstream Tasks Using LM-eval
Install our lm-eval repo if haven't:
```bash
git clone https://github.com/RulinShao/rag-evaluation-harness
cd RAG-evaluation-harness
pip install -e .
```
You can use this repo to prepare the retrieved documents, take the setting with a wikipeida datastore and the TriviaQA queries for example:
```bash
TASK_NAME=triviaqa
PYTHONPATH=/gscratch/zlab/rulins/Scaling  python ric/main_ric.py \
    --config-name default \
    datastore.index.index_shard_ids=[[0]] \
    evaluation.domain=$TASK_NAME \
    datastore.domain=dpr_wiki \
    datastore.embedding.num_shards=8
```
which will generate retrieved documents in `triviaqa_retrieved_results.jsonl`.

Then assume you have the retrieved documents from the Wikipedia datastore in `RETRIEVAL_FILE`. To evaluate Llama2-7B using VLLM:
```bash
TASK_NAME=triviaqa
LM_EVAL_DIR="lm-eval-data"
CONCAT_K=3
FEWSHOT_K=5
OUTPUT_DIR=out
RETRIEVAL_FILE=${TASK_NAME}_retrieved_results.jsonl

mkdir $OUTPUT_DIR
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $TASK_NAME \
    --batch_size auto \
    --inputs_save_dir $LM_EVAL_DIR \
    --retrieval_file $RETRIEVAL_FILE \
    --concat_k $CONCAT_K \
    --num_fewshot $FEWSHOT_K \
    --results_only_save_path $OUTPUT_DIR/$TASK_NAME-$CONCAT_K-docs-$FEWSHOT_K-shots.jsonl
```