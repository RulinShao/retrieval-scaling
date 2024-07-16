# Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore

Code and data for paper "Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore".

[[Website](https://retrievalscaling.github.io)][[Paper](https://drive.google.com/file/d/1FDtWyJTwgyk-CRg6L8Syi6WEuBZrgytE/view)]

**Datastores:** [🤗 MassiveDS-1.4T](https://huggingface.co/datasets/rulins/MassiveDS-1.4T) | [🤗 MassiveDS-140B](https://huggingface.co/datasets/rulins/MassiveDS-140B)


<img src=images/scaling_gif.gif width="666" alt="Scaling overview.">

# Overview
This codebase contains:

1. Easy development and evaluation for retrieval-based language models (LMs)---run all experiments with one YAML file ([Quick Start](#quick-start)).
2. Our efficient MassiveDS pipeline for affordable datastore scaling study with retrieval-based LMs ([Advanced Usage](#advanced-usage)).
3. A comprehensive evaluation suite for retrieval-based LMs ([Evaluation](#evaluation)| [RAG-Evaluation-Harnesses](https://github.com/RulinShao/RAG-evaluation-harnesses)).


## Installation
Install dependent Python libraries by running the command below.

```bash
# clone the repo
git clone https://github.com/RulinShao/retrieval-scaling.git
cd retrieval-scaling

# create a new conda environment and install dependencies
conda env create -f environment.yml
conda activate scaling
```


## Quick Start
For a quick start, we provide a script that constructs a datastore using data in [FineWeb-Edu-1MT](https://huggingface.co/datasets/rulins/FineWeb-Edu-1MT) and evaluates it with LM [Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b). 

**Download example data**

To start, downloaded the example data and save it in `raw_data/`.

```bash
mkdir -p raw_data
wget -O raw_data/fineweb-edu-1m.jsonl https://huggingface.co/datasets/rulins/FineWeb-Edu-1MT/resolve/main/fineweb-edu-1M.jsonl?download=true
```

**Construct a Datastore**

Below command constructs a datastore using [Contriever-MSMACRO](https://huggingface.co/facebook/contriever-msmarco) as the retriever.
You can set the retriever to others that are supported in HuggingFace or SentenceTransformers through `model.datastore_encoder`. We also support sparse retriever BM25, to use which, pass `model.sparse_retriever=bm25`. 

```bash
PYTHONPATH=.  python ric/main_ric.py --config-name example_config
```
<!-- 1B token: 3518123 passages; 47 minutes; -->
<!-- Note: the datastore construction takes X minutes on 1 L40 GPU. If you want to quickly go through the code, please parallelize the job or further subsample the raw data. -->

**Evaluate Perplexity**

Next, we provide an example script to evaluate the perplexity on C4 data. You can use your own eval data by setting `evaluation.data.eval_data` to the path to your own JSONL file. 
```bash
PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.eval.task_name=perplexity \
  tasks.eval.search=true \
  tasks.eval.inference=true
```

The evaluation result will be printed in terminal and saved in `scaling_out/test_c4_ppl.log`.

**Evaluate Downstream Task**

We adapted [lm-evaluation-harnesses](https://github.com/EleutherAI/lm-evaluation-harness) to support RAG evaluation, which we developped in [RAG-evaluation-harnesses](https://github.com/RulinShao/RAG-evaluation-harnesses). We refer to [Downstream Evaluation](#downstream-evaluation) for more details. Below is an example to run evaluation on Natural Questions. 

First, install our RAG evaluation package.
```bash
git clone https://github.com/RulinShao/rag-evaluation-harness.git
pip install -e rag-evaluation-harness
```

Then, run search over the task quries.
```bash
lm_eval --tasks "nq_open" --inputs_save_dir "examples" --save_inputs_only

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.eval.task_name=lm-eval \
  tasks.eval.search=true \
  evaluation.domain=nq_open \
  evaluation.data.eval_data=examples/nq_open.jsonl
```

Finally, evaluate with a LM.
```bash
RETRIEVED_FILE=scaling_out/retrieved_results/facebook/contriever-msmarco/fineweb_edu_1m_datastore-256_chunk_size-1of1_shards/top_3/0/nq_open_retrieved_results.jsonl  # where retrieved documents are saved
lm_eval --model hf \
  --model_args pretrained="EleutherAI/pythia-1b" \
  --tasks nq_open \
  --batch_size auto \
  --inputs_save_dir examples \
  --retrieval_file $RETRIEVED_FILE \
  --concat_k 3 \
  --num_fewshot 5 \
  --results_only_save_path scaling_out/nq_open-5shots.jsonl
```
The evaluation results will be printed in a table in the terminal as saved in `scaling_out/nq_open-5shots.jsonl`.



## Datastore Release
We release the **data**, **embedding**, and **index** of our MassiveDS datastore, along with its 10% subsampled version, on HuggingFace:

* **10% Subsampled MassiveDS**: https://huggingface.co/datasets/rulins/MassiveDS-140B
* **Full MassiveDS (uploading)**: https://huggingface.co/datasets/rulins/MassiveDS-1.4T



# Advanced Usage
We provide more details of advanced usage of our database below.

## Content
1. [Model Configuration](#model-configuration)
2. [Datastore Configuration](#datastore-configuration)
3. [Distributed Datastore Construction](#distributed-datastore-construction)
4. [Document Retrieval](#document-retrieval)
5. [Data Filtering](#data-filtering)
6. [Subsampling](#subsampling)
7. [Evaluation](#evaluation)

## Model Configuration
### Retriever
### LM

## Datastore Configuration

## Distributed Datastore Construction

## Document Retrieval

## Data Filtering
### Decontamination
### Deduplication

## Subsampling

## Evaluation
### Perplexity Evaluation
### Downstream Evaluation