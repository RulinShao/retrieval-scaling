# Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore

Code and data for paper "Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore".

[[Website](https://retrievalscaling.github.io)][[Paper](https://retrievalscaling.github.io/paper.pdf)]

**Datastores:** [🤗 MassiveDS-1.4T](https://huggingface.co/datasets/rulins/MassiveDS-1.4T) | [🤗 MassiveDS-140B](https://huggingface.co/datasets/rulins/MassiveDS-140B)


<img src=images/scaling_gif.gif width="666" alt="Scaling overview.">

If you find our code, data, models, or the paper useful, please cite the paper:
```
@article{shao2024scaling,
  author = {Shao, Rulin and He, Jacqueline and Asai, Akari and Shi, Weijia and Dettmers, Tim and Min, Sewon and Zettlemoyer, Luke and Koh, Pang Wei},
  title  = {Scaling Retrieval-Based Language Models with a Trillion-Token Datastore},
  year   = {2024},
}
```

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

Finally, evaluate with an LM.
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
The evaluation results will be printed in a table and saved in `scaling_out/nq_open-5shots.jsonl`.



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
Retrieval-based LMs involve two models: the reader LM and the retriever.
In the below sections, we illustrate ways to configure the models.
### Retriever
The default configuration uses Contriever-MSMACRO as the retriever. Additionally, we support off-the-shelf retriever models implemented in [HuggingFace](https://huggingface.co/models) or [SentenceTransformers](https://sbert.net/). To change the dense retriever, we require the user to define the 4 arguments below properly, as some models may use different encoders for query and context:

```bash
model:
  datastore_tokenizer: ???
  query_tokenizer: ???
  query_encoder: ???
  datastore_encoder: ???
```

**HuggingFace Dense Retrievers**

You can use HuggingFace retrievers by passing its huggingface model name to the corresponding arguments. re_encoder: ???
```

For example, to use [DRAGON-RoBERTa](https://huggingface.co/facebook/dragon-roberta-query-encoder) as the retriever, set
```bash
model:
  datastore_tokenizer: facebook/dragon-roberta-query-encoder
  query_tokenizer: facebook/dragon-roberta-query-encoder
  query_encoder: facebook/dragon-roberta-query-encoder
  datastore_encoder: facebook/dragon-roberta-context-encoder
```

**SentenceTransformers Dense Retrievers**

Similarly, if you want to use a dense retriever implemented in sentence-transformers, you can pass its model name to the arguments in the same way, where the script will automatically identify which package to use.

For example, to use [GTR-T5-Base](https://huggingface.co/sentence-transformers/gtr-t5-base) as the retriever, set:

```bash
model:
  datastore_tokenizer: sentence-transformers/gtr-t5-base
  query_tokenizer: sentence-transformers/gtr-t5-base
  query_encoder: sentence-transformers/gtr-t5-base
  datastore_encoder: sentence-transformers/gtr-t5-base
```

**Sparse Retriever**

We also support BM25 as a sparse retriever. To use BM25, set
```bash
model:
  sparse_retriever: bm25
```
Note: once `model.sparse_retriever` is set to bm25, the arguments for dense retrievers will be ignored. 


### Reader LM
We support all HuggingFace decoder models as the reader LM. You can set the reader LM by passing the model name or path to `model.lm_model`:
```bash
model:
  lm_model: EleutherAI/pythia-1b
```
Note: the reader LM for RAG-Evaluation-Harnesses is passed to the `lm-eval` command separately. The LM defined here is only for perplexity evaluation.

## Datastore Configuration

**General Configuration**

There are 3 arguments for datastore general configuration:
```bash
datastore:
  domain: fineweb_edu_1m
  raw_data_path: raw_data/fineweb-edu-1m.jsonl
  chunk_size: 256
```
* `domain` is a string used for naming the paths to save the datastore; 
* `raw_data_path` provides the raw data that will be used to construct the datastore, which could be a single JSONL file or a directory of JSONL files; 
* `chunk_size` is used to split raw text into passages, where every passage will have no more than `chunk_size` natural words.

**Embedding Configuration**

We introduce several key arguments for embedding below. The others are either straightforward by its name or do not need to be changed in general usage.
```bash
datastore:
  embedding:
    shard_ids: [0]
    num_shards: 1
    keep_last_chunk: true
    passages_dir: scaling_out/passages/${datastore.domain}/${datastore.embedding.num_shards}-shards

    per_gpu_batch_size: 512

    prefix: "passages"
    embedding_dir: scaling_out/embeddings/${model.datastore_encoder}/${datastore.domain}/${datastore.embedding.num_shards}-shards
    use_saved_if_exists: true
```
* `num_shards` defines the number of shards that the raw data will be divided into for distributed datastore construction. 
* `shard_ids` speficies the IDs of shards that the current worker is building or using. For example, `[0]` means only the first shard will be embeded or used, and `[0,1,2]` means the 0-2 shards will be embeded or used by the current worker.
* `keep_last_chunk`: if set to `true`, the chunked documents that have less than `datastore.chunk_size` will be included in the datastore; these documents will be discarded if `keep_last_chunk` is set to `false`.
* `per_gpu_batch_size` is the batch size per GPU used when running embedding.
* `prefix` is a prefix naming string used to name the outputs.
* `embedding_dir` is the directory to save the embeddings.
* `use_saved_if_exists`: if set to `true`, the embedding task will be skipped if the embedding exists; it will rerun the embedding and overwrite the exisitng ones if `use_saved_if_exists` is set to `false`.

**Index Configuration**

```bash
datastore:
  index:
    index_shard_ids: [[0]]
    indexing_batch_size: 1000000
    projection_size: 768
    overwrite: false
```
* `index_shard_ids`: IDs of passages included in each index. For example, [0, 1, 2] means build a single index over passages 0-2; [[0], [1,2]] means build an index for passage 0 and another index for passages 1-2. This argument is very useful when you want to index or search over different shards in parallel.
* `indexing_batch_size` defines the batch size used for indexing.
* `projection_size` tells the index the size of embedding.
* `overwrite`: if set to `false`, it will save the index and reuse the index if it exists; if set to `true`, the index will be reconstructed and overwrite the old index.

Note: we currently only support flat index. Please stay tuned for more index types.

## Distributed Datastore Construction

## Document Retrieval

## Data Filtering
### Decontamination
### Deduplication

## Subsampling

## Evaluation
### Perplexity Evaluation
### Downstream Evaluation