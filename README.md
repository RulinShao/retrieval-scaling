# Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore

Code and data for paper "Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore".

[[Website](https://retrievalscaling.github.io)][[Paper](https://arxiv.org/abs/2407.12854)]

**Datastores:** [🤗 MassiveDS-1.4T](https://huggingface.co/datasets/rulins/MassiveDS-1.4T) | [🤗 MassiveDS-140B](https://huggingface.co/datasets/rulins/MassiveDS-140B)


<img src=images/scaling_gif.gif width="666" alt="Scaling overview.">

If you find our code, data, models, or the paper useful, please cite the paper:
```
@article{shao2024scaling,
  title={Scaling Retrieval-Based Language Models with a Trillion-Token Datastore},
  author={Shao, Rulin and He, Jacqueline and Asai, Akari and Shi, Weijia and Dettmers, Tim and Min, Sewon and Zettlemoyer, Luke and Koh, Pang Wei},
  journal={arXiv preprint arXiv:2407.12854},
  year={2024}
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
Note: to accelerate the inference, we recommend users to install flash attention based on their accelerator type following the instructions [here](https://github.com/Dao-AILab/flash-attention).

To create a CPU-only environment:
```bash
conda env create -f environment_cpu.yml
conda activate scaling_cpu
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
We release the **data**, **embedding**, and **index** of our MassiveDS datastore, along with its 10% subsampled version for smaller-scale experiments, on HuggingFace:

* **10% Subsampled MassiveDS**: https://huggingface.co/datasets/rulins/MassiveDS-140B
* **Full MassiveDS**: https://huggingface.co/datasets/rulins/MassiveDS-1.4T



# Advanced Usage
We provide more details of advanced usage of our database below.

## Content
1. [Model Configuration](#model-configuration)
2. [Datastore Configuration](#datastore-configuration)
3. [Distributed Datastore Construction](#distributed-datastore-construction)
4. [Distributed Document Retrieval](#distributed-document-retrieval)
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
For datastores with over 1B tokens, we recommend using our distributed datastore construction pipeline, which could linearly accelerate your datastore construction. 

Below is an example slurm job script that parallelize the datastore construction using 16 GPUs.
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint a40|l40
#SBATCH --mem 100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.datastore.embedding=true \
  tasks.datastore.index=true \
  datastore.embedding.num_shards=16 \
  datastore.embedding.shard_ids=[$SLURM_ARRAY_TASK_ID] \
  datastore.index.index_shard_ids=[[$SLURM_ARRAY_TASK_ID]]
```
The above script splits the raw data into 16 shards and builds embeddings and indices for these shards in parallel.

## Distributed Document Retrieval

Since the datastore is sharded, we can run document retrieval in parallel as well. For example, the below slurm script searches the top-K documents from each shard in parallel.

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint a40|l40
#SBATCH --mem 100G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.eval.task_name=perplexity \
  tasks.eval.search=true \
  datastore.embedding.num_shards=16 \
  datastore.index.index_shard_ids=[[$SLURM_ARRAY_TASK_ID]]
```
When the 16 jobs finish, the top-K documents from each shard are saved separately. Next, we can merge the retrieved documents quickly on a single node:
```bash
index_list="[[0]"
for (( i=1; i<=$((16 - 1)); i++ )); do
index_list+=",[$i]"
done
index_list+="]"

PYTHONPATH=.  python ric/main_ric.py --config-name example_config \
  tasks.eval.task_name=perplexity \
  tasks.eval.search=true \
  datastore.embedding.num_shards=16 \
  datastore.index.index_shard_ids=$index_list
```

## Data Filtering
We provide functions to run post-hoc data filtering on the retrieved documents to either improve the retrieval quality or remove contamination data.
### Decontamination
We support two decontamination methods: the longest string overlap based decontamination and the Jaccard similarity based decontamination.

**Longest String Overlap Based Decontamination**

The longest string overlap based decontamination method removes a document from the retrieved top-K if it has a continuous overlapped string that exceeds `n` words with the gold answer. You can set it through

```bash
evaluation:
  decontamination: true
  contamination_threshold: 32
  decontamination_method: longest
```
where `n = 32` in the above case.

**Jaccard Similarity Based Decontamination**

Jaccard similarity based decontamination removes a document from the retrieved top-K if it has a 13-gram Jaccard similarity score that is higher than `t` (0 < `t` < 1) with the gold answer. You can set it through

```bash
evaluation:
  decontamination: true
  contamination_threshold: 0.8
  decontamination_method: jaccard
```
where `p = 0.8` in the above case.

### Deduplication
We provide a post-hoc deduplication method to remove the duplicates in the top-K retrieved documents before taking the top-k (k << K) for evaluation. We use MinHash for deduplication, which marks documents as deplicates if they have a 13-gram Jaccard similarity score over 0.8. For the set of documents that are marked duplicates, we keep the one with the highest retrieval score. You can enable it by setting `evaluation.search.post_process_only=true`.

## Subsampling
We also support running post-hoc subsampling on the retrieved top-K documents, which helps us simulate the sitatuions where the raw data is subsampled to a smaller size with a given random seed. See Appendix A in our paper for more details. 

To run subsampling on the retrieved top-K documents, turn on `evaluation.search.post_process_only`. Then set the subsampling probability through `evaluation.search.topk_subsample_p` and the random seed through `evaluation.search.subsample_seed` after running document retrieval.

## Evaluation
In this section, we provide more details of the evaluation. Here are some general evaluation configurations:
```bash
evaluation:
  domain: test_c4
  concate_k: 3
```
where `evaluation.domain` is the name of the evaluation task and `concate_k` is the number of documents that will be prepended in context.

### Perplexity Evaluation
Perplexity evaluation for retrieval-based LMs is different from perplexity evaluation for LMs. Specifically, given an input, the first part, i.e., the first half in our implementation, is used as the query to retrieve relevant documents, which thus cannot be used for perplexity computation. Therefore, only the second part, i.e., the second half, of the input is used to calculate perplexity. The retrieved top-k documents will be prepended in context before the query in a reversed order, i.e., the document with the highest retrieval score is the cloest to the query. The concatenated context is then fed into the LM and we compute the perplexity using the second half of the input as the target. 

Below is an example configuration that first concatenates all tokens in `eval_data` in one sentence using sequence packaging and then chunks the sentence into smaller chunks of `max_eval_data_seq_length` tokens with a stride of `eval_stride`. For each 1024-token chunk, the first 512 tokens are used as query, and the last 512 tokens are used for perplexity calculation.
```bash
evaluation
  data:
    eval_data: examples/test_c4.jsonl
    max_eval_data_seq_length: 1024
    eval_stride: 512
    merge: True
    num_eval_samples: null 
```

### Downstream Evaluation
We developped a package for retrieval-augmented-generation (RAG) evaluation---RAG-Evaluation-Harness. The package supports over 60 standard academic benchmarks and can be easily used with retrieval augmentation as shown in the [Quick Start](#quick-start). We refer the user to the [README of RAG-Evaluation-Harness](https://github.com/RulinShao/retrieval-scaling/blob/main/rag-evaluation-harness/README.md) for more details, such as accelerating the inference with VLLMs, getting a list of supported tasks, and adding new tasks.
