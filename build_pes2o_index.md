# Instruction on Indexing and Searching Over PES2O Data

### Setup Environment (If you haven't)
```bash
git clone https://github.com/RulinShao/retrieval-scaling.git
cd retrieval-scaling
git checkout pes2o

# create a new conda environment and install dependencies
conda env create -f environment.yml
conda activate scaling
```

### Prepare JSONL Data
Our dataloader assumes a jsonl formatted data input. Download and convert the [pes2o v3 data](https://huggingface.co/datasets/allenai/peS2o/tree/main/data/v3) into jsonl format and save the converted jsonl data to a directory.

### Build Index 

```bash
datastore_raw_data_path=$PES2O_V3_JSONL_DIR  # directory that contains the converted jsonl data
num_shards=16

for SLURM_ARRAY_TASK_ID in {0..15}; do  # we recommend to parallelize this with slurm jobs
PYTHONPATH=.  python ric/main_ric.py \
  --config-name=pes2o_v3 \
  tasks.datastore.embedding=true \
  tasks.datastore.index=true \
  datastore.raw_data_path=$datastore_raw_data_path \
  datastore.embedding.num_shards=$num_shards \
  datastore.embedding.shard_ids=[$SLURM_ARRAY_TASK_ID] \
  hydra.job_logging.handlers.file.filename=embedding.log
done
```

### Search
Prepare your queries in a jsonl format with `query` as one key. E.g., `{'query': "Text used as a retrieval query", other_key: other_value}`.
```bash
EVAL_DOMAIN=$EXP_ID  # self-defined exp id to distinguish outpu
RAW_QUERY=$PATH_TO_QUERY_JSONL_FILE

DS_NAME=pes2o_v3
NUM_SHARDS=16
N_DOCS=100

index_list="[[0]"
for (( i=1; i<=$((NUM_SHARDS - 1)); i++ )); do
index_list+=",[$i]"
done
index_list+="]"
echo INDEX_IDS:$index_list

PYTHONPATH=.  python ric/main_ric.py \
    --config-name pes2o_v3 \
    tasks.eval.task_name=lm-eval \
    tasks.eval.search=true \
    datastore.embedding.num_shards=$NUM_SHARDS \
    datastore.embedding.shard_ids=[] \
    datastore.index.index_shard_ids=$index_list \
    evaluation.domain=$EVAL_DOMAIN \
    evaluation.data.eval_data=$RAW_QUERY \
    evaluation.search.n_docs=$N_DOCS \
    evaluation.eval_output_dir=$EVAL_OUTPUT_DIR # where the retrieved documents will be saved
```