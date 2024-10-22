EVAL_DOMAIN=all_queries_250
N_DOCS=2000


python /mnt/md-256k/retrieval-scaling/scripts/write_retrieval_paths_to_txt.py \
    --eval_domain $EVAL_DOMAIN \
    --n_docs $N_DOCS


RERANK_N_DOCS=100
MERGE_TXT=/mnt/md-256k/scaling_out/retrieved_results/post_processed/${EVAL_DOMAIN}_top${N_DOCS}_8shards.txt
BASE_MERGED_PATH=/mnt/md-256k/scaling_out/retrieved_results/post_processed/dedup_merged_${EVAL_DOMAIN}_top${N_DOCS}.jsonl
MERGED_PATH=/mnt/md-256k/scaling_out/retrieved_results/post_processed/full_subsampled_${p}_${seed}_dedup_merged_${EVAL_DOMAIN}_top${N_DOCS}.jsonl
if [ ! -f $MERGED_PATH ]; then
    PYTHONPATH=/mnt/md-256k/retrieval-scaling  python /mnt/md-256k/retrieval-scaling/ric/main_ric.py \
        --config-name largest_default \
        tasks.eval.merge_search=true \
        evaluation.search.merge_multi_source_results=true \
        evaluation.search.n_docs=$N_DOCS \
        evaluation.search.paths_to_merge=$MERGE_TXT \
        evaluation.search.merged_path=$BASE_MERGED_PATH \
        evaluation.search.rerank_n_docs=$RERANK_N_DOCS \
        evaluation.search.use_saved_dedup_data=true
fi