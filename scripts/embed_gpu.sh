EVAL_DOMAIN=mmlu_pro:mc::none+gpqa::llama3+agi_eval_english::olmes

N_DOCS=1000

datastores=(rpj_wiki rpj_stackexchange rpj_book rpj_arxiv rpj_github pes2o pubmed math dpr_wiki \
          rpj_c4 rpj_commoncrawl_2019-30 rpj_commoncrawl_2020-05 rpj_commoncrawl_2021-04 rpj_commoncrawl_2022-05 rpj_commoncrawl_2023-06)

start_time=$(date +%s)
for TASK_ID in $(seq 0 14); do

    DS_NAME=${datastores[$TASK_ID]}

    if [ $TASK_ID -lt 9 ]; then
        NUM_SHARDS=8
    else
        NUM_SHARDS=32
    fi

    
    index_list="[[0]"
    for (( i=1; i<=$((NUM_SHARDS - 1)); i++ )); do
    index_list+=",[$i]"
    done
    index_list+="]"
    echo INDEX_IDS:$index_list
    PYTHONPATH=/gscratch/zlab/rulins/scaling-clean  python /gscratch/zlab/rulins/scaling-clean/ric/main_ric.py \
    --config-name largest_default \
    tasks.eval.task_name=lm-eval \
    tasks.eval.search=true \
    datastore.domain=$DS_NAME \
    datastore.embedding.num_shards=$NUM_SHARDS \
    datastore.embedding.shard_ids=[] \
    datastore.index.index_shard_ids=$index_list \
    evaluation.domain=$EVAL_DOMAIN \
    evaluation.data.eval_data=/gscratch/zlab/rulins/workspace/micheal_queries/mmlu_pro\:mc\:\:none+gpqa\:\:llama3+agi_eval_english\:\:olmes.jsonl \
    evaluation.search.n_docs=$N_DOCS \
    evaluation.search.cache_query_embedding=true \
    evaluation.search.cache_query_embedding_only=true \
    evaluation.search.query_embedding_save_path=/gscratch/zlab/rulins/workspace/micheal_queries/query_embeddings.pkl

done
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
