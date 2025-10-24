#!/bin/bash
#SBATCH --job-name=worker-api
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         
#SBATCH --hint=multithread   
#SBATCH --account comem
#SBATCH --qos comem_high
#SBATCH --mem 1000G
#SBATCH --gres=gpu:1           
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/retrieval-scaling
#SBATCH --output=/checkpoint/comem/rulin/cache/slurm/slurm-%A_%a.out
#SBATCH --array=0-12



cd /checkpoint/comem/rulin/workspace/retrieval-scaling
source /home/rulin/miniconda3/bin/activate
conda activate scaling



if [ $SLURM_ARRAY_TASK_ID -gt 8 ]; then

    NEW_SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID-8-1))  # 9->0
    all_domains=(
        rpj_c4
    )  # 9-12
    export NUM_SHARDS=32
    export NUM_SHARDS_PER_WORKER=8
    num_worker_per_domain=$((NUM_SHARDS / NUM_SHARDS_PER_WORKER))
    ds_index=$((NEW_SLURM_ARRAY_TASK_ID / num_worker_per_domain))
    worker_id=$((NEW_SLURM_ARRAY_TASK_ID % num_worker_per_domain))
    export DS_DOMAIN=${all_domains[ds_index]}
    export WORKER_ID=$worker_id

else

    all_domains=(
        dpr_wiki math pes2o pubmed
        rpj_arxiv rpj_book rpj_github rpj_stackexchange rpj_wikipedia 
    )  # 0-8
    export NUM_SHARDS=8
    export NUM_SHARDS_PER_WORKER=8
    export DS_DOMAIN=${all_domains[SLURM_ARRAY_TASK_ID]}  
    export WORKER_ID=0

fi

PYTHONPATH=.  python api/serve_worker_node.py
