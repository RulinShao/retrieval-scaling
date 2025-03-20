#!/bin/bash
#SBATCH --job-name=main-api
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         
#SBATCH --hint=multithread   
#SBATCH --account comem
#SBATCH --qos comem_high
#SBATCH --mem 400G         
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/retrieval-scaling
#SBATCH --output=/checkpoint/comem/rulin/cache/slurm/slurm-%A_%a.out
#SBATCH --array=0


cd /checkpoint/comem/rulin/workspace/retrieval-scaling
source /home/rulin/miniconda3/bin/activate
conda activate scaling


PYTHONPATH=.  python api/serve_main_node.py
