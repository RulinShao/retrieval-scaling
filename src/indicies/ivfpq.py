import os
import json
import random
import logging
import pickle
import time
import glob
from tqdm import tqdm
import pdb
from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from omegaconf import ListConfig
import subprocess

import faiss
import numpy as np
import torch
from transformers import GPTNeoXTokenizerFast

import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.data import fast_load_jsonl_shard

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class IVFPQIndexer(object):

    def __init__(self, 
                index_path,
                embed_paths=[],
                trained_index_path='',
                sample_train_size=1000000,
                prev_index_path=None,
                dimension=768,
                dtype=np.float16,
                ncentroids=4096,
                code_size=64,
                probe=128,
                num_keys_to_add_at_a_time=1000000,
                DSTORE_SIZE_BATCH=51200000
                ):
    
        self.embed_paths = embed_paths  # list of paths where saved the embedding of all shards
        self.index_path = index_path  # path to store the final index
        self.prev_index_path = prev_index_path  # add new data to it instead of training new clusters
        self.trained_index_path = trained_index_path  # path to save the trained index
        self.cuda = True

        self.sample_size = sample_train_size
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.code_size = code_size
        self.probe = probe
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time

        if os.path.exists(index_path):
            print(f"Loading index from {index_path}")
            self.index = faiss.read_index(index_path)
            self.index.nprobe = self.probe
        else:
            if not os.path.exists(self.trained_index_path):
                print ("Training index...")
                sampled_data = self._sample_and_train_index()

            print ("Building index...")
            self.index = self._add_keys(self.index_path, self.prev_index_path if self.prev_index_path is not None else self.trained_index_path)

    def load_embeds(self, shard_id=None):
        all_ids, all_embeds = [], []
        offset = 0
        for i, embed_path in enumerate(self.embed_paths):
            if shard_id is not None and i != shard_id:
                continue
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)
            all_ids.extend([i + offset for i in ids])
            all_embeds.extend(embeddings)
            offset += len(ids)
        all_embeds = np.stack(all_embeds).astype(np.float32)
        datastore_size = len(all_ids)
        return all_embeds
        
    def get_embs(self, indices=None, shard_id=None):
        if indices is not None:
            embs = self.embs[indices]
        elif shard_id is not None:
            embs = self.load_embeds(shard_id)
        return embs

    def search_knn(self, query_embs, k=4096):
        all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        return all_indices, -all_scores

    def get_knn_scores(self, query_emb, indices):
        assert self.embeds is not None
        embs = self.get_embs(indices) # [batch_size, k, dimension]
        scores = - np.sqrt(np.sum((np.expand_dims(query_emb, 1)-embs)**2, -1)) # [batch_size, k]
        return scores

    def _sample_and_train_index(self,):
        print(f"Sampling {self.sample_size} examples from {len(self.embed_paths)} files...")
        per_shard_sample_size = self.sample_size // len(self.embed_paths)
        all_sampled_embs = []
        for embed_path in self.embed_paths:
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                _, embeddings = pickle.load(fin)
            shard_size = len(embeddings)
            print(f"Finished loading, sampling {per_shard_sample_size} from {shard_size} for training...")
            random_samples = np.random.choice(np.arange(shard_size), size=[min(per_shard_sample_size, shard_size)], replace=False)
            sampled_embs = embeddings[random_samples]
            all_sampled_embs.extend(sampled_embs)
        all_sampled_embs = np.stack(all_sampled_embs).astype(np.float32)
        
        print ("Training index...")
        start_time = time.time()
        self._train_index(all_sampled_embs, self.trained_index_path)
        print ("Finish training (%ds)" % (time.time()-start_time))

    def _train_index(self, sampled_embs, trained_index_path):
        quantizer = faiss.IndexFlatL2(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                       self.dimension,
                                       self.ncentroids,
                                       self.code_size,
                                       8)
        start_index.nprobe = self.probe
        np.random.seed(1)

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _add_keys(self, index_path, trained_index_path):
        index = faiss.read_index(trained_index_path)
        start_time = time.time()
        for shard_id in range(len(self.embed_paths)):
            to_add = self.get_embs(shard_id=shard_id).copy()
            index.add(to_add)
            faiss.write_index(index, index_path)
            print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
        print ('Adding took {} s'.format(time.time() - start_time))
        return index


if __name__ == '__main__':
    embed_dir = '/gscratch/zlab/rulins/data/scaling_out/embeddings/akariasai/pes2o_contriever/pes2o_v3/16-shards/'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir)]
    index_path = '/gscratch/zlab/rulins/pes2o_v3_ivfpq_1M_4096_64.index'
    index = IVFPQIndexer(
        index_path,
        embed_paths,
        '/gscratch/scrubbed/rulins/cache/debug',
        )